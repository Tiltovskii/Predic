"""Audited importer for Ted Taylor's public CS:GO games dataset.

This adapter deliberately consumes only two downloaded CSV files; it does not
run the upstream repository's scraper and has no HTTP client.  The upstream
repository says its data was scraped from HLTV, so this is a *research
bootstrap* rather than a point-in-time or independently licensed production
feed.  In particular, no source timestamp in these files proves that a feature
was available before a match, and every normalized ``known_at`` is ``NULL``.

The input contracts are intentionally narrow:

* ``historic_games_list.csv`` and ``game_data_rh.csv`` must retain their
  original, exact header layouts;
* each file contributes its first row for an exact ``game_link``;
* exact-link stats rows are joined only to the matching historic row; a valid
  historic-only row remains an auditable map result with no invented lineup;
* the ``mapstatsid`` embedded in that link is the stable map identity.

The source does not carry a trustworthy match/series identifier or player/team
IDs.  Therefore every source map becomes one low-confidence series, teams are
keyed by their source name, and players are scoped to ``(team, nickname)``.
That deliberately prevents an ambiguous nickname on opposite teams from being
merged into one person.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SOURCE = "tedtay-csgo-pro-matches"
# Version changes are intentionally part of the immutable snapshot ID.  A
# stricter parser must not silently re-use a snapshot made by an older parser.
PARSER_VERSION = "tedtay-csgo-pro-matches-v2"
METRIC_VERSION = PARSER_VERSION
DATASET_URL = "https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset"
LICENSE_URL = (
    "https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset/"
    "blob/main/LICENSE.txt"
)
UPSTREAM_COMMIT = "ce8a5f242a768c5698f8068828eeedc4fc134db1"
LICENSE_SHA256 = "85ce7eda4c1d04cba58e5d9852703b1d978b31c196334bd5fa4dbf146136f285"
PROVENANCE = (
    "The upstream repository README says the CSV data were scraped from HLTV "
    "using Selenium and BeautifulSoup. This importer consumes only downloaded "
    "CSV artifacts; it does not execute upstream scripts or access HLTV. "
    "The repository-declared MIT license is recorded for provenance only and "
    "does not verify rights in the CSV data."
)

DEFAULT_BATCH_SIZE = 500
MAX_BATCH_SIZE = 10_000
AMBIGUOUS_TEAM_BINDING_KIND = "tedtay_ambiguous_team_binding"
MISSING_GAME_DATA_KIND = "tedtay_missing_game_data_rh"
QUARANTINE_RECORD_KINDS = (
    AMBIGUOUS_TEAM_BINDING_KIND,
    MISSING_GAME_DATA_KIND,
)
MAX_QUARANTINE_SAMPLES = 20

HISTORIC_FIELDS = (
    "date_ymd",
    "date_unix_iso",
    "date_unix",
    "team1",
    "team2",
    "team1_rounds",
    "team2_rounds",
    "map_name_short",
    "event_name",
    "game_link",
)

_PLAYER_FIELD_SUFFIXES = (
    "name",
    "khs",
    "assists",
    "deaths",
    "kast",
    "kddiff",
    "adr",
    "fkdiff",
    "game_rating",
)

GAME_DATA_FIELDS = (
    "Unnamed: 0",
    "team1_half1_t",
    "team2_half1_ct",
    "team1_half2_ct",
    "team1_half2_t",
    "team1_first_kills",
    "team2_first_kills",
    "team1_clutches_won",
    "team2_clutches_won",
    *tuple(
        f"team{team}_p{slot}_{suffix}"
        for team in (1, 2)
        for slot in range(1, 6)
        for suffix in _PLAYER_FIELD_SUFFIXES
    ),
    "game_link",
    "collected_timestamp",
    "team2_half2_t",
)

_GAME_LINK = re.compile(
    r"^/stats/matches/mapstatsid/([1-9][0-9]*)/[^/?#\s]+$"
)
_PARENTHESIZED_SCORE = re.compile(r"^\(([0-9]{1,4})\)$")
_PAIR_CELL = re.compile(r"^([0-9]{1,4}) \(([0-9]{1,4})\)$")
# Most player cells are small, but ``date_unix`` is a millisecond Unix
# timestamp (currently thirteen digits).  Keep a finite syntactic bound so a
# malformed CSV cannot hand an arbitrarily large integer to Python while still
# accepting the source's documented timestamp representation.
_INTEGER_CELL = re.compile(r"^[+-]?[0-9]{1,18}$")
_PERCENT_CELL = re.compile(r"^([0-9]{1,3}(?:\.[0-9]+)?)%$")
_NUMBER_CELL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)$")


class TedTayImportError(ValueError):
    """Raised when a downloaded source artifact is internally inconsistent."""


@dataclass(frozen=True)
class _Team:
    name: str

    @property
    def team_id(self) -> str:
        return f"tedtay:team:{_digest(self.name.casefold())[:32]}"

    @property
    def source_entity_id(self) -> str:
        return self.name.casefold()


@dataclass(frozen=True)
class _HistoricGame:
    game_link: str
    source_map_id: str
    started_at: str
    event_date: str
    teams: tuple[_Team, _Team]
    score_a: int
    score_b: int
    map_name: str
    event_name: str
    raw_cells: dict[str, str]

    @property
    def map_id(self) -> str:
        return f"tedtay:map:{self.source_map_id}"

    @property
    def series_id(self) -> str:
        # The dataset has only map-stat page IDs.  Never infer a BO3/BO5
        # boundary by matching date/team names: one source map is one series.
        return f"tedtay:series:{self.source_map_id}"


@dataclass(frozen=True)
class _PlayerMetrics:
    nickname: str
    kills: int
    headshots: int
    assists: int
    # Older rows sometimes expose only a total-assists cell.  That is an
    # explicit absence of the flash-assist split, not a zero.
    flash_assists: int | None
    deaths: int
    kast: float | None
    adr: float | None
    rating: float | None
    kddiff: int
    fkdiff: int
    raw_cells: dict[str, str]


@dataclass(frozen=True)
class _StatsBlock:
    source_block: int
    total_score: int
    players: tuple[_PlayerMetrics, ...]


@dataclass(frozen=True)
class _GameStats:
    game_link: str
    source_map_id: str
    # These are only the four regulation-half cells captured by upstream.
    # They establish a team binding for non-tied, non-OT maps, but do not
    # encode the extra rounds needed to safely orient every player table.
    half_column_totals: tuple[int, int]
    raw_cells: dict[str, str]


@dataclass(frozen=True)
class _PlayerPlan:
    team: _Team
    slot: int
    metrics: _PlayerMetrics
    source_block: int
    source_block_score: int
    collected_timestamp: str
    game_link: str
    source_map_id: str

    @property
    def player_id(self) -> str:
        return (
            "tedtay:player:"
            f"{_digest(self.team.team_id, self.metrics.nickname.casefold())[:32]}"
        )

    @property
    def source_entity_id(self) -> str:
        # Nicknames are not global source IDs.  Scoping this alias by team is
        # deliberate: same-screen-name opponents cannot collide.
        return f"{self.team.team_id}\x1f{self.metrics.nickname.casefold()}"

    @property
    def metrics_json(self) -> str:
        return _canonical_json(
            {
                "dataset_url": DATASET_URL,
                "game_link": self.game_link,
                "mapstatsid": self.source_map_id,
                "stats_team_block": self.source_block,
                "stats_team_block_score": self.source_block_score,
                "collected_timestamp_raw": self.collected_timestamp,
                "player_raw_cells": self.metrics.raw_cells,
                "parsed_kddiff": self.metrics.kddiff,
                "parsed_fkdiff": self.metrics.fkdiff,
            }
        )


@dataclass(frozen=True)
class _MapPlan:
    historic: _HistoricGame
    stats: _GameStats | None
    players: tuple[_PlayerPlan, ...]
    ambiguous_binding_reason: str | None = None
    missing_game_data_reason: str | None = None

    @property
    def source_map_id(self) -> str:
        return self.historic.source_map_id

    @property
    def map_id(self) -> str:
        return self.historic.map_id

    @property
    def series_id(self) -> str:
        return self.historic.series_id

    @property
    def teams(self) -> tuple[_Team, _Team]:
        return self.historic.teams

    @property
    def has_ambiguous_team_binding(self) -> bool:
        return self.ambiguous_binding_reason is not None

    @property
    def has_missing_game_data(self) -> bool:
        return self.missing_game_data_reason is not None

    @property
    def is_quarantined(self) -> bool:
        return self.has_ambiguous_team_binding or self.has_missing_game_data

    @property
    def quarantine_kind(self) -> str | None:
        if self.has_ambiguous_team_binding:
            return AMBIGUOUS_TEAM_BINDING_KIND
        if self.has_missing_game_data:
            return MISSING_GAME_DATA_KIND
        return None

    @property
    def quarantine_reason(self) -> str | None:
        if self.has_ambiguous_team_binding:
            return self.ambiguous_binding_reason
        return self.missing_game_data_reason


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _required_text(value: object, field: str, *, max_length: int = 1_000) -> str:
    if not isinstance(value, str):
        raise TedTayImportError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TedTayImportError(f"{field} must be non-empty")
    if len(normalized) > max_length:
        raise TedTayImportError(f"{field} exceeds {max_length} characters")
    return normalized


def _parse_game_link(value: object, field: str = "game_link") -> tuple[str, str]:
    link = _required_text(value, field, max_length=1_000)
    if link != value:
        raise TedTayImportError(f"{field} cannot have edge whitespace")
    match = _GAME_LINK.fullmatch(link)
    if match is None:
        raise TedTayImportError(
            f"{field} must be an exact relative /stats/matches/mapstatsid/<id>/... link"
        )
    return link, match.group(1)


def _parse_integer(value: object, field: str, *, minimum: int | None = None) -> int:
    raw = _required_text(value, field, max_length=128)
    if _INTEGER_CELL.fullmatch(raw) is None:
        raise TedTayImportError(f"{field} must be an integer cell")
    parsed = int(raw)
    if minimum is not None and parsed < minimum:
        raise TedTayImportError(f"{field} must be at least {minimum}")
    return parsed


def _parse_number(value: object, field: str, *, minimum: float | None = None) -> float:
    raw = _required_text(value, field, max_length=128)
    if _NUMBER_CELL.fullmatch(raw) is None:
        raise TedTayImportError(f"{field} must be a decimal number")
    parsed = float(raw)
    if not math.isfinite(parsed):  # defensive; regex has already excluded nan/inf
        raise TedTayImportError(f"{field} must be finite")
    if minimum is not None and parsed < minimum:
        raise TedTayImportError(f"{field} must be at least {minimum}")
    return parsed


def _parse_score(value: object, field: str) -> int:
    raw = _required_text(value, field, max_length=128)
    match = _PARENTHESIZED_SCORE.fullmatch(raw)
    if match is None:
        raise TedTayImportError(f"{field} must be a parenthesized final score")
    return int(match.group(1))


def _parse_pair(value: object, field: str) -> tuple[int, int]:
    raw = _required_text(value, field, max_length=128)
    match = _PAIR_CELL.fullmatch(raw)
    if match is None:
        raise TedTayImportError(f"{field} must have the exact '<total> (<subset>)' form")
    return int(match.group(1)), int(match.group(2))


def _parse_assists(value: object, field: str) -> tuple[int, int | None]:
    """Parse an assists cell without manufacturing a flash-assist count.

    The downloaded corpus has both ``"7 (2)"`` and legacy ``"7"`` forms.
    A plain integer is still a usable total-assist value, but it says nothing
    about flashes, so the normalized nullable column must remain ``NULL``.
    """

    raw = _required_text(value, field, max_length=128)
    pair = _PAIR_CELL.fullmatch(raw)
    if pair is not None:
        return int(pair.group(1)), int(pair.group(2))
    if _INTEGER_CELL.fullmatch(raw) is not None:
        return int(raw), None
    raise TedTayImportError(
        f"{field} must have '<total> (<flash>)' or legacy '<total>' form"
    )


def _parse_percent(value: object, field: str) -> float:
    raw = _required_text(value, field, max_length=128)
    match = _PERCENT_CELL.fullmatch(raw)
    if match is None:
        raise TedTayImportError(f"{field} must be a percentage cell")
    parsed = float(match.group(1))
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise TedTayImportError(f"{field} must be between 0 and 100 percent")
    return parsed


def _parse_optional_percent(value: object, field: str) -> float | None:
    raw = _required_text(value, field, max_length=128)
    if raw == "-":
        return None
    return _parse_percent(raw, field)


def _parse_optional_number(
    value: object, field: str, *, minimum: float | None = None
) -> float | None:
    raw = _required_text(value, field, max_length=128)
    if raw == "-":
        return None
    return _parse_number(raw, field, minimum=minimum)


def _parse_event_date(value: object) -> tuple[str, date]:
    event_date = _required_text(value, "date_ymd", max_length=32)
    try:
        return event_date, date.fromisoformat(event_date)
    except ValueError as error:
        raise TedTayImportError("date_ymd must be an ISO calendar date") from error


def _validate_row_shape(
    row: Mapping[str | None, str | None],
    *,
    expected_fields: tuple[str, ...],
    file_name: str,
    line_number: int,
) -> None:
    if set(row) != set(expected_fields):
        raise TedTayImportError(
            f"{file_name} line {line_number} has a malformed CSV row shape"
        )
    if any(row[field] is None for field in expected_fields):
        raise TedTayImportError(
            f"{file_name} line {line_number} has a missing CSV cell"
        )


def _open_csv(
    path: Path, *, expected_fields: tuple[str, ...]
) -> tuple[object, csv.DictReader]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise TedTayImportError(f"cannot open {path}: {error}") from error
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != expected_fields:
        stream.close()
        raise TedTayImportError(
            f"{path.name} header differs from the audited upstream raw schema"
        )
    return stream, reader


def _parse_historic(row: Mapping[str, str]) -> _HistoricGame:
    game_link, source_map_id = _parse_game_link(row["game_link"])
    event_date, parsed_date = _parse_event_date(row["date_ymd"])
    source_iso = _required_text(row["date_unix_iso"], "date_unix_iso", max_length=64)
    try:
        parsed_iso = datetime.strptime(source_iso, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise TedTayImportError(
            "date_unix_iso must use the upstream YYYY-MM-DD HH:MM:SS form"
        ) from error
    if parsed_iso.date() != parsed_date:
        raise TedTayImportError("date_ymd and date_unix_iso disagree")
    unix_millis = _parse_integer(row["date_unix"], "date_unix", minimum=0)
    try:
        parsed_epoch = datetime.fromtimestamp(unix_millis / 1_000, timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise TedTayImportError("date_unix is outside the supported UTC range") from error
    # ``date_unix_iso`` is an upstream display timestamp without a timezone.
    # In the audited corpus it is either the epoch's UTC rendering or exactly
    # one hour ahead; the millisecond Unix timestamp is the unambiguous event
    # instant that we normalize to UTC.  Reject anything outside that observed
    # source contract instead of silently treating an arbitrary local time as
    # UTC.
    source_offset_seconds = (
        parsed_iso.replace(tzinfo=timezone.utc) - parsed_epoch
    ).total_seconds()
    if source_offset_seconds not in (0.0, 3_600.0):
        raise TedTayImportError(
            "date_unix_iso must be the UTC or UTC+01:00 display of date_unix"
        )

    team_a = _Team(_required_text(row["team1"], "team1", max_length=256))
    team_b = _Team(_required_text(row["team2"], "team2", max_length=256))
    if team_a.team_id == team_b.team_id:
        raise TedTayImportError("a map cannot have the same team on both sides")
    score_a = _parse_score(row["team1_rounds"], "team1_rounds")
    score_b = _parse_score(row["team2_rounds"], "team2_rounds")
    map_name = _required_text(row["map_name_short"], "map_name_short", max_length=128)
    event_name = _required_text(row["event_name"], "event_name", max_length=1_000)
    return _HistoricGame(
        game_link=game_link,
        source_map_id=source_map_id,
        started_at=parsed_epoch.isoformat(),
        event_date=event_date,
        teams=(team_a, team_b),
        score_a=score_a,
        score_b=score_b,
        map_name=map_name,
        event_name=event_name,
        raw_cells={field: row[field] for field in HISTORIC_FIELDS},
    )


def _parse_player_metrics(
    row: Mapping[str, str], *, team_slot: int, player_slot: int
) -> _PlayerMetrics:
    prefix = f"team{team_slot}_p{player_slot}_"
    raw_cells = {
        suffix: row[f"{prefix}{suffix}"] for suffix in _PLAYER_FIELD_SUFFIXES
    }
    nickname = _required_text(raw_cells["name"], f"{prefix}name", max_length=128)
    kills, headshots = _parse_pair(raw_cells["khs"], f"{prefix}khs")
    assists, flash_assists = _parse_assists(raw_cells["assists"], f"{prefix}assists")
    deaths = _parse_integer(raw_cells["deaths"], f"{prefix}deaths", minimum=0)
    kast = _parse_optional_percent(raw_cells["kast"], f"{prefix}kast")
    kddiff = _parse_integer(raw_cells["kddiff"], f"{prefix}kddiff")
    adr = _parse_optional_number(raw_cells["adr"], f"{prefix}adr", minimum=0.0)
    fkdiff = _parse_integer(raw_cells["fkdiff"], f"{prefix}fkdiff")
    rating = _parse_optional_number(
        raw_cells["game_rating"], f"{prefix}game_rating", minimum=0.0
    )
    if headshots > kills:
        raise TedTayImportError(f"{prefix}khs has more headshots than kills")
    if flash_assists is not None and flash_assists > assists:
        raise TedTayImportError(f"{prefix}assists has more flash assists than assists")
    if kills - deaths != kddiff:
        raise TedTayImportError(f"{prefix}kddiff disagrees with kills and deaths")
    return _PlayerMetrics(
        nickname=nickname,
        kills=kills,
        headshots=headshots,
        assists=assists,
        flash_assists=flash_assists,
        deaths=deaths,
        kast=kast,
        adr=adr,
        rating=rating,
        kddiff=kddiff,
        fkdiff=fkdiff,
        raw_cells=raw_cells,
    )


def _parse_stats_block(
    row: Mapping[str, str], *, source_block: int, total_score: int
) -> _StatsBlock:
    players = tuple(
        _parse_player_metrics(row, team_slot=source_block, player_slot=slot)
        for slot in range(1, 6)
    )
    nicknames = [player.nickname.casefold() for player in players]
    if len(set(nicknames)) != 5:
        raise TedTayImportError(
            f"team{source_block} lineup must contain exactly five distinct nicknames"
        )
    return _StatsBlock(
        source_block=source_block,
        total_score=total_score,
        players=players,
    )


def _parse_game_stats(row: Mapping[str, str]) -> _GameStats:
    game_link, source_map_id = _parse_game_link(row["game_link"])
    # These raw headings are awkward but intentional.  In the upstream
    # script, the first two half-pairs are emitted as (team1, team2), so the
    # scores for its two player-stat tables are respectively 1+3 and 2+4.
    half_1_a = _parse_integer(row["team1_half1_t"], "team1_half1_t", minimum=0)
    half_1_b = _parse_integer(row["team2_half1_ct"], "team2_half1_ct", minimum=0)
    half_2_a = _parse_integer(row["team1_half2_ct"], "team1_half2_ct", minimum=0)
    half_2_b = _parse_integer(row["team1_half2_t"], "team1_half2_t", minimum=0)
    # Preserve the spare upstream column exactly, rather than treating a blank
    # as a guessed third-half/OT score.
    if row["team2_half2_t"] is None:
        raise TedTayImportError("team2_half2_t cannot be a missing CSV cell")
    return _GameStats(
        game_link=game_link,
        source_map_id=source_map_id,
        half_column_totals=(half_1_a + half_2_a, half_1_b + half_2_b),
        raw_cells={field: row[field] for field in GAME_DATA_FIELDS},
    )


def _plan_map(historic: _HistoricGame, stats: _GameStats) -> _MapPlan:
    if historic.game_link != stats.game_link:
        raise TedTayImportError("inner-joined game links must match exactly")
    if historic.source_map_id != stats.source_map_id:
        raise TedTayImportError("one exact game_link cannot bind two mapstats IDs")
    first_total, second_total = stats.half_column_totals
    # A tied final score is valid source data, but it cannot tell us which of
    # the two anonymous player tables belongs to which historical team.  Do
    # not use link order, roster history, or another heuristic to fill that
    # gap.  The caller stores the full joined raw payload for later evidence.
    if historic.score_a == historic.score_b:
        return _MapPlan(
            historic=historic,
            stats=stats,
            players=(),
            ambiguous_binding_reason="tied_final_score",
        )
    if (first_total, second_total) == (historic.score_a, historic.score_b):
        stat_teams = (historic.teams[0], historic.teams[1])
    elif (first_total, second_total) == (historic.score_b, historic.score_a):
        # The two CSVs do not promise the same team ordering.  Reconcile it
        # only through the independently present final score; do not guess by
        # player nickname or page slug.
        stat_teams = (historic.teams[1], historic.teams[0])
    else:
        # Overtime results are the known common case: the downloaded fields
        # retain only two regulation halves, so both totals can be 15 while
        # the final is, for example, 25:23.  This is usable map-level data but
        # not evidence for assigning table #1/#2 to a team.
        return _MapPlan(
            historic=historic,
            stats=stats,
            players=(),
            ambiguous_binding_reason="four_half_totals_do_not_resolve_team_binding",
        )

    collected_timestamp = _required_text(
        stats.raw_cells["collected_timestamp"],
        "collected_timestamp",
        max_length=128,
    )
    blocks = (
        _parse_stats_block(
            stats.raw_cells, source_block=1, total_score=first_total
        ),
        _parse_stats_block(
            stats.raw_cells, source_block=2, total_score=second_total
        ),
    )

    player_plans: list[_PlayerPlan] = []
    for block, team in zip(blocks, stat_teams, strict=True):
        for slot, metrics in enumerate(block.players, start=1):
            player_plans.append(
                _PlayerPlan(
                    team=team,
                    slot=slot,
                    metrics=metrics,
                    source_block=block.source_block,
                    source_block_score=block.total_score,
                    collected_timestamp=collected_timestamp,
                    game_link=historic.game_link,
                    source_map_id=historic.source_map_id,
                )
            )
    if len(player_plans) != 10:
        raise TedTayImportError("a joined source map must have exactly ten player rows")
    for team in historic.teams:
        if sum(player.team.team_id == team.team_id for player in player_plans) != 5:
            raise TedTayImportError("a joined source map must have exactly five players per team")
    return _MapPlan(historic=historic, stats=stats, players=tuple(player_plans))


def _plan_missing_game_data(historic: _HistoricGame) -> _MapPlan:
    """Retain a valid historic map whose first stats row is absent.

    The historic file alone proves the map result and two participants, but it
    cannot prove a five-player lineup.  Keep the source row in an immutable
    quarantine record rather than manufacture roster evidence.
    """

    return _MapPlan(
        historic=historic,
        stats=None,
        players=(),
        missing_game_data_reason="game_link_absent_from_game_data_rh",
    )


def _create_temp_tables(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS temp.tedtay_historic_first")
    connection.execute("DROP TABLE IF EXISTS temp.tedtay_stats_first")
    connection.execute(
        """
        CREATE TEMP TABLE tedtay_historic_first (
            game_link TEXT PRIMARY KEY,
            source_map_id TEXT NOT NULL,
            event_date TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL UNIQUE,
            row_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE tedtay_stats_first (
            game_link TEXT PRIMARY KEY,
            source_map_id TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL UNIQUE,
            row_json TEXT NOT NULL
        )
        """
    )


def _stage_historic(
    connection: sqlite3.Connection, path: Path
) -> tuple[int, int]:
    stream, reader = _open_csv(path, expected_fields=HISTORIC_FIELDS)
    first_rows = 0
    duplicates = 0
    try:
        for ordinal, raw_row in enumerate(reader, start=1):
            line_number = ordinal + 1
            _validate_row_shape(
                raw_row,
                expected_fields=HISTORIC_FIELDS,
                file_name=path.name,
                line_number=line_number,
            )
            row = {field: str(raw_row[field]) for field in HISTORIC_FIELDS}
            game_link, source_map_id = _parse_game_link(row["game_link"])
            if connection.execute(
                "SELECT 1 FROM temp.tedtay_historic_first WHERE game_link = ?",
                (game_link,),
            ).fetchone() is not None:
                duplicates += 1
                continue
            # The source begins with two pre-2018 rows whose unfinished-score
            # syntax is unusable for a map result.  ``from_date`` is an
            # import filter, so stage only exact link/date provenance here and
            # defer score/lineup semantics until a row is joined and eligible.
            event_date, _ = _parse_event_date(row["date_ymd"])
            connection.execute(
                """
                INSERT INTO temp.tedtay_historic_first (
                    game_link, source_map_id, event_date, row_ordinal, row_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    game_link,
                    source_map_id,
                    event_date,
                    ordinal,
                    _canonical_json(row),
                ),
            )
            first_rows += 1
    finally:
        stream.close()
    return first_rows, duplicates


def _stage_stats(connection: sqlite3.Connection, path: Path) -> tuple[int, int]:
    stream, reader = _open_csv(path, expected_fields=GAME_DATA_FIELDS)
    first_rows = 0
    duplicates = 0
    try:
        for ordinal, raw_row in enumerate(reader, start=1):
            line_number = ordinal + 1
            _validate_row_shape(
                raw_row,
                expected_fields=GAME_DATA_FIELDS,
                file_name=path.name,
                line_number=line_number,
            )
            row = {field: str(raw_row[field]) for field in GAME_DATA_FIELDS}
            game_link, source_map_id = _parse_game_link(row["game_link"])
            if connection.execute(
                "SELECT 1 FROM temp.tedtay_stats_first WHERE game_link = ?",
                (game_link,),
            ).fetchone() is not None:
                duplicates += 1
                continue
            connection.execute(
                """
                INSERT INTO temp.tedtay_stats_first (
                    game_link, source_map_id, row_ordinal, row_json
                ) VALUES (?, ?, ?, ?)
                """,
                (game_link, source_map_id, ordinal, _canonical_json(row)),
            )
            first_rows += 1
    finally:
        stream.close()
    return first_rows, duplicates


def _assert_no_staged_rebinding(connection: sqlite3.Connection, table: str) -> None:
    row = connection.execute(
        f"""
        SELECT source_map_id, COUNT(*) AS links
        FROM temp.{table}
        GROUP BY source_map_id
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if row is not None:
        raise TedTayImportError(
            f"stable mapstatsid {row['source_map_id']} is bound to multiple game_link values "
            f"in {table}"
        )


def _snapshot_id(historic_hash: str, stats_hash: str) -> str:
    # Include parser semantics: a tightened adapter is a different immutable
    # interpretation of exactly the same bronze bytes.
    return f"tedtay:{_digest(PARSER_VERSION, historic_hash, stats_hash)[:32]}"


def _license_ref() -> str:
    return (
        "Repository-declared MIT license (CSV data rights unverified); "
        f"upstream commit: {UPSTREAM_COMMIT}; LICENSE sha256: "
        f"{LICENSE_SHA256}; license: {LICENSE_URL}; provenance: {PROVENANCE}"
    )


def _ensure_snapshot(
    connection: sqlite3.Connection,
    *,
    historic_path: Path,
    historic_hash: str,
    stats_path: Path,
    stats_hash: str,
) -> tuple[str, str]:
    content_hash = _digest("tedtay-raw-pair-v2", historic_hash, stats_hash)
    snapshot_id = _snapshot_id(historic_hash, stats_hash)
    observed_at = _utc_now()
    metadata = _canonical_json(
        {
            "dataset_url": DATASET_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "repository_declared_license": "MIT",
            "dataset_rights_verified": False,
            "license_url": LICENSE_URL,
            "license_sha256": LICENSE_SHA256,
            "license_scope": (
                "Repository-declared license only; it does not assert that "
                "the CSV data themselves are licensed under MIT."
            ),
            "upstream_provenance": PROVENANCE,
            "import_scope": "research_bootstrap_only",
            "point_in_time_note": (
                "Downloaded historical rows do not establish feature availability "
                "before an event; all normalized known_at values are NULL."
            ),
            "input_files": {
                "historic_games_list.csv": {
                    "path": str(historic_path),
                    "sha256": historic_hash,
                },
                "game_data_rh.csv": {
                    "path": str(stats_path),
                    "sha256": stats_hash,
                },
            },
        }
    )
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO source_snapshot (
                snapshot_id, source, source_locator, source_revision, observed_at,
                content_sha256, parser_version, license_ref,
                point_in_time_eligible, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                snapshot_id,
                SOURCE,
                DATASET_URL,
                UPSTREAM_COMMIT,
                observed_at,
                content_hash,
                PARSER_VERSION,
                _license_ref(),
                metadata,
            ),
        )
    row = connection.execute(
        """
        SELECT source, source_locator, source_revision, content_sha256,
               parser_version, license_ref, point_in_time_eligible, observed_at,
               metadata_json
        FROM source_snapshot WHERE snapshot_id = ?
        """,
        (snapshot_id,),
    ).fetchone()
    if row is None:  # pragma: no cover - defensive SQLite invariant
        raise TedTayImportError("failed to create the source snapshot")
    if (
        row["source"] != SOURCE
        or row["source_locator"] != DATASET_URL
        or row["source_revision"] != UPSTREAM_COMMIT
        or row["content_sha256"] != content_hash
        or row["parser_version"] != PARSER_VERSION
        or row["license_ref"] != _license_ref()
        or row["point_in_time_eligible"] != 0
    ):
        raise TedTayImportError("an existing source snapshot has incompatible provenance")
    try:
        persisted_metadata = json.loads(str(row["metadata_json"]))
    except json.JSONDecodeError as error:
        raise TedTayImportError("an existing source snapshot has malformed provenance") from error
    input_files = persisted_metadata.get("input_files")
    if (
        persisted_metadata.get("dataset_url") != DATASET_URL
        or persisted_metadata.get("upstream_commit") != UPSTREAM_COMMIT
        or persisted_metadata.get("repository_declared_license") != "MIT"
        or persisted_metadata.get("dataset_rights_verified") is not False
        or persisted_metadata.get("license_url") != LICENSE_URL
        or persisted_metadata.get("license_sha256") != LICENSE_SHA256
        or persisted_metadata.get("import_scope") != "research_bootstrap_only"
        or not isinstance(input_files, dict)
        or input_files.get("historic_games_list.csv", {}).get("sha256")
        != historic_hash
        or input_files.get("game_data_rh.csv", {}).get("sha256") != stats_hash
    ):
        raise TedTayImportError("an existing source snapshot has incompatible provenance")
    return snapshot_id, str(row["observed_at"])


def _change_count(
    connection: sqlite3.Connection, statement: str, values: tuple[Any, ...]
) -> int:
    before = connection.total_changes
    connection.execute(statement, values)
    return connection.total_changes - before


def _alias_conflict(
    connection: sqlite3.Connection,
    *,
    entity_type: str,
    source_entity_id: str,
    canonical_entity_id: str,
) -> bool:
    rows = connection.execute(
        """
        SELECT canonical_entity_id
        FROM entity_alias
        WHERE source = ? AND entity_type = ? AND source_entity_id = ?
        """,
        (SOURCE, entity_type, source_entity_id),
    ).fetchall()
    return any(str(row["canonical_entity_id"]) != canonical_entity_id for row in rows)


def _ensure_team(
    connection: sqlite3.Connection,
    team: _Team,
    snapshot_id: str,
    inserted: dict[str, int],
) -> None:
    current = connection.execute(
        "SELECT canonical_name FROM team_core WHERE team_id = ?", (team.team_id,)
    ).fetchone()
    if current is not None and str(current["canonical_name"]).casefold() != team.name.casefold():
        raise TedTayImportError(f"team identity collision for {team.team_id}")
    if _alias_conflict(
        connection,
        entity_type="team",
        source_entity_id=team.source_entity_id,
        canonical_entity_id=team.team_id,
    ):
        raise TedTayImportError(f"team alias {team.name!r} is already rebound")
    inserted["teams"] += _change_count(
        connection,
        """
        INSERT OR IGNORE INTO team_core (
            team_id, canonical_name, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'low', ?)
        """,
        (team.team_id, team.name, snapshot_id),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES (?, 'team', ?, ?, ?)
        """,
        (SOURCE, team.source_entity_id, team.team_id, snapshot_id),
    )


def _ensure_player(
    connection: sqlite3.Connection,
    player: _PlayerPlan,
    snapshot_id: str,
    inserted: dict[str, int],
) -> None:
    current = connection.execute(
        "SELECT canonical_nickname FROM player WHERE player_id = ?", (player.player_id,)
    ).fetchone()
    if (
        current is not None
        and str(current["canonical_nickname"]).casefold()
        != player.metrics.nickname.casefold()
    ):
        raise TedTayImportError(f"player identity collision for {player.player_id}")
    if _alias_conflict(
        connection,
        entity_type="player",
        source_entity_id=player.source_entity_id,
        canonical_entity_id=player.player_id,
    ):
        raise TedTayImportError(
            f"scoped player alias {player.source_entity_id!r} is already rebound"
        )
    inserted["players"] += _change_count(
        connection,
        """
        INSERT OR IGNORE INTO player (
            player_id, canonical_nickname, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'low', ?)
        """,
        (player.player_id, player.metrics.nickname, snapshot_id),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES (?, 'player', ?, ?, ?)
        """,
        (SOURCE, player.source_entity_id, player.player_id, snapshot_id),
    )


def _winner_team_id(historic: _HistoricGame) -> str | None:
    if historic.score_a > historic.score_b:
        return historic.teams[0].team_id
    if historic.score_b > historic.score_a:
        return historic.teams[1].team_id
    # A valid draw has no winner.  It remains a map/series result, but cannot
    # be used unchanged as a binary win label.
    return None


def _series_expectation(
    plan: _MapPlan, *, snapshot_id: str, observed_at: str
) -> dict[str, object | None]:
    historic = plan.historic
    return {
        "series_id": plan.series_id,
        "source": SOURCE,
        "source_series_id": historic.source_map_id,
        "scheduled_at": None,
        "started_at": historic.started_at,
        "ended_at": None,
        "known_at": None,
        "observed_at": observed_at,
        "best_of": None,
        "lan_online": None,
        "event_name": historic.event_name,
        "stage_name": None,
        "status": "finished",
        "winner_team_id": _winner_team_id(historic),
        "identity_confidence": "low",
        "source_snapshot_id": snapshot_id,
    }


def _map_expectation(
    plan: _MapPlan, *, snapshot_id: str, observed_at: str
) -> dict[str, object | None]:
    historic = plan.historic
    return {
        "map_id": plan.map_id,
        "series_id": plan.series_id,
        "source_map_id": historic.source_map_id,
        "map_order": 1,
        "map_name": historic.map_name,
        "game_version": "CSGO",
        "ruleset": "UNKNOWN",
        "started_at": historic.started_at,
        "ended_at": None,
        "known_at": None,
        "observed_at": observed_at,
        "team_a_id": historic.teams[0].team_id,
        "team_b_id": historic.teams[1].team_id,
        "score_a": historic.score_a,
        "score_b": historic.score_b,
        "winner_team_id": _winner_team_id(historic),
        "picked_by_team_id": None,
        "legacy_target": None,
        "source_snapshot_id": snapshot_id,
    }


def _lineup_expectation(
    plan: _MapPlan, player: _PlayerPlan, *, snapshot_id: str
) -> dict[str, object | None]:
    return {
        "map_id": plan.map_id,
        "team_id": player.team.team_id,
        "player_id": player.player_id,
        "slot": player.slot,
        "role": None,
        "member_type": "starter",
        "announced_at": None,
        "known_at": None,
        "actual_at": plan.historic.started_at,
        "source_snapshot_id": snapshot_id,
    }


def _stats_expectation(
    plan: _MapPlan,
    player: _PlayerPlan,
    *,
    snapshot_id: str,
    observed_at: str,
) -> dict[str, object | None]:
    metrics = player.metrics
    return {
        "map_id": plan.map_id,
        "team_id": player.team.team_id,
        "player_id": player.player_id,
        "side": "BOTH",
        "metric_version": METRIC_VERSION,
        "known_at": None,
        "observed_at": observed_at,
        "kills": metrics.kills,
        "deaths": metrics.deaths,
        "assists": metrics.assists,
        "flash_assists": metrics.flash_assists,
        "headshots": metrics.headshots,
        "traded_deaths": None,
        "opening_kills": None,
        "opening_deaths": None,
        "adr": metrics.adr,
        "kast": metrics.kast,
        "kpr": None,
        "dpr": None,
        "swing": None,
        "rating": metrics.rating,
        "metrics_json": player.metrics_json,
        "source_snapshot_id": snapshot_id,
    }


def _rows_match_exactly(
    rows: Iterable[sqlite3.Row], expected_rows: Iterable[Mapping[str, object | None]]
) -> bool:
    """Compare all schema fields provided by the importer, not just keys.

    It deliberately includes nullable/defaulted columns.  This turns a rerun
    into an integrity assertion rather than an ``INSERT OR IGNORE`` no-op.
    """

    def normalized(rows_to_normalize: Iterable[Mapping[str, object | None]]) -> set[tuple[tuple[str, object | None], ...]]:
        return {
            tuple((field, row[field]) for field in sorted(row))
            for row in rows_to_normalize
        }

    actual = normalized(dict(row) for row in rows)
    expected = normalized(expected_rows)
    return actual == expected


def _quarantine_stream(kind: str, snapshot_id: str) -> str:
    if kind == AMBIGUOUS_TEAM_BINDING_KIND:
        return f"tedtay-ambiguous-team-binding:{snapshot_id}"
    if kind == MISSING_GAME_DATA_KIND:
        return f"tedtay-missing-game-data-rh:{snapshot_id}"
    raise TedTayImportError(f"unknown TedTay quarantine kind {kind!r}")


def _quarantine_payload(plan: _MapPlan) -> str:
    kind = plan.quarantine_kind
    reason = plan.quarantine_reason
    if kind is None or reason is None:  # pragma: no cover - internal invariant
        raise TedTayImportError("only a quarantined map can have a raw payload")
    historic = plan.historic
    payload: dict[str, object] = {
        "adapter": PARSER_VERSION,
        "dataset_url": DATASET_URL,
        "record_kind": kind,
        "reason": reason,
        "game_link": historic.game_link,
        "mapstatsid": historic.source_map_id,
        "historical_final_score": [historic.score_a, historic.score_b],
        "historic_games_list_raw_cells": historic.raw_cells,
    }
    if kind == AMBIGUOUS_TEAM_BINDING_KIND:
        if plan.stats is None:  # pragma: no cover - internal invariant
            raise TedTayImportError("ambiguous binding requires a stats source row")
        payload["four_half_column_totals"] = list(plan.stats.half_column_totals)
        payload["game_data_rh_raw_cells"] = plan.stats.raw_cells
    return _canonical_json(payload)


def _quarantine_raw_expectation(
    plan: _MapPlan, *, snapshot_id: str, observed_at: str
) -> dict[str, str | None]:
    kind = plan.quarantine_kind
    if kind is None:  # pragma: no cover - internal invariant
        raise TedTayImportError("only a quarantined map can have a raw record")
    payload_json = _quarantine_payload(plan)
    return {
        "raw_record_id": (
            f"tedtay:raw:{kind}:{_digest(snapshot_id, kind, plan.source_map_id)[:32]}"
        ),
        "source": SOURCE,
        "stream": _quarantine_stream(kind, snapshot_id),
        "source_record_id": plan.source_map_id,
        "record_kind": kind,
        "event_at": plan.historic.started_at,
        "known_at": None,
        "observed_at": observed_at,
        "content_sha256": _sha256_text(payload_json),
        "payload_json": payload_json,
        "source_snapshot_id": snapshot_id,
    }


def _quarantine_label(plan: _MapPlan) -> str:
    if plan.quarantine_kind == AMBIGUOUS_TEAM_BINDING_KIND:
        return "ambiguous-team-binding"
    if plan.quarantine_kind == MISSING_GAME_DATA_KIND:
        return "missing-game-data-rh"
    return "quarantine"


def _load_quarantine_raw_index(
    connection: sqlite3.Connection,
) -> dict[str, set[str]]:
    """Load both immutable quarantine namespaces once per import invocation."""

    index: dict[str, set[str]] = {}
    placeholders = ", ".join("?" for _ in QUARANTINE_RECORD_KINDS)
    rows = connection.execute(
        f"""
        SELECT source_record_id, raw_record_id
        FROM raw_ingest_record
        WHERE source = ? AND record_kind IN ({placeholders})
        """,
        (SOURCE, *QUARANTINE_RECORD_KINDS),
    ).fetchall()
    for row in rows:
        source_map_id = str(row["source_record_id"])
        index.setdefault(source_map_id, set()).add(str(row["raw_record_id"]))
    return index


def _assert_exact_quarantine_raw(
    connection: sqlite3.Connection,
    plan: _MapPlan,
    *,
    snapshot_id: str,
    observed_at: str,
    raw_index: Mapping[str, set[str]],
) -> None:
    expected = _quarantine_raw_expectation(
        plan, snapshot_id=snapshot_id, observed_at=observed_at
    )
    raw_record_id = str(expected["raw_record_id"])
    label = _quarantine_label(plan)
    if raw_index.get(plan.source_map_id) != {raw_record_id}:
        raise TedTayImportError(
            f"stable mapstatsid {plan.source_map_id} has incompatible or missing "
            f"{label} raw record"
        )
    row = connection.execute(
        """
        SELECT raw_record_id, source, stream, source_record_id, record_kind,
               event_at, known_at, observed_at, content_sha256, payload_json,
               source_snapshot_id
        FROM raw_ingest_record WHERE raw_record_id = ?
        """,
        (raw_record_id,),
    ).fetchone()
    if row is None or dict(row) != expected:
        raise TedTayImportError(
            f"stable mapstatsid {plan.source_map_id} has incompatible or missing "
            f"{label} raw record"
        )


def _assert_no_quarantine_raw(
    source_map_id: str, raw_index: Mapping[str, set[str]]
) -> None:
    if raw_index.get(source_map_id):
        raise TedTayImportError(
            f"stable mapstatsid {source_map_id} changed between quarantined and "
            "team-bound representations"
        )


def _assert_existing_plan(
    connection: sqlite3.Connection,
    plan: _MapPlan,
    *,
    snapshot_id: str,
    observed_at: str,
    raw_index: Mapping[str, set[str]],
) -> bool:
    """Return true for an exact replay; reject any stable-ID rebind.

    All expected row-level relationships are checked before an insert.  This
    prevents an interrupted/manual partial load or a later revision from
    silently retaining a different normalized relation.
    """

    historic = plan.historic
    source_rows = connection.execute(
        """
        SELECT mg.map_id
        FROM map_game AS mg
        JOIN series AS s ON s.series_id = mg.series_id
        WHERE s.source = ? AND mg.source_map_id = ?
        """,
        (SOURCE, historic.source_map_id),
    ).fetchall()
    if any(str(row["map_id"]) != plan.map_id for row in source_rows):
        raise TedTayImportError(
            f"stable mapstatsid {historic.source_map_id} is already rebound to another map_id"
        )

    map_row = connection.execute(
        "SELECT * FROM map_game WHERE map_id = ?", (plan.map_id,)
    ).fetchone()
    if map_row is None:
        dangling_series = connection.execute(
            "SELECT 1 FROM series WHERE series_id = ?", (plan.series_id,)
        ).fetchone()
        if dangling_series is not None:
            raise TedTayImportError(
                f"{plan.series_id} exists without its stable map; refusing partial rebinding"
            )
        if plan.is_quarantined:
            # A raw record without the transactional parent map proves a
            # partial/manual load.  Never attach it to a new relation.
            if raw_index.get(plan.source_map_id):
                raise TedTayImportError(
                    f"stable mapstatsid {plan.source_map_id} has an orphan "
                    "quarantine raw record"
                )
        else:
            _assert_no_quarantine_raw(plan.source_map_id, raw_index)
        return False

    if dict(map_row) != _map_expectation(
        plan, snapshot_id=snapshot_id, observed_at=observed_at
    ):
        raise TedTayImportError(
            f"stable mapstatsid {historic.source_map_id} is already bound to a "
            "different map relation"
        )

    series_row = connection.execute(
        "SELECT * FROM series WHERE series_id = ?",
        (plan.series_id,),
    ).fetchone()
    if series_row is None or dict(series_row) != _series_expectation(
        plan, snapshot_id=snapshot_id, observed_at=observed_at
    ):
        raise TedTayImportError(
            f"stable series {plan.series_id} has incompatible source metadata"
        )

    participants = connection.execute(
        "SELECT * FROM series_participant WHERE series_id = ?",
        (plan.series_id,),
    ).fetchall()
    expected_participants = (
        {
            "series_id": plan.series_id,
            "team_id": plan.teams[0].team_id,
            "team_slot": 1,
            "source_snapshot_id": snapshot_id,
        },
        {
            "series_id": plan.series_id,
            "team_id": plan.teams[1].team_id,
            "team_slot": 2,
            "source_snapshot_id": snapshot_id,
        },
    )
    if not _rows_match_exactly(participants, expected_participants):
        raise TedTayImportError(
            f"stable series {plan.series_id} has incompatible participants"
        )

    lineup_rows = connection.execute(
        "SELECT * FROM lineup_member WHERE map_id = ?",
        (plan.map_id,),
    ).fetchall()
    stat_rows = connection.execute(
        "SELECT * FROM player_map_stats WHERE map_id = ?",
        (plan.map_id,),
    ).fetchall()
    if plan.is_quarantined:
        if lineup_rows or stat_rows:
            raise TedTayImportError(
                f"stable mapstatsid {historic.source_map_id} must have zero "
                "team-bound lineup and player-stat rows"
            )
        _assert_exact_quarantine_raw(
            connection,
            plan,
            snapshot_id=snapshot_id,
            observed_at=observed_at,
            raw_index=raw_index,
        )
        return True

    _assert_no_quarantine_raw(plan.source_map_id, raw_index)
    expected_lineups = tuple(
        _lineup_expectation(plan, player, snapshot_id=snapshot_id)
        for player in plan.players
    )
    if not _rows_match_exactly(lineup_rows, expected_lineups):
        raise TedTayImportError(
            f"stable mapstatsid {historic.source_map_id} has incompatible or partial lineup rows"
        )

    expected_stats = tuple(
        _stats_expectation(
            plan,
            player,
            snapshot_id=snapshot_id,
            observed_at=observed_at,
        )
        for player in plan.players
    )
    if not _rows_match_exactly(stat_rows, expected_stats):
        raise TedTayImportError(
            f"stable mapstatsid {historic.source_map_id} has incompatible or partial player stats"
        )
    return True


def _insert_quarantine_raw(
    connection: sqlite3.Connection,
    plan: _MapPlan,
    *,
    snapshot_id: str,
    observed_at: str,
    inserted: dict[str, int],
    raw_index: dict[str, set[str]],
) -> None:
    expected = _quarantine_raw_expectation(
        plan, snapshot_id=snapshot_id, observed_at=observed_at
    )
    changes = _change_count(
        connection,
        """
        INSERT INTO raw_ingest_record (
            raw_record_id, source, stream, source_record_id, record_kind,
            event_at, known_at, observed_at, content_sha256, payload_json,
            source_snapshot_id
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
        """,
        (
            expected["raw_record_id"],
            expected["source"],
            expected["stream"],
            expected["source_record_id"],
            expected["record_kind"],
            expected["event_at"],
            expected["observed_at"],
            expected["content_sha256"],
            expected["payload_json"],
            expected["source_snapshot_id"],
        ),
    )
    inserted["quarantine_raw_records"] += changes
    if plan.quarantine_kind == AMBIGUOUS_TEAM_BINDING_KIND:
        inserted["ambiguous_team_binding_raw_records"] += changes
    elif plan.quarantine_kind == MISSING_GAME_DATA_KIND:
        inserted["missing_game_data_rh_raw_records"] += changes
    raw_index.setdefault(plan.source_map_id, set()).add(
        str(expected["raw_record_id"])
    )


def _insert_new_plan(
    connection: sqlite3.Connection,
    plan: _MapPlan,
    *,
    snapshot_id: str,
    observed_at: str,
    inserted: dict[str, int],
    raw_index: dict[str, set[str]],
) -> None:
    historic = plan.historic
    for team in historic.teams:
        _ensure_team(connection, team, snapshot_id, inserted)
    if not plan.is_quarantined:
        for player in plan.players:
            _ensure_player(connection, player, snapshot_id, inserted)

    winner_team_id = _winner_team_id(historic)
    inserted["series"] += _change_count(
        connection,
        """
        INSERT INTO series (
            series_id, source, source_series_id, started_at, known_at,
            observed_at, event_name, status, winner_team_id,
            identity_confidence, source_snapshot_id
        ) VALUES (?, ?, ?, ?, NULL, ?, ?, 'finished', ?, 'low', ?)
        """,
        (
            plan.series_id,
            SOURCE,
            historic.source_map_id,
            historic.started_at,
            observed_at,
            historic.event_name,
            winner_team_id,
            snapshot_id,
        ),
    )
    for slot, team in enumerate(historic.teams, start=1):
        inserted["series_participants"] += _change_count(
            connection,
            """
            INSERT INTO series_participant (
                series_id, team_id, team_slot, source_snapshot_id
            ) VALUES (?, ?, ?, ?)
            """,
            (plan.series_id, team.team_id, slot, snapshot_id),
        )

    inserted["maps"] += _change_count(
        connection,
        """
        INSERT INTO map_game (
            map_id, series_id, source_map_id, map_order, map_name,
            game_version, ruleset, started_at, known_at, observed_at,
            team_a_id, team_b_id, score_a, score_b, winner_team_id,
            source_snapshot_id
        ) VALUES (?, ?, ?, 1, ?, 'CSGO', 'UNKNOWN', ?, NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plan.map_id,
            plan.series_id,
            historic.source_map_id,
            historic.map_name,
            historic.started_at,
            observed_at,
            historic.teams[0].team_id,
            historic.teams[1].team_id,
            historic.score_a,
            historic.score_b,
            winner_team_id,
            snapshot_id,
        ),
    )
    if plan.is_quarantined:
        _insert_quarantine_raw(
            connection,
            plan,
            snapshot_id=snapshot_id,
            observed_at=observed_at,
            inserted=inserted,
            raw_index=raw_index,
        )
        return

    for player in plan.players:
        inserted["lineup_members"] += _change_count(
            connection,
            """
            INSERT INTO lineup_member (
                map_id, team_id, player_id, slot, member_type, known_at,
                actual_at, source_snapshot_id
            ) VALUES (?, ?, ?, ?, 'starter', NULL, ?, ?)
            """,
            (
                plan.map_id,
                player.team.team_id,
                player.player_id,
                player.slot,
                historic.started_at,
                snapshot_id,
            ),
        )
        metrics = player.metrics
        inserted["player_map_stats"] += _change_count(
            connection,
            """
            INSERT INTO player_map_stats (
                map_id, team_id, player_id, side, metric_version, known_at,
                observed_at, kills, deaths, assists, flash_assists, headshots,
                adr, kast, rating, metrics_json, source_snapshot_id
            ) VALUES (?, ?, ?, 'BOTH', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.map_id,
                player.team.team_id,
                player.player_id,
                METRIC_VERSION,
                observed_at,
                metrics.kills,
                metrics.deaths,
                metrics.assists,
                metrics.flash_assists,
                metrics.headshots,
                metrics.adr,
                metrics.kast,
                metrics.rating,
                player.metrics_json,
                snapshot_id,
            ),
        )

    for team in historic.teams:
        lineup_count = connection.execute(
            "SELECT COUNT(*) FROM lineup_member WHERE map_id = ? AND team_id = ?",
            (plan.map_id, team.team_id),
        ).fetchone()[0]
        stat_count = connection.execute(
            """
            SELECT COUNT(*) FROM player_map_stats
            WHERE map_id = ? AND team_id = ? AND side = 'BOTH'
              AND metric_version = ?
            """,
            (plan.map_id, team.team_id, METRIC_VERSION),
        ).fetchone()[0]
        if lineup_count != 5 or stat_count != 5:
            raise TedTayImportError(
                f"{plan.map_id} did not produce exactly five lineup and stat rows per team"
            )


def _write_batch(
    connection: sqlite3.Connection,
    plans: Iterable[_MapPlan],
    *,
    snapshot_id: str,
    observed_at: str,
    inserted: dict[str, int],
    raw_index: dict[str, set[str]],
) -> None:
    plans = tuple(plans)
    with connection:
        replayed = {
            plan.map_id: _assert_existing_plan(
                connection,
                plan,
                snapshot_id=snapshot_id,
                observed_at=observed_at,
                raw_index=raw_index,
            )
            for plan in plans
        }
        for plan in plans:
            if not replayed[plan.map_id]:
                _insert_new_plan(
                    connection,
                    plan,
                    snapshot_id=snapshot_id,
                    observed_at=observed_at,
                    inserted=inserted,
                    raw_index=raw_index,
                )


def _count(connection: sqlite3.Connection, statement: str, values: tuple[Any, ...]) -> int:
    return int(connection.execute(statement, values).fetchone()[0])


def import_tedtay_dataset(
    connection: sqlite3.Connection,
    historic_games_list_csv: str | Path,
    game_data_rh_csv: str | Path,
    from_date: date | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    """Import an audited, downloaded TedTay raw CSV pair.

    The two files are staged into temporary SQLite tables using their first
    exact ``game_link`` rows, then traversed in historic-file order.  Each
    normalized batch is one transaction.  A process may therefore be restarted
    safely after any successful batch: deterministic stable IDs plus structural
    checks make an exact replay a no-op and reject a rebind.  When four
    captured half-score cells cannot prove which player-stat table belongs to
    which team (overtime/long maps and tied finals), the map result is still
    imported but the two raw rows are retained as an explicit immutable
    quarantine record instead of assigning players by heuristic.  A historic
    row without a first exact stats row is likewise imported as a map result
    with no lineup/stats and its historic CSV row in a separate quarantine.

    ``from_date`` filters validated historic source ``date_ymd`` values before
    full result/lineup parsing.  It never changes the source snapshot or converts any current
    observation timestamp into a historical ``known_at``.
    """

    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise TedTayImportError(
            f"batch_size must be between 1 and {MAX_BATCH_SIZE}, inclusive"
        )
    if from_date is not None and (
        not isinstance(from_date, date) or isinstance(from_date, datetime)
    ):
        raise TedTayImportError("from_date must be a date or null")
    historic_path = Path(historic_games_list_csv).expanduser().resolve()
    stats_path = Path(game_data_rh_csv).expanduser().resolve()
    if historic_path == stats_path:
        raise TedTayImportError("historic_games_list.csv and game_data_rh.csv must differ")
    if not historic_path.is_file() or not stats_path.is_file():
        raise TedTayImportError("both downloaded CSV input files must exist")

    historic_hash = _file_sha256(historic_path)
    stats_hash = _file_sha256(stats_path)
    _create_temp_tables(connection)
    try:
        historic_rows, historic_duplicates = _stage_historic(connection, historic_path)
        stats_rows, stats_duplicates = _stage_stats(connection, stats_path)
        _assert_no_staged_rebinding(connection, "tedtay_historic_first")
        _assert_no_staged_rebinding(connection, "tedtay_stats_first")

        # The hashes are a bronze manifest, not just a preflight diagnostic.
        # Do not normalize a mix of bytes from before and after an in-place
        # downloader/editor update.
        if (
            _file_sha256(historic_path) != historic_hash
            or _file_sha256(stats_path) != stats_hash
        ):
            raise TedTayImportError(
                "downloaded source files changed while staging; retry with immutable copies"
            )

        joined_total = _count(
            connection,
            """
            SELECT COUNT(*)
            FROM temp.tedtay_historic_first AS h
            JOIN temp.tedtay_stats_first AS s ON s.game_link = h.game_link
            """,
            (),
        )
        date_filter = from_date.isoformat() if from_date is not None else None
        eligible_historic_rows = _count(
            connection,
            """
            SELECT COUNT(*) FROM temp.tedtay_historic_first AS h
            WHERE ? IS NULL OR h.event_date >= ?
            """,
            (date_filter, date_filter),
        )
        eligible_joined_rows = _count(
            connection,
            """
            SELECT COUNT(*)
            FROM temp.tedtay_historic_first AS h
            JOIN temp.tedtay_stats_first AS s ON s.game_link = h.game_link
            WHERE ? IS NULL OR h.event_date >= ?
            """,
            (date_filter, date_filter),
        )
        eligible_unmatched_historic_rows = _count(
            connection,
            """
            SELECT COUNT(*)
            FROM temp.tedtay_historic_first AS h
            LEFT JOIN temp.tedtay_stats_first AS s ON s.game_link = h.game_link
            WHERE s.game_link IS NULL
              AND (? IS NULL OR h.event_date >= ?)
            """,
            (date_filter, date_filter),
        )
        snapshot_id, observed_at = _ensure_snapshot(
            connection,
            historic_path=historic_path,
            historic_hash=historic_hash,
            stats_path=stats_path,
            stats_hash=stats_hash,
        )
        quarantine_raw_index = _load_quarantine_raw_index(connection)

        inserted = {
            "teams": 0,
            "players": 0,
            "series": 0,
            "series_participants": 0,
            "maps": 0,
            "lineup_members": 0,
            "player_map_stats": 0,
            "quarantine_raw_records": 0,
            "ambiguous_team_binding_raw_records": 0,
            "missing_game_data_rh_raw_records": 0,
        }
        quarantine_samples: list[dict[str, str]] = []
        ambiguous_samples: list[dict[str, str]] = []
        missing_game_data_samples: list[dict[str, str]] = []
        quarantine_by_kind: dict[str, int] = {}
        quarantine_by_reason: dict[str, int] = {}
        ambiguous_by_reason: dict[str, int] = {}
        missing_game_data_by_reason: dict[str, int] = {}
        batches_committed = 0
        cursor_ordinal = 0
        while True:
            rows = connection.execute(
                """
                SELECT h.row_ordinal, h.row_json AS historic_json,
                       s.row_json AS stats_json
                FROM temp.tedtay_historic_first AS h
                LEFT JOIN temp.tedtay_stats_first AS s ON s.game_link = h.game_link
                WHERE h.row_ordinal > ?
                  AND (? IS NULL OR h.event_date >= ?)
                ORDER BY h.row_ordinal
                LIMIT ?
                """,
                (cursor_ordinal, date_filter, date_filter, batch_size),
            ).fetchall()
            if not rows:
                break
            plans_list: list[_MapPlan] = []
            for row in rows:
                historic = _parse_historic(json.loads(str(row["historic_json"])))
                stats_json = row["stats_json"]
                if stats_json is None:
                    plans_list.append(_plan_missing_game_data(historic))
                else:
                    plans_list.append(
                        _plan_map(
                            historic,
                            _parse_game_stats(json.loads(str(stats_json))),
                        )
                    )
            plans = tuple(plans_list)
            for plan in plans:
                if plan.is_quarantined:
                    kind = str(plan.quarantine_kind)
                    reason = str(plan.quarantine_reason)
                    raw_record_id = str(
                        _quarantine_raw_expectation(
                            plan,
                            snapshot_id=snapshot_id,
                            observed_at=observed_at,
                        )["raw_record_id"]
                    )
                    sample = {
                        "mapstatsid": plan.source_map_id,
                        "kind": kind,
                        "reason": reason,
                        "raw_record_id": raw_record_id,
                    }
                    quarantine_by_kind[kind] = quarantine_by_kind.get(kind, 0) + 1
                    reason_key = f"{kind}:{reason}"
                    quarantine_by_reason[reason_key] = (
                        quarantine_by_reason.get(reason_key, 0) + 1
                    )
                    if len(quarantine_samples) < MAX_QUARANTINE_SAMPLES:
                        quarantine_samples.append(sample)
                    if kind == AMBIGUOUS_TEAM_BINDING_KIND:
                        ambiguous_by_reason[reason] = (
                            ambiguous_by_reason.get(reason, 0) + 1
                        )
                        if len(ambiguous_samples) < MAX_QUARANTINE_SAMPLES:
                            ambiguous_samples.append(sample)
                    else:
                        missing_game_data_by_reason[reason] = (
                            missing_game_data_by_reason.get(reason, 0) + 1
                        )
                        if len(missing_game_data_samples) < MAX_QUARANTINE_SAMPLES:
                            missing_game_data_samples.append(sample)
            _write_batch(
                connection,
                plans,
                snapshot_id=snapshot_id,
                observed_at=observed_at,
                inserted=inserted,
                raw_index=quarantine_raw_index,
            )
            batches_committed += 1
            cursor_ordinal = int(rows[-1]["row_ordinal"])

        return {
            "source": SOURCE,
            "source_snapshot_id": snapshot_id,
            "historic_first_rows": historic_rows,
            "historic_duplicate_links_ignored": historic_duplicates,
            "stats_first_rows": stats_rows,
            "stats_duplicate_links_ignored": stats_duplicates,
            "joined_rows": joined_total,
            "eligible_rows": eligible_historic_rows,
            "eligible_joined_rows": eligible_joined_rows,
            "eligible_unmatched_historic_rows": eligible_unmatched_historic_rows,
            "filtered_by_from_date": historic_rows - eligible_historic_rows,
            "unmatched_historic_rows": historic_rows - joined_total,
            "unmatched_stats_rows": stats_rows - joined_total,
            "batches_committed": batches_committed,
            "quarantine_count": sum(quarantine_by_kind.values()),
            "quarantine_by_kind": dict(sorted(quarantine_by_kind.items())),
            "quarantine_by_reason": dict(sorted(quarantine_by_reason.items())),
            "quarantine_samples": quarantine_samples,
            "ambiguous_team_binding_stream": _quarantine_stream(
                AMBIGUOUS_TEAM_BINDING_KIND, snapshot_id
            ),
            "ambiguous_team_binding_count": quarantine_by_kind.get(
                AMBIGUOUS_TEAM_BINDING_KIND, 0
            ),
            "ambiguous_team_binding_by_reason": dict(sorted(ambiguous_by_reason.items())),
            "ambiguous_team_binding_samples": ambiguous_samples,
            "missing_game_data_rh_stream": _quarantine_stream(
                MISSING_GAME_DATA_KIND, snapshot_id
            ),
            "missing_game_data_rh_count": quarantine_by_kind.get(
                MISSING_GAME_DATA_KIND, 0
            ),
            "missing_game_data_rh_by_reason": dict(
                sorted(missing_game_data_by_reason.items())
            ),
            "missing_game_data_rh_samples": missing_game_data_samples,
            "inserted": inserted,
        }
    finally:
        connection.execute("DROP TABLE IF EXISTS temp.tedtay_historic_first")
        connection.execute("DROP TABLE IF EXISTS temp.tedtay_stats_first")
