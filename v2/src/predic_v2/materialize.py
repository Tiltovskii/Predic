"""Materialize verified, typed HLTV records into the v2 SQLite schema.

This module deliberately has no file or network client.  It is the small
"silver" step after either :func:`predic_v2.raw_jsonl.import_jsonl` or an
equivalent authorized ingestion process has stored immutable parsed records in
``raw_ingest_record``.

Two properties are intentional:

* IDs are provider IDs, never names.  For example, HLTV team ``10`` is always
  ``hltv:team:10`` and map-stat page ``501`` is always ``hltv:map:501``.
* A relation that the parser cannot prove is not fabricated.  In particular,
  a match-page lineup is scoped to the displayed match page, not to a specific
  map, while ``lineup_member`` is map-scoped.  Such lineups remain in bronze
  and are returned in the bounded quarantine report instead of being copied to
  every map in a series.

The normalizer does not manufacture ``known_at``.  It copies the parsed value
verbatim (including ``None``); ``observed_at`` is kept distinct throughout.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_RECORDS = 1_000
MAX_RECORDS = 10_000
DEFAULT_MAX_QUARANTINE = 200
MAX_RECORD_BYTES = 2 * 1024 * 1024

_HLTV_ID = re.compile(r"^[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SUPPORTED_KINDS = frozenset(
    {"series", "map", "ranking", "lineup", "player_map_stats"}
)
_GAME_VERSIONS = frozenset({"CSGO", "CS2", "UNKNOWN"})
_SIDES = frozenset({"BOTH", "T", "CT", "UNKNOWN"})


class MaterializationError(ValueError):
    """Raised for a caller-level materialization contract violation."""


class BatchLimitError(MaterializationError):
    """Raised before any write when a direct batch exceeds its explicit limit."""


class _InvalidRecord(ValueError):
    """Expected malformed-record condition, represented in the quarantine."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class _Envelope:
    record_id: str
    kind: str
    source_document_sha256: str
    parser_version: str
    event_at: str | None
    known_at: str | None
    observed_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class _Team:
    source_id: str
    name: str

    @property
    def team_id(self) -> str:
        return f"hltv:team:{self.source_id}"


@dataclass(frozen=True)
class _SeriesPlan:
    envelope: _Envelope
    match_id: str
    teams: tuple[_Team, _Team]
    status: str
    best_of: int | None
    lan_online: str | None
    scheduled_at: str | None
    event_name: str | None
    stage_name: str | None

    @property
    def series_id(self) -> str:
        return f"hltv:series:{self.match_id}"


@dataclass(frozen=True)
class _MapPlan:
    envelope: _Envelope
    match_id: str
    source_map_id: str
    map_id: str
    map_order: int
    map_name: str
    game_version: str
    ruleset: str
    teams: tuple[_Team, _Team]
    score_a: int
    score_b: int
    winner_team_id: str
    picked_by_team_id: str | None

    @property
    def series_id(self) -> str:
        return f"hltv:series:{self.match_id}"


@dataclass(frozen=True)
class _RankingPlan:
    envelope: _Envelope
    match_id: str
    team: _Team
    rank: int
    ranking_system: str

    @property
    def ranking_snapshot_id(self) -> str:
        return (
            f"hltv:ranking:match:{self.match_id}:team:{self.team.source_id}:"
            f"doc-{self.envelope.source_document_sha256[:16]}"
        )


@dataclass(frozen=True)
class _StatsPlan:
    envelope: _Envelope
    map_stats_id: str
    team_source_id: str
    player_source_id: str
    nickname: str
    side: str
    metric_version: str
    metrics: dict[str, Any]

    @property
    def map_id(self) -> str:
        return f"hltv:map:{self.map_stats_id}"

    @property
    def team_id(self) -> str:
        return f"hltv:team:{self.team_source_id}"

    @property
    def player_id(self) -> str:
        return f"hltv:player:{self.player_source_id}"


class _Quarantine:
    """A bounded, caller-visible quarantine report.

    The existing v2 schema intentionally has no catch-all error table.  Raw
    records are already durable in ``raw_ingest_record``; returning stable
    record IDs and reasons lets a caller persist or inspect only the bounded
    report without turning a malformed relation into normalized data.
    """

    def __init__(self, max_items: int) -> None:
        if max_items < 0:
            raise MaterializationError("max_quarantine must be non-negative")
        self._max_items = max_items
        self._items: list[dict[str, str]] = []
        self.count = 0

    def add(
        self,
        *,
        record_id: str | None,
        kind: str | None,
        reason: str,
        detail: str | None = None,
    ) -> None:
        self.count += 1
        if len(self._items) >= self._max_items:
            return
        item = {"reason": reason}
        if record_id is not None:
            item["record_id"] = record_id
        if kind is not None:
            item["kind"] = kind
        if detail:
            item["detail"] = detail
        self._items.append(item)

    def as_result(self) -> tuple[list[dict[str, str]], bool]:
        return list(self._items), self.count > len(self._items)


def _canonical_json(value: object) -> str:
    try:
        result = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise _InvalidRecord("record_not_json_serializable", str(error)) from error
    if len(result.encode("utf-8")) > MAX_RECORD_BYTES:
        raise _InvalidRecord(
            "record_too_large",
            f"record exceeds the {MAX_RECORD_BYTES}-byte materialization limit",
        )
    return result


def _nonempty_text(
    value: object, field: str, *, max_length: int = 1_000
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _InvalidRecord("invalid_field", f"{field} must be a non-empty string")
    if value != value.strip():
        raise _InvalidRecord("invalid_field", f"{field} cannot have edge whitespace")
    if len(value) > max_length:
        raise _InvalidRecord("invalid_field", f"{field} exceeds {max_length} characters")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, field)


def _hltv_id(value: object, field: str) -> str:
    raw = _nonempty_text(value, field, max_length=32)
    if not _HLTV_ID.fullmatch(raw):
        raise _InvalidRecord("invalid_hltv_id", f"{field} must be a positive numeric HLTV ID")
    normalized = raw.lstrip("0") or "0"
    if normalized == "0":
        raise _InvalidRecord("invalid_hltv_id", f"{field} must be greater than zero")
    return normalized


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidRecord("invalid_field", f"{field} must be an integer or null")
    if value < -(2**63) or value > 2**63 - 1:
        raise _InvalidRecord("invalid_field", f"{field} is outside SQLite integer range")
    return value


def _optional_number(value: object, field: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidRecord("invalid_field", f"{field} must be numeric or null")
    if not math.isfinite(value):
        raise _InvalidRecord("invalid_field", f"{field} must be finite")
    return value


def _parsed_envelopes(
    records: Iterable[Mapping[str, object]],
    *,
    max_records: int,
    observed_at_fallback: str | None,
    quarantine: _Quarantine,
) -> tuple[list[_Envelope], int, int]:
    if max_records < 1 or max_records > MAX_RECORDS:
        raise MaterializationError(
            f"max_records must be between 1 and {MAX_RECORDS}, inclusive"
        )
    fallback = (
        _nonempty_text(observed_at_fallback, "observed_at_fallback")
        if observed_at_fallback is not None
        else None
    )

    valid: list[_Envelope] = []
    fingerprint_by_id: dict[str, str] = {}
    conflicting_ids: set[str] = set()
    duplicate_records = 0
    input_records = 0
    iterator = iter(records)
    for position, record in enumerate(iterator, start=1):
        input_records += 1
        if input_records > max_records:
            raise BatchLimitError(
                f"direct materialization batch exceeds max_records={max_records}; "
                "split it before calling materialize_records"
            )
        if not isinstance(record, Mapping):
            quarantine.add(
                record_id=None,
                kind=None,
                reason="record_not_object",
                detail=f"input position {position} is not an object",
            )
            continue
        raw_record_id = record.get("record_id")
        raw_kind = record.get("kind")
        record_id = raw_record_id if isinstance(raw_record_id, str) else None
        kind = raw_kind if isinstance(raw_kind, str) else None
        try:
            canonical = _canonical_json(record)
            parsed_record_id = _nonempty_text(record.get("record_id"), "record_id")
            parsed_kind = _nonempty_text(record.get("kind"), "kind", max_length=128)
            if parsed_kind not in _SUPPORTED_KINDS:
                raise _InvalidRecord(
                    "unsupported_record_kind",
                    f"{parsed_kind!r} is not materializable by this adapter",
                )
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                raise _InvalidRecord("invalid_field", "payload must be an object")
            document_sha = _nonempty_text(
                record.get("source_document_sha256"), "source_document_sha256", max_length=64
            ).lower()
            if not _SHA256.fullmatch(document_sha):
                raise _InvalidRecord(
                    "invalid_document_hash",
                    "source_document_sha256 must be a SHA-256 hex digest",
                )
            parser_version = _nonempty_text(
                record.get("parser_version"), "parser_version", max_length=128
            )
            event_at = _optional_text(record.get("event_at"), "event_at")
            # Do not substitute observed_at/event_at/source time here.  A null
            # known_at is material information for point-in-time evaluation.
            known_at = _optional_text(record.get("known_at"), "known_at")
            observed_at = _optional_text(record.get("observed_at"), "observed_at")
            if observed_at is None:
                observed_at = fallback
            if observed_at is None:
                raise _InvalidRecord(
                    "observed_at_missing",
                    "captured/ingested observed_at is required for normalized rows",
                )
            previous = fingerprint_by_id.get(parsed_record_id)
            if previous is not None:
                if previous == canonical:
                    duplicate_records += 1
                    continue
                conflicting_ids.add(parsed_record_id)
                continue
            fingerprint_by_id[parsed_record_id] = canonical
            valid.append(
                _Envelope(
                    record_id=parsed_record_id,
                    kind=parsed_kind,
                    source_document_sha256=document_sha,
                    parser_version=parser_version,
                    event_at=event_at,
                    known_at=known_at,
                    observed_at=observed_at,
                    payload=dict(payload),
                )
            )
        except _InvalidRecord as error:
            quarantine.add(
                record_id=record_id,
                kind=kind,
                reason=error.reason,
                detail=error.detail,
            )

    if conflicting_ids:
        valid = [item for item in valid if item.record_id not in conflicting_ids]
        for record_id in sorted(conflicting_ids):
            quarantine.add(
                record_id=record_id,
                kind=None,
                reason="conflicting_duplicate_record_id",
                detail="the same record_id appeared with different JSON payloads",
            )
    return valid, input_records, duplicate_records


def _teams(payload: Mapping[str, object]) -> tuple[_Team, _Team]:
    raw_teams = payload.get("teams")
    if not isinstance(raw_teams, list) or len(raw_teams) != 2:
        raise _InvalidRecord("invalid_team_link", "teams must contain exactly two teams")
    parsed: list[_Team] = []
    for position, raw_team in enumerate(raw_teams, start=1):
        if not isinstance(raw_team, Mapping):
            raise _InvalidRecord("invalid_team_link", f"team {position} is not an object")
        parsed.append(
            _Team(
                source_id=_hltv_id(raw_team.get("id"), f"teams[{position}].id"),
                name=_nonempty_text(raw_team.get("name"), f"teams[{position}].name"),
            )
        )
    if parsed[0].source_id == parsed[1].source_id:
        raise _InvalidRecord("invalid_team_link", "a match cannot link a team to itself")
    return parsed[0], parsed[1]


def _map_teams(payload: Mapping[str, object]) -> tuple[_Team, _Team]:
    raw_ids = payload.get("team_ids")
    raw_names = payload.get("team_names")
    if not isinstance(raw_ids, list) or len(raw_ids) != 2:
        raise _InvalidRecord("invalid_team_link", "map team_ids must have exactly two values")
    if not isinstance(raw_names, list) or len(raw_names) != 2:
        raise _InvalidRecord("invalid_team_link", "map team_names must have exactly two values")
    teams = (
        _Team(
            _hltv_id(raw_ids[0], "team_ids[0]"),
            _nonempty_text(raw_names[0], "team_names[0]"),
        ),
        _Team(
            _hltv_id(raw_ids[1], "team_ids[1]"),
            _nonempty_text(raw_names[1], "team_names[1]"),
        ),
    )
    if teams[0].source_id == teams[1].source_id:
        raise _InvalidRecord("invalid_team_link", "a map cannot link a team to itself")
    return teams


def _series_plan(envelope: _Envelope) -> _SeriesPlan:
    payload = envelope.payload
    match_id = _hltv_id(payload.get("match_id"), "match_id")
    teams = _teams(payload)
    status = _nonempty_text(payload.get("status"), "status", max_length=64)
    best_of = _optional_int(payload.get("best_of"), "best_of")
    if best_of is not None and not 1 <= best_of <= 7:
        raise _InvalidRecord("invalid_field", "best_of must be from 1 through 7")
    lan_online = _optional_text(payload.get("lan_online"), "lan_online")
    event = payload.get("event")
    event_name = None
    stage_name = None
    if event is not None:
        if not isinstance(event, Mapping):
            raise _InvalidRecord("invalid_field", "event must be an object or null")
        event_name = _optional_text(event.get("name"), "event.name")
        stage_name = _optional_text(event.get("stage_name"), "event.stage_name")
    # The parser calls this scheduled_at and records the same value in event_at.
    # Never derive a start/end timestamp that the page did not expose.
    scheduled_at = _optional_text(payload.get("scheduled_at"), "scheduled_at")
    return _SeriesPlan(
        envelope=envelope,
        match_id=match_id,
        teams=teams,
        status=status,
        best_of=best_of,
        lan_online=lan_online,
        scheduled_at=scheduled_at,
        event_name=event_name,
        stage_name=stage_name,
    )


def _map_plan(envelope: _Envelope) -> _MapPlan:
    payload = envelope.payload
    if _nonempty_text(payload.get("status"), "status", max_length=64) != "played":
        raise _InvalidRecord(
            "map_not_finished",
            "map_game has no status column; only parser-confirmed played maps are materialized",
        )
    match_id = _hltv_id(payload.get("match_id"), "match_id")
    map_stats_raw = payload.get("map_stats_id")
    map_order = _optional_int(payload.get("map_order"), "map_order")
    if map_order is None or map_order < 1:
        raise _InvalidRecord("invalid_field", "map_order must be a positive integer")
    if map_stats_raw is None:
        source_map_id = f"match:{match_id}:position:{map_order}"
        map_id = f"hltv:map:{match_id}:position:{map_order}"
    else:
        source_map_id = _hltv_id(map_stats_raw, "map_stats_id")
        map_id = f"hltv:map:{source_map_id}"
    teams = _map_teams(payload)
    map_name = _nonempty_text(payload.get("map_name"), "map_name")
    game_version = _nonempty_text(payload.get("game_version"), "game_version", max_length=16)
    if game_version not in _GAME_VERSIONS:
        raise _InvalidRecord("invalid_field", "game_version is not a recognized CS version")
    ruleset = _nonempty_text(payload.get("ruleset"), "ruleset", max_length=32)
    score_a = _optional_int(payload.get("score_a"), "score_a")
    score_b = _optional_int(payload.get("score_b"), "score_b")
    if score_a is None or score_b is None or score_a < 0 or score_b < 0:
        raise _InvalidRecord(
            "invalid_score", "a played map requires two non-negative scores"
        )
    if score_a == score_b:
        raise _InvalidRecord("invalid_score", "a played map cannot have a tied final score")
    expected_winner = teams[0] if score_a > score_b else teams[1]
    winner_source_id = _hltv_id(payload.get("winner_team_id"), "winner_team_id")
    if winner_source_id != expected_winner.source_id:
        raise _InvalidRecord(
            "invalid_winner_link",
            "winner_team_id does not agree with the final map score and team IDs",
        )
    raw_pick = payload.get("picked_by_team_id")
    picked_by_team_id = None
    if raw_pick is not None:
        pick_source_id = _hltv_id(raw_pick, "picked_by_team_id")
        if pick_source_id not in {teams[0].source_id, teams[1].source_id}:
            raise _InvalidRecord(
                "invalid_pick_link", "picked_by_team_id is not a participant in this map"
            )
        picked_by_team_id = f"hltv:team:{pick_source_id}"
    return _MapPlan(
        envelope=envelope,
        match_id=match_id,
        source_map_id=source_map_id,
        map_id=map_id,
        map_order=map_order,
        map_name=map_name,
        game_version=game_version,
        ruleset=ruleset,
        teams=teams,
        score_a=score_a,
        score_b=score_b,
        winner_team_id=expected_winner.team_id,
        picked_by_team_id=picked_by_team_id,
    )


def _ranking_plan(envelope: _Envelope) -> _RankingPlan:
    payload = envelope.payload
    match_id = _hltv_id(payload.get("match_id"), "match_id")
    team = _Team(
        _hltv_id(payload.get("team_id"), "team_id"),
        _nonempty_text(payload.get("team_name"), "team_name"),
    )
    rank = _optional_int(payload.get("rank"), "rank")
    if rank is None or rank < 1:
        raise _InvalidRecord("invalid_field", "rank must be a positive integer")
    ranking_system = _nonempty_text(
        payload.get("ranking_system"), "ranking_system", max_length=128
    )
    return _RankingPlan(
        envelope=envelope,
        match_id=match_id,
        team=team,
        rank=rank,
        ranking_system=ranking_system,
    )


def _stats_plan(envelope: _Envelope) -> _StatsPlan:
    payload = envelope.payload
    map_stats_id = _hltv_id(payload.get("map_stats_id"), "map_stats_id")
    team_source_id = _hltv_id(payload.get("team_id"), "team_id")
    player_source_id = _hltv_id(payload.get("player_id"), "player_id")
    nickname = _nonempty_text(payload.get("nickname"), "nickname")
    side = _nonempty_text(payload.get("side"), "side", max_length=16)
    if side not in _SIDES:
        raise _InvalidRecord("invalid_field", "side is not a recognized player-stat side")
    metric_version = _nonempty_text(
        payload.get("metric_version"), "metric_version", max_length=128
    )
    # Validate the typed columns before writing.  Raw metrics stay in JSON, so
    # future HLTV fields are retained without guessing a schema mapping.
    metrics: dict[str, Any] = {}
    for field in (
        "kills",
        "deaths",
        "assists",
        "flash_assists",
        "headshots",
        "traded_deaths",
        "opening_kills",
        "opening_deaths",
        "multi_kill_rounds",
        "clutch_wins",
    ):
        value = _optional_int(payload.get(field), field)
        if value is not None and value < 0:
            raise _InvalidRecord("invalid_field", f"{field} cannot be negative")
        metrics[field] = value
    for field in ("adr", "kast", "kpr", "dpr", "swing", "rating"):
        metrics[field] = _optional_number(payload.get(field), field)
    # The full source payload is intentionally retained for evolving fields
    # such as Swing variants, round breakdowns, raw metric cells, etc.
    _canonical_json(payload)
    metrics["metrics_json"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return _StatsPlan(
        envelope=envelope,
        map_stats_id=map_stats_id,
        team_source_id=team_source_id,
        player_source_id=player_source_id,
        nickname=nickname,
        side=side,
        metric_version=metric_version,
        metrics=metrics,
    )


def _chunks(values: Sequence[str], size: int = 500) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _existing_ids(
    connection: sqlite3.Connection, table: str, column: str, values: Sequence[str]
) -> set[str]:
    found: set[str] = set()
    for group in _chunks(values):
        if not group:
            continue
        placeholders = ",".join("?" for _ in group)
        rows = connection.execute(
            f"SELECT {column} FROM {table} WHERE {column} IN ({placeholders})",
            tuple(group),
        )
        found.update(str(row[0]) for row in rows)
    return found


def _existing_maps(
    connection: sqlite3.Connection, map_ids: Sequence[str]
) -> dict[str, tuple[str, str]]:
    return {
        map_id: (identity[3], identity[4])
        for map_id, identity in _existing_map_identities(connection, map_ids).items()
    }


def _existing_map_identities(
    connection: sqlite3.Connection, map_ids: Sequence[str]
) -> dict[str, tuple[str, str | None, int | None, str, str, int | None, int | None, str | None, str | None]]:
    """Load the immutable structural link for existing global HLTV map IDs.

    ``hltv:map:<mapstatsid>`` is global, not scoped to one source snapshot.
    If a later match page tries to attach that ID to a different series/map,
    ``INSERT OR IGNORE`` would otherwise keep the old row silently and let
    incoming player stats join the wrong match.  Keep the comparison separate
    from mutable labels/timestamps, but include every relationship/outcome
    field that defines a map result.
    """

    found: dict[
        str,
        tuple[
            str,
            str | None,
            int | None,
            str,
            str,
            int | None,
            int | None,
            str | None,
            str | None,
        ],
    ] = {}
    for group in _chunks(map_ids):
        if not group:
            continue
        placeholders = ",".join("?" for _ in group)
        for row in connection.execute(
            """
            SELECT map_id, series_id, source_map_id, map_order,
                   team_a_id, team_b_id, score_a, score_b,
                   winner_team_id, picked_by_team_id
            FROM map_game
            """
            f"WHERE map_id IN ({placeholders})",
            tuple(group),
        ):
            found[str(row["map_id"])] = (
                str(row["series_id"]),
                str(row["source_map_id"]) if row["source_map_id"] is not None else None,
                int(row["map_order"]) if row["map_order"] is not None else None,
                str(row["team_a_id"]),
                str(row["team_b_id"]),
                int(row["score_a"]) if row["score_a"] is not None else None,
                int(row["score_b"]) if row["score_b"] is not None else None,
                str(row["winner_team_id"]) if row["winner_team_id"] is not None else None,
                str(row["picked_by_team_id"])
                if row["picked_by_team_id"] is not None
                else None,
            )
    return found


def _map_identity(
    candidate: _MapPlan,
) -> tuple[str, str, int, str, str, int, int, str, str | None]:
    return (
        candidate.series_id,
        candidate.source_map_id,
        candidate.map_order,
        candidate.teams[0].team_id,
        candidate.teams[1].team_id,
        candidate.score_a,
        candidate.score_b,
        candidate.winner_team_id,
        candidate.picked_by_team_id,
    )


def _stats_key(candidate: _StatsPlan) -> tuple[str, str, str, str]:
    return (
        candidate.map_id,
        candidate.player_id,
        candidate.side,
        candidate.metric_version,
    )


def _existing_stats_team_links(
    connection: sqlite3.Connection,
    keys: Sequence[tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """Return durable team and canonical payload for each player-map key.

    The schema primary key intentionally omits ``team_id`` because a player
    can only have one stat line for one map/side/rating version.  Query it
    before ``INSERT OR IGNORE`` so a malformed later source cannot silently
    attach that player to the other valid participant team.
    """

    found: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for index in range(0, len(keys), 200):
        group = keys[index : index + 200]
        if not group:
            continue
        predicate = " OR ".join(
            "(map_id = ? AND player_id = ? AND side = ? AND metric_version = ?)"
            for _ in group
        )
        values = tuple(value for key in group for value in key)
        for row in connection.execute(
            "SELECT map_id, player_id, side, metric_version, team_id, metrics_json "
            "FROM player_map_stats WHERE " + predicate,
            values,
        ):
            found[
                (
                    str(row["map_id"]),
                    str(row["player_id"]),
                    str(row["side"]),
                    str(row["metric_version"]),
                )
            ] = (str(row["team_id"]), str(row["metrics_json"]))
    return found


def _existing_series_participants(
    connection: sqlite3.Connection, series_ids: Sequence[str]
) -> dict[str, set[str]]:
    """Return the durable, provider-backed participant set for each series.

    ``series`` itself has no two-team relation.  Keeping that relation in a
    separate normalized table is what lets later map/ranking phases validate
    their team IDs rather than trusting a previously materialized series ID.
    """

    found: dict[str, set[str]] = {}
    for group in _chunks(series_ids):
        if not group:
            continue
        placeholders = ",".join("?" for _ in group)
        for row in connection.execute(
            "SELECT series_id, team_id FROM series_participant "
            f"WHERE series_id IN ({placeholders})",
            tuple(group),
        ):
            found.setdefault(str(row["series_id"]), set()).add(str(row["team_id"]))
    return found


def _insert_team(
    connection: sqlite3.Connection,
    team: _Team,
    source_snapshot_id: str,
    inserted: dict[str, int],
) -> None:
    before = connection.total_changes
    connection.execute(
        """
        INSERT OR IGNORE INTO team_core (
            team_id, canonical_name, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'high', ?)
        """,
        (team.team_id, team.name, source_snapshot_id),
    )
    inserted["teams"] += connection.total_changes - before
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES ('hltv', 'team', ?, ?, ?)
        """,
        (team.source_id, team.team_id, source_snapshot_id),
    )


def _insert_player(
    connection: sqlite3.Connection,
    player_source_id: str,
    nickname: str,
    source_snapshot_id: str,
    inserted: dict[str, int],
) -> None:
    player_id = f"hltv:player:{player_source_id}"
    before = connection.total_changes
    connection.execute(
        """
        INSERT OR IGNORE INTO player (
            player_id, canonical_nickname, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'high', ?)
        """,
        (player_id, nickname, source_snapshot_id),
    )
    inserted["players"] += connection.total_changes - before
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES ('hltv', 'player', ?, ?, ?)
        """,
        (player_source_id, player_id, source_snapshot_id),
    )


def _rowcount_change(
    connection: sqlite3.Connection, statement: str, values: tuple[Any, ...]
) -> int:
    before = connection.total_changes
    connection.execute(statement, values)
    return connection.total_changes - before


def materialize_records(
    connection: sqlite3.Connection,
    records: Iterable[Mapping[str, object]],
    *,
    source_snapshot_id: str,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_quarantine: int = DEFAULT_MAX_QUARANTINE,
    observed_at_fallback: str | None = None,
) -> dict[str, object]:
    """Materialize one bounded batch of parsed HLTV envelopes.

    ``source_snapshot_id`` must already exist (normally it is the snapshot
    created by ``import_jsonl``).  Valid records commit together in one SQLite
    transaction.  Invalid envelopes/links are not written and are returned in
    ``quarantined``; the raw source remains the durable source of truth.

    ``known_at`` is copied exactly as supplied.  The optional observed-at
    fallback is intended for the raw-stream wrapper, where
    ``raw_ingest_record.observed_at`` is already a durable ingestion timestamp.
    It is never used as a known-at fallback.
    """

    snapshot_id = _nonempty_text(source_snapshot_id, "source_snapshot_id")
    snapshot = connection.execute(
        "SELECT snapshot_id FROM source_snapshot WHERE snapshot_id = ?", (snapshot_id,)
    ).fetchone()
    if snapshot is None:
        raise MaterializationError(
            f"source_snapshot_id {snapshot_id!r} does not exist; ingest bronze data first"
        )

    quarantine = _Quarantine(max_quarantine)
    envelopes, input_records, duplicate_records = _parsed_envelopes(
        records,
        max_records=max_records,
        observed_at_fallback=observed_at_fallback,
        quarantine=quarantine,
    )

    series_candidates: list[_SeriesPlan] = []
    map_candidates: list[_MapPlan] = []
    ranking_candidates: list[_RankingPlan] = []
    stats_candidates: list[_StatsPlan] = []
    for envelope in envelopes:
        try:
            if envelope.kind == "series":
                series_candidates.append(_series_plan(envelope))
            elif envelope.kind == "map":
                map_candidates.append(_map_plan(envelope))
            elif envelope.kind == "ranking":
                ranking_candidates.append(_ranking_plan(envelope))
            elif envelope.kind == "player_map_stats":
                stats_candidates.append(_stats_plan(envelope))
            else:
                # Parsed match-page lineup is not map-specific.  The schema
                # has no series-level lineup table, so materializing it would
                # manufacture a relation to every/any map.
                quarantine.add(
                    record_id=envelope.record_id,
                    kind=envelope.kind,
                    reason="lineup_not_map_scoped",
                    detail="match_page_displayed lineup cannot populate map-scoped lineup_member",
                )
        except _InvalidRecord as error:
            quarantine.add(
                record_id=envelope.record_id,
                kind=envelope.kind,
                reason=error.reason,
                detail=error.detail,
            )

    # Stable IDs make exact replay harmless.  If a caller accidentally gives
    # us two revision records for the same source entity in one batch, retain
    # one deterministic representative rather than making output depend on
    # iteration order.  Conflicting team identity is quarantined as a bad link.
    series_groups: dict[str, list[_SeriesPlan]] = {}
    for candidate in series_candidates:
        series_groups.setdefault(candidate.series_id, []).append(candidate)
    series_by_id: dict[str, _SeriesPlan] = {}
    for series_id, candidates in series_groups.items():
        participant_sets = {
            frozenset(team.source_id for team in candidate.teams)
            for candidate in candidates
        }
        if len(participant_sets) != 1:
            for candidate in candidates:
                quarantine.add(
                    record_id=candidate.envelope.record_id,
                    kind="series",
                    reason="conflicting_series_team_link",
                    detail=f"stable series {series_id} has incompatible team IDs",
                )
            continue
        series_by_id[series_id] = min(
            candidates, key=lambda item: item.envelope.record_id
        )
    relevant_series_ids = sorted(
        {
            *series_by_id,
            *(f"hltv:series:{candidate.match_id}" for candidate in map_candidates),
            *(
                f"hltv:series:{candidate.match_id}"
                for candidate in ranking_candidates
            ),
        }
    )
    existing_series_ids = _existing_ids(
        connection,
        "series",
        "series_id",
        relevant_series_ids,
    )
    existing_series_teams = _existing_series_participants(
        connection, relevant_series_ids
    )
    retained_series_by_id: dict[str, _SeriesPlan] = {}
    for series_id, candidate in series_by_id.items():
        incoming_teams = {team.team_id for team in candidate.teams}
        existing_teams = existing_series_teams.get(series_id)
        if existing_teams is not None and existing_teams != incoming_teams:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="series",
                reason="conflicting_series_team_link",
                detail=(
                    f"stable series {series_id} is already linked to incompatible "
                    "team IDs"
                ),
            )
            continue
        retained_series_by_id[series_id] = candidate
    series_by_id = retained_series_by_id
    series_plans = list(series_by_id.values())
    planned_series_ids = set(series_by_id)

    valid_maps: list[_MapPlan] = []
    blocked_map_ids: set[str] = set()
    series_teams = {
        candidate.series_id: {team.team_id for team in candidate.teams}
        for candidate in series_plans
    }
    for candidate in map_candidates:
        if (
            candidate.series_id not in planned_series_ids
            and candidate.series_id not in existing_series_ids
        ):
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="map",
                reason="missing_series_link",
                detail=f"{candidate.series_id} is not materialized in this batch or database",
            )
            blocked_map_ids.add(candidate.map_id)
            continue
        expected_teams = series_teams.get(candidate.series_id) or existing_series_teams.get(
            candidate.series_id
        )
        if expected_teams is None:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="map",
                reason="series_participants_missing",
                detail=(
                    f"{candidate.series_id} has no verified participant relation; "
                    "materialize its series record first"
                ),
            )
            blocked_map_ids.add(candidate.map_id)
            continue
        if expected_teams != {
            candidate.teams[0].team_id,
            candidate.teams[1].team_id,
        }:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="map",
                reason="map_teams_do_not_match_series",
                detail="map participant IDs conflict with its series record",
            )
            blocked_map_ids.add(candidate.map_id)
            continue
        valid_maps.append(candidate)

    map_groups: dict[str, list[_MapPlan]] = {}
    for candidate in valid_maps:
        map_groups.setdefault(candidate.map_id, []).append(candidate)
    map_by_id: dict[str, _MapPlan] = {}
    for map_id, candidates in map_groups.items():
        source_links = {
            (
                candidate.series_id,
                candidate.source_map_id,
                tuple(team.source_id for team in candidate.teams),
                candidate.score_a,
                candidate.score_b,
                candidate.winner_team_id,
                candidate.picked_by_team_id,
            )
            for candidate in candidates
        }
        if len(source_links) != 1:
            for candidate in candidates:
                quarantine.add(
                    record_id=candidate.envelope.record_id,
                    kind="map",
                    reason="conflicting_map_link",
                    detail=f"stable map {map_id} has incompatible series/team links",
                )
                blocked_map_ids.add(candidate.map_id)
            continue
        map_by_id[map_id] = min(candidates, key=lambda item: item.envelope.record_id)
    map_plans = list(map_by_id.values())

    existing_source_maps: dict[str, str] = {}
    source_map_ids = [item.source_map_id for item in map_plans]
    for group in _chunks(source_map_ids):
        if not group:
            continue
        placeholders = ",".join("?" for _ in group)
        for row in connection.execute(
            "SELECT source_map_id, map_id FROM map_game "
            "WHERE source_snapshot_id = ? AND source_map_id IN ("
            + placeholders
            + ")",
            (snapshot_id, *group),
        ):
            existing_source_maps[str(row["source_map_id"])] = str(row["map_id"])
    retained_map_plans: list[_MapPlan] = []
    for candidate in map_plans:
        existing_map_id = existing_source_maps.get(candidate.source_map_id)
        if existing_map_id is not None and existing_map_id != candidate.map_id:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="map",
                reason="source_map_identity_conflict",
                detail=(
                    f"source map {candidate.source_map_id} is already bound to "
                    f"{existing_map_id}, not {candidate.map_id}"
                ),
            )
            blocked_map_ids.add(candidate.map_id)
            continue
        retained_map_plans.append(candidate)
    map_plans = retained_map_plans

    # ``map_id`` is based on the provider's map-stats ID and is therefore
    # global across batches and source snapshots.  SQLite's INSERT OR IGNORE
    # would silently retain an older row on conflict, so compare the full
    # structural map relation before any player stats can use it as a join.
    existing_map_identities = _existing_map_identities(
        connection, [candidate.map_id for candidate in map_plans]
    )
    retained_map_plans = []
    for candidate in map_plans:
        existing_identity = existing_map_identities.get(candidate.map_id)
        if existing_identity is not None and existing_identity != _map_identity(candidate):
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="map",
                reason="map_identity_conflict",
                detail=(
                    f"global map {candidate.map_id} is already linked to a different "
                    "series/team/score relation"
                ),
            )
            blocked_map_ids.add(candidate.map_id)
            continue
        retained_map_plans.append(candidate)
    map_plans = retained_map_plans

    # Rankings are contextual to a match page.  Do not retain an orphan rank
    # merely because both IDs happen to look numeric.
    valid_rankings: list[_RankingPlan] = []
    for candidate in ranking_candidates:
        series_id = f"hltv:series:{candidate.match_id}"
        if series_id not in planned_series_ids and series_id not in existing_series_ids:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="ranking",
                reason="missing_series_link",
                detail=f"{series_id} is not materialized in this batch or database",
            )
            continue
        expected_teams = series_teams.get(series_id) or existing_series_teams.get(
            series_id
        )
        if expected_teams is None:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="ranking",
                reason="series_participants_missing",
                detail=(
                    f"{series_id} has no verified participant relation; "
                    "materialize its series record first"
                ),
            )
            continue
        if candidate.team.team_id not in expected_teams:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="ranking",
                reason="ranking_team_not_in_series",
                detail="ranking team ID is not a participant in its match record",
            )
            continue
        valid_rankings.append(candidate)

    existing_map_links = _existing_maps(
        connection, [candidate.map_id for candidate in stats_candidates]
    )
    planned_map_links = {
        candidate.map_id: (candidate.teams[0].team_id, candidate.teams[1].team_id)
        for candidate in map_plans
    }
    valid_stats: list[_StatsPlan] = []
    for candidate in stats_candidates:
        if candidate.map_id in blocked_map_ids:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="player_map_stats",
                reason="map_identity_conflict",
                detail=(
                    f"map-stats ID {candidate.map_stats_id} had a conflicting map "
                    "relation in this batch"
                ),
            )
            continue
        map_teams = planned_map_links.get(candidate.map_id) or existing_map_links.get(
            candidate.map_id
        )
        if map_teams is None:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="player_map_stats",
                reason="missing_map_link",
                detail=(
                    f"map-stats ID {candidate.map_stats_id} has no verified "
                    "map_game link; no map is invented"
                ),
            )
            continue
        if candidate.team_id not in set(map_teams):
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="player_map_stats",
                reason="stats_team_not_in_map",
                detail="player-stat team ID is not one of the materialized map participants",
            )
            continue
        valid_stats.append(candidate)

    # A player-map primary key has exactly one team relation.  Check both the
    # current batch and durable rows before INSERT OR IGNORE can hide a
    # contradiction.  Exact duplicate source lines are deterministically
    # collapsed; divergent metrics for the same key are quarantined rather
    # than making result depend on iteration order.
    stats_by_key: dict[tuple[str, str, str, str], list[_StatsPlan]] = {}
    for candidate in valid_stats:
        stats_by_key.setdefault(_stats_key(candidate), []).append(candidate)
    deduplicated_stats: list[_StatsPlan] = []
    for key, candidates in stats_by_key.items():
        team_ids = {candidate.team_id for candidate in candidates}
        if len(team_ids) != 1:
            for candidate in candidates:
                quarantine.add(
                    record_id=candidate.envelope.record_id,
                    kind="player_map_stats",
                    reason="conflicting_player_map_stats_team",
                    detail=(
                        "one map/player/side/metric-version key points to multiple "
                        "participant teams"
                    ),
                )
            continue
        payloads = {str(candidate.metrics["metrics_json"]) for candidate in candidates}
        if len(payloads) != 1:
            for candidate in candidates:
                quarantine.add(
                    record_id=candidate.envelope.record_id,
                    kind="player_map_stats",
                    reason="conflicting_player_map_stats_payload",
                    detail=(
                        "one map/player/side/metric-version key has incompatible "
                        "metric payloads"
                    ),
                )
            continue
        deduplicated_stats.append(
            min(candidates, key=lambda item: item.envelope.record_id)
        )
    existing_stats_teams = _existing_stats_team_links(
        connection, [_stats_key(candidate) for candidate in deduplicated_stats]
    )
    valid_stats = []
    for candidate in deduplicated_stats:
        existing = existing_stats_teams.get(_stats_key(candidate))
        if existing is None:
            valid_stats.append(candidate)
            continue
        existing_team_id, existing_metrics_json = existing
        if existing_team_id != candidate.team_id:
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="player_map_stats",
                reason="conflicting_player_map_stats_team",
                detail=(
                    f"player-map stats are already linked to {existing_team_id}, not "
                    f"{candidate.team_id}"
                ),
            )
            continue
        if existing_metrics_json != str(candidate.metrics["metrics_json"]):
            quarantine.add(
                record_id=candidate.envelope.record_id,
                kind="player_map_stats",
                reason="conflicting_player_map_stats_payload",
                detail=(
                    "player-map stats key already exists with a different metric payload; "
                    "raw revisions are retained but normalized rows are not replaced"
                ),
            )
            continue
        valid_stats.append(candidate)

    inserted = {
        "teams": 0,
        "players": 0,
        "series": 0,
        "maps": 0,
        "rankings": 0,
        "player_map_stats": 0,
    }
    # All valid derived writes share one transaction.  The raw records and the
    # quarantine report remain untouched if a foreign-key/SQLite error happens.
    with connection:
        teams_by_id: dict[str, _Team] = {}
        for series in series_plans:
            for team in series.teams:
                teams_by_id[team.team_id] = team
        for item in map_plans:
            for team in item.teams:
                teams_by_id[team.team_id] = team
        for item in valid_rankings:
            teams_by_id[item.team.team_id] = item.team
        for team in teams_by_id.values():
            _insert_team(connection, team, snapshot_id, inserted)

        for item in series_plans:
            inserted["series"] += _rowcount_change(
                connection,
                """
                INSERT OR IGNORE INTO series (
                    series_id, source, source_series_id, scheduled_at,
                    started_at, ended_at, known_at, observed_at, best_of,
                    lan_online, event_name, stage_name, status, winner_team_id,
                    identity_confidence, source_snapshot_id
                ) VALUES (?, 'hltv', ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 'high', ?)
                """,
                (
                    item.series_id,
                    item.match_id,
                    item.scheduled_at,
                    item.envelope.known_at,
                    item.envelope.observed_at,
                    item.best_of,
                    item.lan_online,
                    item.event_name,
                    item.stage_name,
                    item.status,
                    snapshot_id,
                ),
            )
            for team_slot, team in enumerate(item.teams, start=1):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO series_participant (
                        series_id, team_id, team_slot, source_snapshot_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (item.series_id, team.team_id, team_slot, snapshot_id),
                )

        for item in map_plans:
            inserted["maps"] += _rowcount_change(
                connection,
                """
                INSERT OR IGNORE INTO map_game (
                    map_id, series_id, source_map_id, map_order, map_name,
                    game_version, ruleset, started_at, ended_at, known_at,
                    observed_at, team_a_id, team_b_id, score_a, score_b,
                    winner_team_id, picked_by_team_id, source_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.map_id,
                    item.series_id,
                    item.source_map_id,
                    item.map_order,
                    item.map_name,
                    item.game_version,
                    item.ruleset,
                    item.envelope.event_at,
                    item.envelope.known_at,
                    item.envelope.observed_at,
                    item.teams[0].team_id,
                    item.teams[1].team_id,
                    item.score_a,
                    item.score_b,
                    item.winner_team_id,
                    item.picked_by_team_id,
                    snapshot_id,
                ),
            )

        for item in valid_rankings:
            inserted["rankings"] += _rowcount_change(
                connection,
                """
                INSERT OR IGNORE INTO ranking_snapshot (
                    ranking_snapshot_id, ranking_system, team_id, rank, points,
                    published_at, known_at, observed_at, metric_version,
                    source_snapshot_id
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    item.ranking_snapshot_id,
                    item.ranking_system,
                    item.team.team_id,
                    item.rank,
                    item.envelope.known_at,
                    item.envelope.observed_at,
                    item.envelope.parser_version,
                    snapshot_id,
                ),
            )

        for item in valid_stats:
            _insert_player(
                connection,
                item.player_source_id,
                item.nickname,
                snapshot_id,
                inserted,
            )
            metrics = item.metrics
            inserted["player_map_stats"] += _rowcount_change(
                connection,
                """
                INSERT OR IGNORE INTO player_map_stats (
                    map_id, team_id, player_id, side, metric_version, known_at,
                    observed_at, kills, deaths, assists, flash_assists, headshots,
                    traded_deaths, opening_kills, opening_deaths, adr, kast, kpr,
                    dpr, swing, rating, metrics_json, source_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.map_id,
                    item.team_id,
                    item.player_id,
                    item.side,
                    item.metric_version,
                    item.envelope.known_at,
                    item.envelope.observed_at,
                    metrics["kills"],
                    metrics["deaths"],
                    metrics["assists"],
                    metrics["flash_assists"],
                    metrics["headshots"],
                    metrics["traded_deaths"],
                    metrics["opening_kills"],
                    metrics["opening_deaths"],
                    metrics["adr"],
                    metrics["kast"],
                    metrics["kpr"],
                    metrics["dpr"],
                    metrics["swing"],
                    metrics["rating"],
                    metrics["metrics_json"],
                    snapshot_id,
                ),
            )

    quarantined, quarantine_truncated = quarantine.as_result()
    accepted_records = (
        len(series_plans) + len(map_plans) + len(valid_rankings) + len(valid_stats)
    )
    return {
        "input_records": input_records,
        "duplicate_records_ignored": duplicate_records,
        "accepted_records": accepted_records,
        "inserted": inserted,
        "quarantined_count": quarantine.count,
        "quarantined": quarantined,
        "quarantined_truncated": quarantine_truncated,
        "source_snapshot_id": snapshot_id,
    }


def materialize_raw_stream(
    connection: sqlite3.Connection,
    *,
    source: str,
    stream: str,
    max_records: int = DEFAULT_MAX_RECORDS,
    after_raw_record_id: str | None = None,
    record_kinds: Sequence[str] | None = None,
    max_quarantine: int = DEFAULT_MAX_QUARANTINE,
) -> dict[str, object]:
    """Materialize one cursor-addressable, bounded raw JSONL batch.

    This wrapper reads only a ``max_records`` window from ``raw_ingest_record``
    and returns ``next_raw_record_id``/``has_more`` so a caller can drive a
    resumable job without an unbounded in-memory export.  For a very large
    stream, materialize ``('series',)`` first, then ``('map',)``, then rankings,
    then ``('player_map_stats',)``; map stats are intentionally quarantined
    until their map link is already verified.  Reset the cursor when changing
    the kind filter, because each phase is a separate ordered scan.
    """

    source = _nonempty_text(source, "source", max_length=256)
    stream = _nonempty_text(stream, "stream", max_length=256)
    if max_records < 1 or max_records > MAX_RECORDS:
        raise MaterializationError(
            f"max_records must be between 1 and {MAX_RECORDS}, inclusive"
        )
    cursor = (
        _nonempty_text(after_raw_record_id, "after_raw_record_id", max_length=512)
        if after_raw_record_id is not None
        else None
    )
    kinds: tuple[str, ...] | None = None
    if record_kinds is not None:
        kinds = tuple(
            _nonempty_text(value, "record_kinds[]", max_length=128)
            for value in record_kinds
        )
        if not kinds:
            raise MaterializationError("record_kinds cannot be empty when supplied")

    conditions = ["source = ?", "stream = ?"]
    values: list[object] = [source, stream]
    if cursor is not None:
        conditions.append("raw_record_id > ?")
        values.append(cursor)
    if kinds is not None:
        conditions.append("record_kind IN (" + ",".join("?" for _ in kinds) + ")")
        values.extend(kinds)
    where = " AND ".join(conditions)
    rows = connection.execute(
        """
        SELECT raw_record_id, source_record_id, record_kind, known_at,
               observed_at, payload_json, source_snapshot_id
        FROM raw_ingest_record
        WHERE """
        + where
        + " ORDER BY raw_record_id LIMIT ?",
        tuple([*values, max_records + 1]),
    ).fetchall()
    has_more = len(rows) > max_records
    rows = rows[:max_records]
    if not rows:
        return {
            "input_records": 0,
            "duplicate_records_ignored": 0,
            "accepted_records": 0,
            "inserted": {
                "teams": 0,
                "players": 0,
                "series": 0,
                "maps": 0,
                "rankings": 0,
                "player_map_stats": 0,
            },
            "quarantined_count": 0,
            "quarantined": [],
            "quarantined_truncated": False,
            "source_snapshot_id": None,
            "next_raw_record_id": None,
            "has_more": False,
        }

    snapshot_ids = {str(row["source_snapshot_id"]) for row in rows}
    if len(snapshot_ids) != 1:
        raise MaterializationError(
            "a raw stream window spans multiple source snapshots; use immutable source streams"
        )
    snapshot_id = next(iter(snapshot_ids))
    decoded: list[Mapping[str, object]] = []
    raw_quarantine = _Quarantine(max_quarantine)
    for row in rows:
        raw_id = str(row["raw_record_id"])
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as error:
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_payload_not_json",
                detail=str(error),
            )
            continue
        if not isinstance(payload, dict):
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_payload_not_object",
                detail="raw_ingest_record.payload_json is not an object",
            )
            continue
        if payload.get("record_id") != row["source_record_id"]:
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_record_id_mismatch",
                detail="payload record_id disagrees with raw_ingest_record source_record_id",
            )
            continue
        if payload.get("kind") != row["record_kind"]:
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_record_kind_mismatch",
                detail="payload kind disagrees with raw_ingest_record record_kind",
            )
            continue
        if payload.get("known_at") != row["known_at"]:
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_known_at_mismatch",
                detail="payload known_at disagrees with raw_ingest_record known_at",
            )
            continue
        payload = dict(payload)
        payload_observed_at = payload.get("observed_at")
        if payload_observed_at is None:
            payload["observed_at"] = str(row["observed_at"])
        elif payload_observed_at != row["observed_at"]:
            raw_quarantine.add(
                record_id=str(row["source_record_id"]),
                kind=str(row["record_kind"]),
                reason="raw_observed_at_mismatch",
                detail="payload observed_at disagrees with raw_ingest_record observed_at",
            )
            continue
        decoded.append(payload)

    result = materialize_records(
        connection,
        decoded,
        source_snapshot_id=snapshot_id,
        max_records=max_records,
        max_quarantine=max(0, max_quarantine - raw_quarantine.count),
    )
    prior_items, prior_truncated = raw_quarantine.as_result()
    existing_items = list(result["quarantined"])
    merged = (prior_items + existing_items)[:max_quarantine]
    result["input_records"] = len(rows)
    result["quarantined_count"] = int(result["quarantined_count"]) + raw_quarantine.count
    result["quarantined"] = merged
    result["quarantined_truncated"] = bool(
        prior_truncated
        or result["quarantined_truncated"]
        or int(result["quarantined_count"]) > len(merged)
    )
    result["next_raw_record_id"] = str(rows[-1]["raw_record_id"])
    result["has_more"] = has_more
    return result
