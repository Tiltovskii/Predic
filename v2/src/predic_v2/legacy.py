from __future__ import annotations

import ast
import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


PARSER_VERSION = "legacy-csgo-csv-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug_id(namespace: str, value: str) -> str:
    return f"{namespace}:{_digest(value.casefold().strip())[:24]}"


def _players(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _integer(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class CanonicalLegacyMap:
    event_date: str
    team_a: str
    team_b: str
    players_a: tuple[str, ...]
    players_b: tuple[str, ...]
    score_a: int | None
    score_b: int | None
    rank_a: int | None
    rank_b: int | None
    mean_rating_a: float | None
    mean_rating_b: float | None
    map_name: str
    picked_by_a: bool | None
    target_a: float | None

    @property
    def identity(self) -> str:
        return _digest(
            self.event_date,
            self.team_a.casefold(),
            self.team_b.casefold(),
            self.map_name.casefold(),
            self.score_a,
            self.score_b,
            self.players_a,
            self.players_b,
        )


def _canonicalize(row: dict[str, str]) -> CanonicalLegacyMap:
    left_name = row.get("team_1", "").strip()
    right_name = row.get("team_2", "").strip()
    left_players = tuple(_players(row.get("players1", "")))
    right_players = tuple(_players(row.get("players2", "")))
    left_score = _integer(row.get("score_1"))
    right_score = _integer(row.get("score_2"))
    left_rank = _integer(row.get("team_rank_1"))
    right_rank = _integer(row.get("team_rank_2"))
    left_mean = _number(row.get("mean_rating1"))
    right_mean = _number(row.get("mean_rating2"))
    right_pick = _integer(row.get("right_pick"))
    target = _number(row.get("score"))

    left_first = (left_name.casefold(), left_name) <= (right_name.casefold(), right_name)
    if left_first:
        picked_by_a = None if right_pick is None else right_pick == 0
        target_a = target
        values = (
            left_name,
            right_name,
            left_players,
            right_players,
            left_score,
            right_score,
            left_rank,
            right_rank,
            left_mean,
            right_mean,
        )
    else:
        picked_by_a = None if right_pick is None else right_pick == 1
        target_a = None if target is None else 1.0 - target
        values = (
            right_name,
            left_name,
            right_players,
            left_players,
            right_score,
            left_score,
            right_rank,
            left_rank,
            right_mean,
            left_mean,
        )

    return CanonicalLegacyMap(
        event_date=row.get("date", "").strip(),
        team_a=values[0],
        team_b=values[1],
        players_a=values[2],
        players_b=values[3],
        score_a=values[4],
        score_b=values[5],
        rank_a=values[6],
        rank_b=values[7],
        mean_rating_a=values[8],
        mean_rating_b=values[9],
        map_name=row.get("Map", "UNKNOWN").strip() or "UNKNOWN",
        picked_by_a=picked_by_a,
        target_a=target_a,
    )


def _rows(path: Path, from_date: date | None) -> Iterable[CanonicalLegacyMap]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            item = _canonicalize(row)
            if not item.event_date or not item.team_a or not item.team_b:
                continue
            try:
                item_date = date.fromisoformat(item.event_date)
            except ValueError:
                continue
            if from_date is not None and item_date < from_date:
                continue
            if item.identity in seen:
                continue
            seen.add(item.identity)
            yield item


def _ensure_team(
    connection: sqlite3.Connection, snapshot_id: str, name: str
) -> str:
    team_id = _slug_id("legacy-team", name)
    connection.execute(
        """
        INSERT OR IGNORE INTO team_core (
            team_id, canonical_name, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'low', ?)
        """,
        (team_id, name, snapshot_id),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES ('legacy-predic-csv', 'team', ?, ?, ?)
        """,
        (name, team_id, snapshot_id),
    )
    return team_id


def _ensure_player(
    connection: sqlite3.Connection, snapshot_id: str, nickname: str
) -> str:
    player_id = _slug_id("legacy-player", nickname)
    connection.execute(
        """
        INSERT OR IGNORE INTO player (
            player_id, canonical_nickname, identity_confidence, source_snapshot_id
        ) VALUES (?, ?, 'low', ?)
        """,
        (player_id, nickname, snapshot_id),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_alias (
            source, entity_type, source_entity_id, canonical_entity_id,
            source_snapshot_id
        ) VALUES ('legacy-predic-csv', 'player', ?, ?, ?)
        """,
        (nickname, player_id, snapshot_id),
    )
    return player_id


def import_legacy_csv(
    connection: sqlite3.Connection,
    csv_path: str | Path,
    from_date: date | None = None,
) -> dict[str, int]:
    path = Path(csv_path).resolve()
    content_hash = _file_sha256(path)
    observed_at = _utc_now()
    snapshot_id = f"legacy:{content_hash[:32]}"
    connection.execute(
        """
        INSERT OR IGNORE INTO source_snapshot (
            snapshot_id, source, source_locator, observed_at, content_sha256,
            parser_version, license_ref, point_in_time_eligible, metadata_json
        ) VALUES (?, 'legacy-predic-csv', ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            snapshot_id,
            str(path),
            observed_at,
            content_hash,
            PARSER_VERSION,
            "Predic repository LICENSE; upstream data provenance not established",
            json.dumps(
                {
                    "identity_confidence": "low",
                    "point_in_time_note": (
                        "The file was reconstructed after the events; historical "
                        "feature availability is not established."
                    ),
                },
                sort_keys=True,
            ),
        ),
    )

    counts = {"maps": 0, "lineup_members": 0, "rankings": 0}
    for item in _rows(path, from_date):
        team_a_id = _ensure_team(connection, snapshot_id, item.team_a)
        team_b_id = _ensure_team(connection, snapshot_id, item.team_b)
        # The legacy CSV has no stable match/series identifier. Keeping each map
        # in its own low-confidence series avoids silently merging two BO1/BO3
        # matches played by the same teams on the same date.
        series_key = item.identity
        series_id = f"legacy-series:{series_key[:32]}"
        winner_id = None
        if item.score_a is not None and item.score_b is not None:
            if item.score_a > item.score_b:
                winner_id = team_a_id
            elif item.score_b > item.score_a:
                winner_id = team_b_id
        connection.execute(
            """
            INSERT OR IGNORE INTO series (
                series_id, source, source_series_id, started_at, ended_at,
                known_at, observed_at, status, winner_team_id,
                identity_confidence, source_snapshot_id
            ) VALUES (?, 'legacy-predic-csv', ?, ?, ?, ?, ?, 'finished', ?, 'low', ?)
            """,
            (
                series_id,
                series_key,
                item.event_date,
                item.event_date,
                None,
                observed_at,
                None,
                snapshot_id,
            ),
        )

        map_id = f"legacy-map:{item.identity[:32]}"
        inserted = connection.execute(
            """
            INSERT OR IGNORE INTO map_game (
                map_id, series_id, source_map_id, map_order, map_name,
                game_version, ruleset, started_at, ended_at, known_at,
                observed_at, team_a_id, team_b_id, score_a, score_b,
                winner_team_id, picked_by_team_id, legacy_target,
                source_snapshot_id
            ) VALUES (?, ?, ?, ?, ?, 'CSGO', 'MR15', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                map_id,
                series_id,
                item.identity,
                1,
                item.map_name,
                item.event_date,
                item.event_date,
                None,
                observed_at,
                team_a_id,
                team_b_id,
                item.score_a,
                item.score_b,
                winner_id,
                team_a_id if item.picked_by_a is True else team_b_id if item.picked_by_a is False else None,
                item.target_a,
                snapshot_id,
            ),
        ).rowcount
        if not inserted:
            continue
        counts["maps"] += 1

        for team_id, players in (
            (team_a_id, item.players_a),
            (team_b_id, item.players_b),
        ):
            for slot, nickname in enumerate(players, start=1):
                player_id = _ensure_player(connection, snapshot_id, nickname)
                counts["lineup_members"] += connection.execute(
                    """
                    INSERT OR IGNORE INTO lineup_member (
                        map_id, team_id, player_id, slot, member_type,
                        known_at, actual_at, source_snapshot_id
                    ) VALUES (?, ?, ?, ?, 'starter', ?, ?, ?)
                    """,
                    (
                        map_id,
                        team_id,
                        player_id,
                        slot,
                        None,
                        item.event_date,
                        snapshot_id,
                    ),
                ).rowcount

        for team_id, rank, mean_rating, side in (
            (team_a_id, item.rank_a, item.mean_rating_a, "a"),
            (team_b_id, item.rank_b, item.mean_rating_b, "b"),
        ):
            if rank is None and mean_rating is None:
                continue
            ranking_id = f"legacy-ranking:{_digest(map_id, side)[:32]}"
            counts["rankings"] += connection.execute(
                """
                INSERT OR IGNORE INTO ranking_snapshot (
                    ranking_snapshot_id, ranking_system, team_id, rank, points,
                    published_at, known_at, observed_at, metric_version,
                    source_snapshot_id
                ) VALUES (?, 'legacy-row-rank', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ranking_id,
                    team_id,
                    rank,
                    mean_rating,
                    None,
                    None,
                    observed_at,
                    PARSER_VERSION,
                    snapshot_id,
                ),
            ).rowcount

    connection.commit()
    return counts
