from __future__ import annotations

import bisect
import csv
import heapq
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .counters import EnrichedCounterStore, strict_veto_complete

_STREAM = "bo3-history-2020-2026-v2"
_WINDOWS = (30, 90, 180)
_TIER_WEIGHTS = {
    "s": 1.0,
    "a": 0.80,
    "b": 0.55,
    "c": 0.35,
    "d": 0.20,
    "unknown": 0.30,
}


def _normalise(value: str) -> str:
    value = value.casefold().replace("&", "and")
    value = re.sub(r"\b(team|esports|gaming|club)\b", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _json_tuple(value: str) -> tuple[str, ...]:
    raw = json.loads(value)
    return tuple(str(item) for item in raw)


def _extract_veto_actions(payload: dict[str, object]) -> list[dict[str, object]]:
    """Keep only immutable pick/ban fields from the match-level veto payload."""
    raw_actions = payload.get("match_maps") or []
    if not isinstance(raw_actions, list):
        return []
    actions: list[dict[str, object]] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            return []
        maps = raw.get("maps") or {}
        if not isinstance(maps, dict):
            return []
        map_name = maps.get("map_name") or maps.get("slug")
        try:
            order = int(raw.get("order") or 0)
            choice_type = int(raw.get("choice_type") or 0)
            team_id = int(raw.get("team_id") or 0)
        except (TypeError, ValueError):
            return []
        if not map_name or choice_type not in {1, 2, 3}:
            return []
        actions.append(
            {
                "order": order,
                "choice_type": choice_type,
                "team_id": team_id,
                "map_name": str(map_name),
            }
        )
    return sorted(actions, key=lambda item: (int(item["order"]), item["map_name"]))


@dataclass(frozen=True)
class _ExternalRanking:
    rank: int
    points: float
    team_name: str
    roster: frozenset[str]
    published_at: datetime


class ExternalRankingIndex:
    def __init__(self, csv_path: str | Path | None):
        self._dates: dict[str, list[datetime]] = defaultdict(list)
        self._snapshots: dict[tuple[str, datetime], list[_ExternalRanking]] = (
            defaultdict(list)
        )
        if csv_path is None:
            return
        with Path(csv_path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                system = row["ranking_system"]
                published = _parse_time(row["published_at"] + "T00:00:00Z")
                ranking = _ExternalRanking(
                    rank=int(row["rank"]),
                    points=float(row["points"]),
                    team_name=row["team_name"],
                    roster=frozenset(
                        _normalise(player) for player in _json_tuple(row["roster"])
                    ),
                    published_at=published,
                )
                self._snapshots[(system, published)].append(ranking)
        for system, published in self._snapshots:
            self._dates[system].append(published)
        for system in self._dates:
            self._dates[system] = sorted(set(self._dates[system]))

    def lookup(
        self,
        system: str,
        prediction_at: datetime,
        team_name: str,
        roster: Iterable[str],
    ) -> dict[str, float]:
        dates = self._dates.get(system, [])
        # Snapshot dates have no publication time. Using the previous date for
        # same-day matches is the conservative, no-future choice.
        prediction_day = prediction_at.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        position = bisect.bisect_left(dates, prediction_day) - 1
        if position < 0:
            return _missing_ranking()
        published = dates[position]
        snapshot = self._snapshots[(system, published)]
        wanted_name = _normalise(team_name)
        wanted_roster = frozenset(_normalise(player) for player in roster if player)
        named = [row for row in snapshot if _normalise(row.team_name) == wanted_name]
        candidates = named or snapshot
        scored = [(len(wanted_roster & row.roster), row) for row in candidates]
        overlap, best = max(scored, default=(0, None), key=lambda item: item[0])
        if best is None or (not named and overlap < 3):
            return _missing_ranking()
        confidence = 1.0 if named and overlap >= 3 else 0.85 if named else overlap / 5.0
        return {
            "rank": float(best.rank),
            "points": best.points,
            "age_days": (prediction_at - published).total_seconds() / 86_400,
            "confidence": confidence,
            "roster_overlap": float(overlap),
            "missing": 0.0,
        }


def _missing_ranking() -> dict[str, float]:
    return {
        "rank": math.nan,
        "points": math.nan,
        "age_days": math.nan,
        "confidence": 0.0,
        "roster_overlap": 0.0,
        "missing": 1.0,
    }


def _load_lineups(
    connection: sqlite3.Connection, stream: str
) -> dict[tuple[int, int], tuple[str, ...]]:
    rows = connection.execute(
        """
        SELECT g.match_id, p.team_id, p.steam_profile_id,
               MAX(COALESCE(p.nickname, '')) AS nickname,
               COUNT(DISTINCT g.game_id) AS maps_played,
               SUM(COALESCE(p.rounds_participated, 0)) AS rounds_played
        FROM bo3_player_map_index AS p
        JOIN bo3_game_index AS g
          ON g.stream = p.stream AND g.game_id = p.game_id
        WHERE p.stream = ?
        GROUP BY g.match_id, p.team_id, p.steam_profile_id
        ORDER BY g.match_id, p.team_id, rounds_played DESC,
                 maps_played DESC, p.steam_profile_id
        """,
        (stream,),
    )
    grouped: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for match_id, team_id, player_id, nickname, _, _ in rows:
        grouped[(int(match_id), int(team_id))].append((int(player_id), str(nickname)))
    # Keep the five principal participants. `current_is_coach` is deliberately
    # ignored: it describes the profile now, not that player's historical role.
    # With substitutions/coach rows, query ordering retains the five who
    # participated in the most rounds, then maps.
    return {
        key: tuple(f"{player_id}:{nickname}" for player_id, nickname in players[:5])
        for key, players in grouped.items()
    }


def extract_bo3_match_table(
    state_db: str | Path,
    output_csv: str | Path,
    *,
    stream: str = _STREAM,
) -> dict[str, object]:
    """Create a compact series table without reading round or player metrics."""

    state = Path(state_db).resolve()
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state)
    connection.row_factory = sqlite3.Row
    lineups = _load_lineups(connection, stream)
    rows = connection.execute(
        """
        SELECT m.*, s.object_path, j.output_dir
        FROM bo3_match_index AS m
        JOIN bo3_snapshot AS s ON s.snapshot_id = m.last_snapshot_id
        JOIN bo3_capture_job AS j ON j.stream = m.stream
        WHERE m.stream = ? AND m.detail_complete = 1 AND m.status = 'finished'
        ORDER BY m.start_date, m.match_id
        """,
        (stream,),
    ).fetchall()
    fields = [
        "match_id",
        "start_at",
        "end_at",
        "team1_id",
        "team1_name",
        "team1_roster",
        "team2_id",
        "team2_name",
        "team2_roster",
        "winner_team_id",
        "team1_win",
        "bo_type",
        "game_version",
        "tournament_id",
        "tournament_name",
        "tournament_tier",
        "tournament_tier_rank",
        "event_type",
        "round_index",
        "bracket_type",
        "is_decider",
        "prize",
        "maps_played",
        "team1_map_wins",
        "team2_map_wins",
        "team1_rounds",
        "team2_rounds",
        "rounds_known",
        "map_results",
        "veto_actions",
        "score_label",
    ]
    skipped = defaultdict(int)
    written = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            path = Path(row["output_dir"]) / row["object_path"]
            try:
                payload = json.loads(path.read_bytes())
            except (OSError, json.JSONDecodeError):
                skipped["unreadable_payload"] += 1
                continue
            team1 = payload.get("team1") or {}
            team2 = payload.get("team2") or {}
            end_at = payload.get("end_date") or row["end_date"]
            if not end_at:
                skipped["missing_end_at"] += 1
                continue
            team1_id = int(team1.get("id") or row["team1_id"] or 0)
            team2_id = int(team2.get("id") or row["team2_id"] or 0)
            winner_id = payload.get("winner_team_id")
            if (
                not team1_id
                or not team2_id
                or team1_id == team2_id
                or winner_id not in {team1_id, team2_id}
            ):
                skipped["invalid_result"] += 1
                continue
            games = [
                game for game in (payload.get("games") or []) if isinstance(game, dict)
            ]
            map_wins = {team1_id: 0, team2_id: 0}
            team_rounds = {team1_id: 0, team2_id: 0}
            map_results: list[dict[str, object]] = []
            rounds_known = True
            for game in games:
                winner = ((game.get("winner_team_clan") or {}).get("team") or {}).get(
                    "id"
                )
                loser = ((game.get("loser_team_clan") or {}).get("team") or {}).get(
                    "id"
                )
                if winner in map_wins:
                    map_wins[int(winner)] += 1
                winner_score = game.get("winner_clan_score")
                loser_score = game.get("loser_clan_score")
                if winner in team_rounds and loser in team_rounds:
                    map_results.append(
                        {
                            "map_name": game.get("map_name") or "unknown",
                            "winner_team_id": int(winner),
                            "loser_team_id": int(loser),
                            "winner_score": winner_score,
                            "loser_score": loser_score,
                        }
                    )
                if (
                    winner in team_rounds
                    and loser in team_rounds
                    and isinstance(winner_score, int)
                    and isinstance(loser_score, int)
                ):
                    team_rounds[int(winner)] += winner_score
                    team_rounds[int(loser)] += loser_score
                else:
                    rounds_known = False
            score_label = ""
            if games and sum(map_wins.values()) == len(games):
                score_label = f"{map_wins[team1_id]}-{map_wins[team2_id]}"
            tournament = payload.get("tournament") or {}
            tournament_round = payload.get("round") or {}
            writer.writerow(
                {
                    "match_id": int(row["match_id"]),
                    "start_at": payload.get("start_date") or row["start_date"],
                    "end_at": end_at,
                    "team1_id": team1_id,
                    "team1_name": team1.get("name") or str(team1_id),
                    "team1_roster": json.dumps(
                        lineups.get((int(row["match_id"]), team1_id), ())
                    ),
                    "team2_id": team2_id,
                    "team2_name": team2.get("name") or str(team2_id),
                    "team2_roster": json.dumps(
                        lineups.get((int(row["match_id"]), team2_id), ())
                    ),
                    "winner_team_id": int(winner_id),
                    "team1_win": int(winner_id == team1_id),
                    "bo_type": payload.get("bo_type") or row["bo_type"] or 0,
                    "game_version": payload.get("game_version")
                    or row["game_version"]
                    or 0,
                    "tournament_id": tournament.get("id") or 0,
                    "tournament_name": tournament.get("name") or "unknown",
                    "tournament_tier": str(
                        tournament.get("tier") or "unknown"
                    ).casefold(),
                    "tournament_tier_rank": tournament.get("tier_rank") or 0,
                    "event_type": tournament.get("event_type") or "unknown",
                    "round_index": tournament_round.get("round_index") or 0,
                    "bracket_type": tournament_round.get("bracket_type") or "unknown",
                    "is_decider": int(bool(tournament_round.get("is_decider"))),
                    "prize": tournament.get("prize") or 0,
                    "maps_played": len(games),
                    "team1_map_wins": map_wins[team1_id],
                    "team2_map_wins": map_wins[team2_id],
                    "team1_rounds": team_rounds[team1_id] if rounds_known else "",
                    "team2_rounds": team_rounds[team2_id] if rounds_known else "",
                    "rounds_known": int(rounds_known and bool(games)),
                    "map_results": json.dumps(map_results, separators=(",", ":")),
                    "veto_actions": json.dumps(
                        _extract_veto_actions(payload), separators=(",", ":")
                    ),
                    "score_label": score_label,
                }
            )
            written += 1
            if index % 10_000 == 0:
                print(
                    f"extracted {index}/{len(rows)} matches",
                    file=sys.stderr,
                    flush=True,
                )
    temporary.replace(output)
    connection.close()
    return {
        "output_csv": str(output),
        "source_matches": len(rows),
        "written_matches": written,
        "matches_with_lineups": len({match_id for match_id, _ in lineups}),
        "skipped": dict(skipped),
    }


@dataclass
class _Outcome:
    at: datetime
    win: float
    map_win_rate: float | None
    round_share: float | None
    opponent_elo: float


@dataclass
class _TeamState:
    elo: float = 1500.0
    matches: int = 0
    wins: float = 0.0
    maps_won: int = 0
    maps_played: int = 0
    last_at: datetime | None = None
    outcomes: deque[_Outcome] = field(default_factory=deque)


@dataclass
class _PlayerState:
    elo: float = 1500.0
    matches: int = 0
    wins: float = 0.0
    last_at: datetime | None = None


@dataclass
class _PendingMatchUpdate:
    known_at: datetime
    match: dict[str, str]
    roster1: tuple[str, ...]
    roster2: tuple[str, ...]
    signature1: tuple[str, ...]
    signature2: tuple[str, ...]
    team1_elo: float
    team2_elo: float
    team1_elo_delta: float
    player_deltas1: tuple[tuple[str, float], ...]
    player_deltas2: tuple[tuple[str, float], ...]


def _smoothed_rate(successes: float, total: float) -> float:
    return (successes + 2.0) / (total + 4.0)


def _team_features(state: _TeamState, at: datetime) -> dict[str, float]:
    result = {
        "elo": state.elo,
        "matches": float(state.matches),
        "log_matches": math.log1p(state.matches),
        "career_win_rate": _smoothed_rate(state.wins, state.matches),
        "career_map_win_rate": _smoothed_rate(state.maps_won, state.maps_played),
        "days_since_match": 999.0
        if state.last_at is None
        else min(999.0, (at - state.last_at).total_seconds() / 86_400),
    }
    for window in _WINDOWS:
        recent = [
            item
            for item in state.outcomes
            if (at - item.at).total_seconds() < window * 86_400
        ]
        count = len(recent)
        result[f"matches_{window}d"] = float(count)
        result[f"win_rate_{window}d"] = _smoothed_rate(
            sum(item.win for item in recent), count
        )
        known_maps = [
            item.map_win_rate for item in recent if item.map_win_rate is not None
        ]
        result[f"map_win_rate_{window}d"] = _smoothed_rate(
            sum(known_maps), len(known_maps)
        )
        result[f"map_win_rate_matches_{window}d"] = float(len(known_maps))
        known_rounds = [
            item.round_share for item in recent if item.round_share is not None
        ]
        result[f"round_share_{window}d"] = _smoothed_rate(
            sum(known_rounds), len(known_rounds)
        )
        result[f"round_share_matches_{window}d"] = float(len(known_rounds))
        result[f"opponent_elo_{window}d"] = (
            sum(item.opponent_elo for item in recent) / count if count else 1500.0
        )
    return result


def _player_features(
    roster: tuple[str, ...], players: dict[str, _PlayerState], at: datetime
) -> dict[str, float]:
    states = [players[player.split(":", 1)[0]] for player in roster]
    if not states:
        return {
            "roster_size": 0.0,
            "player_elo_mean": 1500.0,
            "player_elo_min": 1500.0,
            "player_elo_max": 1500.0,
            "player_elo_std": 0.0,
            "player_matches_mean": 0.0,
            "player_win_rate_mean": 0.5,
            "player_days_inactive_mean": 999.0,
        }
    elos = [state.elo for state in states]
    mean = sum(elos) / len(elos)
    inactivity = [
        999.0
        if state.last_at is None
        else min(999.0, (at - state.last_at).total_seconds() / 86_400)
        for state in states
    ]
    return {
        "roster_size": float(len(states)),
        "player_elo_mean": mean,
        "player_elo_min": min(elos),
        "player_elo_max": max(elos),
        "player_elo_std": math.sqrt(sum((elo - mean) ** 2 for elo in elos) / len(elos)),
        "player_matches_mean": sum(state.matches for state in states) / len(states),
        "player_win_rate_mean": sum(
            _smoothed_rate(state.wins, state.matches) for state in states
        )
        / len(states),
        "player_days_inactive_mean": sum(inactivity) / len(inactivity),
    }


def _prefix(target: dict[str, object], prefix: str, values: dict[str, object]) -> None:
    target.update({f"{prefix}_{key}": value for key, value in values.items()})


def build_point_in_time_features(
    matches_csv: str | Path,
    output_csv: str | Path,
    *,
    rankings_csv: str | Path | None = None,
) -> dict[str, object]:
    ranking_index = ExternalRankingIndex(rankings_csv)
    enriched_counters = EnrichedCounterStore()
    with Path(matches_csv).open(encoding="utf-8", newline="") as source:
        matches = list(csv.DictReader(source))
    matches.sort(key=lambda row: (row["start_at"], int(row["match_id"])))
    teams: dict[int, _TeamState] = defaultdict(_TeamState)
    players: dict[str, _PlayerState] = defaultdict(_PlayerState)
    lineup_counts: dict[tuple[int, tuple[str, ...]], int] = defaultdict(int)
    pair_counts: dict[tuple[int, str, str], int] = defaultdict(int)
    h2h: dict[tuple[int, int], deque[tuple[datetime, int]]] = defaultdict(deque)
    feature_rows: list[dict[str, object]] = []
    pending_updates: list[tuple[datetime, int, _PendingMatchUpdate]] = []

    def apply_update(pending: _PendingMatchUpdate) -> None:
        match = pending.match
        known_at = pending.known_at
        team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
        team1, team2 = teams[team1_id], teams[team2_id]
        outcome1 = int(match["team1_win"])
        enriched_counters.update(
            match,
            known_at,
            pending.roster1,
            pending.roster2,
            team1_elo=pending.team1_elo,
            team2_elo=pending.team2_elo,
            team1_elo_delta=pending.team1_elo_delta,
        )
        team1.elo += pending.team1_elo_delta
        team2.elo -= pending.team1_elo_delta
        map_total = int(match["team1_map_wins"]) + int(match["team2_map_wins"])
        map_result_valid = map_total > 0 and map_total == int(match["maps_played"])
        map_rate1 = (
            int(match["team1_map_wins"]) / map_total if map_result_valid else None
        )
        map_rate2 = None if map_rate1 is None else 1.0 - map_rate1
        round_share1: float | None = None
        if int(match["rounds_known"]):
            rounds1, rounds2 = int(match["team1_rounds"]), int(match["team2_rounds"])
            if rounds1 + rounds2:
                round_share1 = rounds1 / (rounds1 + rounds2)
        for state, win, map_rate, round_share, opponent_elo in (
            (team1, float(outcome1), map_rate1, round_share1, pending.team2_elo),
            (
                team2,
                float(1 - outcome1),
                map_rate2,
                None if round_share1 is None else 1.0 - round_share1,
                pending.team1_elo,
            ),
        ):
            state.matches += 1
            state.wins += win
            if map_rate is not None:
                state.maps_won += round(map_rate * map_total)
                state.maps_played += map_total
            state.last_at = known_at
            state.outcomes.append(
                _Outcome(known_at, win, map_rate, round_share, opponent_elo)
            )
            while (
                state.outcomes
                and (known_at - state.outcomes[0].at).total_seconds() >= 365 * 86_400
            ):
                state.outcomes.popleft()
        for side_deltas, win in (
            (pending.player_deltas1, outcome1),
            (pending.player_deltas2, 1 - outcome1),
        ):
            for player_id, delta in side_deltas:
                state = players[player_id]
                state.elo += delta
                state.matches += 1
                state.wins += win
                state.last_at = known_at
        for team_id, roster, signature in (
            (team1_id, pending.roster1, pending.signature1),
            (team2_id, pending.roster2, pending.signature2),
        ):
            if signature:
                lineup_counts[(team_id, signature)] += 1
            ids = sorted(player.split(":", 1)[0] for player in roster)
            for left in range(len(ids)):
                for right in range(left + 1, len(ids)):
                    pair_counts[(team_id, ids[left], ids[right])] += 1
        pair_key = tuple(sorted((team1_id, team2_id)))
        h2h[pair_key].append(
            (
                known_at,
                outcome1 if pair_key[0] == team1_id else 1 - outcome1,
            )
        )
        while (
            h2h[pair_key]
            and (known_at - h2h[pair_key][0][0]).total_seconds() >= 365 * 86_400
        ):
            h2h[pair_key].popleft()

    for index, match in enumerate(matches, start=1):
        at = _parse_time(match["start_at"])
        raw_end_at = _parse_time(match["end_at"])
        known_at = raw_end_at if raw_end_at > at else None
        while pending_updates and pending_updates[0][0] <= at:
            _, _, pending = heapq.heappop(pending_updates)
            apply_update(pending)
        team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
        team1, team2 = teams[team1_id], teams[team2_id]
        roster1, roster2 = (
            _json_tuple(match["team1_roster"]),
            _json_tuple(match["team2_roster"]),
        )
        veto_known = int(
            int(match["bo_type"]) in {1, 3} and strict_veto_complete(match)
        )
        player1, player2 = (
            _player_features(roster1, players, at),
            _player_features(roster2, players, at),
        )
        features: dict[str, object] = {
            "match_id": int(match["match_id"]),
            "start_at": match["start_at"],
            # Label-availability metadata. It is excluded from every model input
            # and is used only to enforce temporal training cutoffs.
            "known_at": known_at.isoformat() if known_at is not None else "",
            "veto_known": veto_known,
            "team1_id": str(team1_id),
            "team2_id": str(team2_id),
            "team1_name": match["team1_name"],
            "team2_name": match["team2_name"],
            "bo_type": str(match["bo_type"]),
            "game_version": str(match["game_version"]),
            "tournament_tier": match["tournament_tier"],
            "event_type": match["event_type"],
            "bracket_type": match.get("bracket_type") or "unknown",
            "round_index": float(match.get("round_index") or 0),
            "is_decider": int(match.get("is_decider") or 0),
            "tournament_tier_rank": float(match["tournament_tier_rank"] or 0),
            "log_prize": math.log1p(float(match["prize"] or 0)),
            "year": at.year,
            "month": at.month,
            "team1_win": int(match["team1_win"]),
            "score_label": match["score_label"],
            "maps_played": int(match["maps_played"]),
            "team1_rounds": int(match["team1_rounds"] or 0),
            "team2_rounds": int(match["team2_rounds"] or 0),
            "rounds_known": int(match["rounds_known"]),
            "round_share_target": (
                int(match["team1_rounds"])
                / (int(match["team1_rounds"]) + int(match["team2_rounds"]))
                if int(match["rounds_known"])
                and int(match["team1_rounds"]) + int(match["team2_rounds"])
                else ""
            ),
            "sample_weight": _TIER_WEIGHTS.get(
                match["tournament_tier"], _TIER_WEIGHTS["unknown"]
            ),
        }
        team1_features, team2_features = (
            _team_features(team1, at),
            _team_features(team2, at),
        )
        _prefix(features, "team1", team1_features)
        _prefix(features, "team2", team2_features)
        _prefix(features, "team1", player1)
        _prefix(features, "team2", player2)
        for key in sorted(team1_features):
            features[f"diff_{key}"] = team1_features[key] - team2_features[key]
        for key in sorted(player1):
            features[f"diff_{key}"] = player1[key] - player2[key]

        signature1 = tuple(sorted(player.split(":", 1)[0] for player in roster1))
        signature2 = tuple(sorted(player.split(":", 1)[0] for player in roster2))
        features["team1_lineup_prior_matches"] = (
            lineup_counts[(team1_id, signature1)] if signature1 else 0
        )
        features["team2_lineup_prior_matches"] = (
            lineup_counts[(team2_id, signature2)] if signature2 else 0
        )
        features["diff_lineup_prior_matches"] = float(
            features["team1_lineup_prior_matches"]
        ) - float(features["team2_lineup_prior_matches"])
        for side, team_id, roster in (
            ("team1", team1_id, roster1),
            ("team2", team2_id, roster2),
        ):
            ids = sorted(player.split(":", 1)[0] for player in roster)
            pairs = [
                pair_counts[(team_id, ids[left], ids[right])]
                for left in range(len(ids))
                for right in range(left + 1, len(ids))
            ]
            features[f"{side}_pair_experience_mean"] = (
                sum(pairs) / len(pairs) if pairs else 0.0
            )
        features["diff_pair_experience_mean"] = float(
            features["team1_pair_experience_mean"]
        ) - float(features["team2_pair_experience_mean"])

        pair_key = tuple(sorted((team1_id, team2_id)))
        prior_h2h = [
            item
            for item in h2h[pair_key]
            if (at - item[0]).total_seconds() < 365 * 86_400
        ]
        team1_h2h_wins = sum(
            result if pair_key[0] == team1_id else 1 - result for _, result in prior_h2h
        )
        features["h2h_matches_365d"] = len(prior_h2h)
        team1_h2h_rate = _smoothed_rate(team1_h2h_wins, len(prior_h2h))
        features["team1_h2h_win_rate_365d"] = team1_h2h_rate
        features["team2_h2h_win_rate_365d"] = 1.0 - team1_h2h_rate
        features["diff_h2h_win_rate_365d"] = 2.0 * team1_h2h_rate - 1.0

        for system in ("valve_regional", "valve_global"):
            rank1 = ranking_index.lookup(
                system,
                at,
                match["team1_name"],
                (item.split(":", 1)[-1] for item in roster1),
            )
            rank2 = ranking_index.lookup(
                system,
                at,
                match["team2_name"],
                (item.split(":", 1)[-1] for item in roster2),
            )
            short = "vrs_regional" if system == "valve_regional" else "vrs_global"
            _prefix(features, f"team1_{short}", rank1)
            _prefix(features, f"team2_{short}", rank2)
            features[f"diff_{short}_rank"] = rank2["rank"] - rank1["rank"]
            features[f"diff_{short}_points"] = rank1["points"] - rank2["points"]

        features.update(enriched_counters.features(match, at, roster1, roster2))

        feature_rows.append(features)

        # The result becomes eligible only when the match has ended. This also
        # freezes simultaneous/overlapping matches against one another.
        outcome1 = int(match["team1_win"])
        expected1 = 1.0 / (1.0 + 10.0 ** ((team2.elo - team1.elo) / 400.0))
        tier_weight = _TIER_WEIGHTS.get(
            match["tournament_tier"], _TIER_WEIGHTS["unknown"]
        )
        margin = abs(int(match["team1_map_wins"]) - int(match["team2_map_wins"]))
        delta = (
            28.0
            * (0.75 + 0.5 * tier_weight)
            * (1.0 + 0.12 * margin)
            * (outcome1 - expected1)
        )
        pre1, pre2 = team1.elo, team2.elo
        player_deltas: list[tuple[tuple[str, float], ...]] = []
        for roster, win, opponent_mean in (
            (roster1, outcome1, player2["player_elo_mean"]),
            (roster2, 1 - outcome1, player1["player_elo_mean"]),
        ):
            side_deltas: list[tuple[str, float]] = []
            for player in roster:
                player_id = player.split(":", 1)[0]
                state = players[player_id]
                expected = 1.0 / (1.0 + 10.0 ** ((opponent_mean - state.elo) / 400.0))
                side_deltas.append((player_id, 16.0 * (win - expected)))
            player_deltas.append(tuple(side_deltas))
        # A positive recorded end time is kept even when the series is unusually
        # long. Moving it earlier would expose the result before it was known.
        # Non-positive durations cannot be ordered safely, so their result never
        # updates causal state (the row can still be evaluated as a prediction).
        if known_at is not None:
            pending = _PendingMatchUpdate(
                known_at=known_at,
                match=match,
                roster1=roster1,
                roster2=roster2,
                signature1=tuple(sorted(player.split(":", 1)[0] for player in roster1)),
                signature2=tuple(sorted(player.split(":", 1)[0] for player in roster2)),
                team1_elo=pre1,
                team2_elo=pre2,
                team1_elo_delta=delta,
                player_deltas1=player_deltas[0],
                player_deltas2=player_deltas[1],
            )
            heapq.heappush(pending_updates, (known_at, index, pending))
        if index % 10_000 == 0:
            print(
                f"built {index}/{len(matches)} feature rows",
                file=sys.stderr,
                flush=True,
            )

    while pending_updates:
        _, _, pending = heapq.heappop(pending_updates)
        apply_update(pending)

    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = list(feature_rows[0]) if feature_rows else []
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)
    temporary.replace(output)
    return {
        "output_csv": str(output),
        "rows": len(feature_rows),
        "features": len(fieldnames),
        "teams": len(teams),
        "players": len(players),
    }


def _auc(y: Any, probability: Any) -> float:
    pairs = sorted(zip(probability, y), key=lambda item: item[0])
    positives = sum(int(item) for item in y)
    negatives = len(y) - positives
    if not positives or not negatives:
        return math.nan
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(int(pairs[pos][1]) for pos in range(index, end))
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _binary_metrics(
    y: Any, probability: Any, weights: Any | None = None
) -> dict[str, float]:
    import numpy as np

    y = np.asarray(y, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    logloss = -np.sum(
        weights * (y * np.log(probability) + (1 - y) * np.log(1 - probability))
    )
    brier = np.sum(weights * (probability - y) ** 2)
    accuracy = np.sum(weights * ((probability >= 0.5) == y))
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1)
        if mask.any():
            bin_weight = weights[mask].sum()
            ece += bin_weight * abs(
                np.average(probability[mask], weights=weights[mask])
                - np.average(y[mask], weights=weights[mask])
            )
    return {
        "rows": len(y),
        "positive_rate": float(np.average(y, weights=weights)),
        "logloss": float(logloss),
        "brier": float(brier),
        "accuracy": float(accuracy),
        "auc": float(_auc(y, probability)),
        "ece_10bin": float(ece),
    }


def _regression_metrics(y: Any, prediction: Any) -> dict[str, float]:
    import numpy as np

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - y
    return {
        "rows": len(y),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_target": float(np.mean(y)),
        "mean_prediction": float(np.mean(prediction)),
    }


def _confidence_slices(y: Any, probability: Any) -> dict[str, dict[str, float]]:
    import numpy as np

    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = (probability >= 0.5).astype(int) == y
    result: dict[str, dict[str, float]] = {}
    for threshold in (0.60, 0.65, 0.70, 0.75):
        mask = confidence >= threshold
        key = f"confidence_at_least_{threshold:.2f}"
        result[key] = {
            "rows": int(mask.sum()),
            "coverage": float(mask.mean()),
            "accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
            "mean_confidence": float(confidence[mask].mean())
            if mask.any()
            else math.nan,
        }
    return result


def _mirror(frame: Any, feature_columns: list[str]) -> Any:
    mirrored = frame.copy()
    for column in feature_columns:
        if column.startswith("team1_"):
            counterpart = "team2_" + column[len("team1_") :]
            if counterpart in frame.columns:
                mirrored[column] = frame[counterpart]
        elif column.startswith("team2_"):
            counterpart = "team1_" + column[len("team2_") :]
            if counterpart in frame.columns:
                mirrored[column] = frame[counterpart]
        elif column.startswith("diff_"):
            mirrored[column] = -frame[column]
    mirrored["team1_win"] = 1 - frame["team1_win"]
    if "round_share_target" in frame:
        mirrored["round_share_target"] = 1.0 - frame["round_share_target"]
    score_swap = {"2-0": "0-2", "2-1": "1-2", "1-2": "2-1", "0-2": "2-0"}
    mirrored["score_label"] = (
        frame["score_label"].map(score_swap).fillna(frame["score_label"])
    )
    return mirrored


def _select_feature_columns(
    frame: Any, excluded: set[str], feature_set: str
) -> list[str]:
    all_columns = [column for column in frame.columns if column not in excluded]
    veto_columns = [
        column
        for column in all_columns
        if column.startswith("veto_") or "_veto_" in column
    ]
    columns = [column for column in all_columns if column not in veto_columns]
    include_veto = feature_set == "core-veto"
    base_set = "core" if include_veto else feature_set
    if base_set == "all":
        return columns
    if base_set == "base":
        return [
            column
            for column in columns
            if "_counter_" not in column and not column.startswith("counter_")
        ]
    if base_set != "core":
        raise ValueError(f"unknown feature set: {feature_set}")
    support_tokens = (
        "matches",
        "known",
        "effective",
        "roster_size",
        "roster_full5",
        "newcomers",
        "days_since",
        "tenure",
        "inactivity",
        "maps_played",
        "map_pool_maps",
        "map_pool_games",
    )
    result: list[str] = []
    for column in columns:
        is_counter = "_counter_" in column or column.startswith("counter_")
        if (
            not is_counter
            or column.startswith(("diff_counter_", "counter_"))
            or any(token in column for token in support_tokens)
        ):
            result.append(column)
    if include_veto:
        result.extend(veto_columns)
    return result


def _labels_known_before(frame: Any, cutoff: Any) -> Any:
    """Return only rows whose result timestamp is strictly before a cutoff."""
    return frame[frame["known_at"].notna() & (frame["known_at"] < cutoff)]


def _categorical_feature_columns(feature_columns: list[str]) -> list[str]:
    fixed = {
        "team1_id",
        "team2_id",
        "bo_type",
        "game_version",
        "tournament_tier",
        "event_type",
        "bracket_type",
    }
    veto_map = re.compile(
        r"(?:team[12]_veto_(?:pick|ban)_\d+_map|"
        r"veto_decider_map|veto_selected_map_\d+)"
    )
    return [
        column
        for column in feature_columns
        if column in fixed or veto_map.fullmatch(column)
    ]


def _parse_feature_times(frame: Any, pandas: Any) -> None:
    """Parse mixed ISO precision without silently discarding valid timestamps."""
    frame["start_at"] = pandas.to_datetime(frame["start_at"], utc=True, format="mixed")
    frame["known_at"] = pandas.to_datetime(
        frame["known_at"], utc=True, format="mixed", errors="coerce"
    )


def train_catboost_baseline(
    features_csv: str | Path,
    output_dir: str | Path,
    *,
    train_before: str = "2025-01-01",
    test_from: str = "2026-01-01",
    iterations: int = 900,
    feature_set: str = "core",
    veto_known_only: bool = False,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    frame = pd.read_csv(features_csv, low_memory=False)
    _parse_feature_times(frame, pd)
    veto_known_only = veto_known_only or feature_set == "core-veto"
    if veto_known_only:
        if "veto_known" not in frame:
            raise ValueError("veto-known filtering requires a veto_known column")
        frame = frame[frame.veto_known == 1].copy()
    train_cut = pd.Timestamp(train_before, tz="UTC")
    test_cut = pd.Timestamp(test_from, tz="UTC")
    train = _labels_known_before(frame, train_cut)
    train = train[train.start_at < train_cut].copy()
    validation = _labels_known_before(frame, test_cut)
    validation = validation[
        (validation.start_at >= train_cut) & (validation.start_at < test_cut)
    ].copy()
    test = frame[frame.start_at >= test_cut].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("temporal split produced an empty partition")

    excluded = {
        "match_id",
        "start_at",
        "known_at",
        "veto_known",
        "team1_name",
        "team2_name",
        "team1_win",
        "score_label",
        "maps_played",
        "team1_rounds",
        "team2_rounds",
        "rounds_known",
        "round_share_target",
        "sample_weight",
    }
    feature_columns = _select_feature_columns(frame, excluded, feature_set)
    categorical = _categorical_feature_columns(feature_columns)
    for partition in (train, validation, test):
        for column in categorical:
            partition[column] = partition[column].fillna("missing").astype(str)
    augmented_train = pd.concat(
        [train, _mirror(train, feature_columns)], ignore_index=True
    )
    augmented_validation = pd.concat(
        [validation, _mirror(validation, feature_columns)], ignore_index=True
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        random_seed=42,
        thread_count=-1,
        verbose=100,
        allow_writing_files=False,
    )
    train_pool = Pool(
        augmented_train[feature_columns],
        augmented_train.team1_win,
        cat_features=categorical,
        weight=augmented_train.sample_weight,
    )
    validation_pool = Pool(
        augmented_validation[feature_columns],
        augmented_validation.team1_win,
        cat_features=categorical,
        weight=augmented_validation.sample_weight,
    )
    model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=100)

    def symmetric_probability(partition: Any) -> Any:
        direct = model.predict_proba(partition[feature_columns])[:, 1]
        mirrored = _mirror(partition, feature_columns)
        inverse = 1.0 - model.predict_proba(mirrored[feature_columns])[:, 1]
        return (direct + inverse) / 2.0

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "winner_catboost.cbm")
    probability = symmetric_probability(test)
    elo_probability = 1.0 / (
        1.0 + 10.0 ** (-test["diff_elo"].to_numpy(dtype=float) / 400.0)
    )
    train_prior = float(train.team1_win.mean())
    metrics: dict[str, Any] = {
        "split": {
            "train_before": train_before,
            "test_from": test_from,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_rows_after_mirroring": len(augmented_train),
            "feature_set": feature_set,
            "feature_count": len(feature_columns),
            "veto_known_only": veto_known_only,
        },
        "winner": {
            "overall": _binary_metrics(test.team1_win, probability),
            "tier_weighted": _binary_metrics(
                test.team1_win, probability, test.sample_weight
            ),
            "tier_s_a": _binary_metrics(
                test[test.tournament_tier.isin(["s", "a"])].team1_win,
                probability[test.tournament_tier.isin(["s", "a"])],
            )
            if test.tournament_tier.isin(["s", "a"]).any()
            else None,
            "confidence_slices": _confidence_slices(test.team1_win, probability),
            "tier_s_a_confidence_slices": _confidence_slices(
                test[test.tournament_tier.isin(["s", "a"])].team1_win,
                probability[test.tournament_tier.isin(["s", "a"])],
            )
            if test.tournament_tier.isin(["s", "a"]).any()
            else None,
            "best_iteration": model.get_best_iteration(),
            "baselines": {
                "constant_0_5": _binary_metrics(
                    test.team1_win, np.full(len(test), 0.5)
                ),
                "source_side_train_prior": _binary_metrics(
                    test.team1_win, np.full(len(test), train_prior)
                ),
                "dynamic_elo_only": _binary_metrics(test.team1_win, elo_probability),
            },
        },
    }

    score_classes = ["0-2", "1-2", "2-0", "2-1"]
    score_train = train[
        (train.bo_type.astype(str) == "3") & train.score_label.isin(score_classes)
    ].copy()
    score_validation = validation[
        (validation.bo_type.astype(str) == "3")
        & validation.score_label.isin(score_classes)
    ].copy()
    score_test = test[
        (test.bo_type.astype(str) == "3") & test.score_label.isin(score_classes)
    ].copy()
    if len(score_train) and len(score_validation) and len(score_test):
        score_test["_winner_probability"] = probability[
            test.index.get_indexer(score_test.index)
        ]
        metrics["winner"]["bo3_exact_score_subset"] = _binary_metrics(
            score_test.team1_win, score_test["_winner_probability"]
        )
        augmented_score_train = pd.concat(
            [score_train, _mirror(score_train, feature_columns)], ignore_index=True
        )
        augmented_score_validation = pd.concat(
            [score_validation, _mirror(score_validation, feature_columns)],
            ignore_index=True,
        )
        score_model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=max(500, iterations - 150),
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=7.0,
            random_strength=0.5,
            random_seed=43,
            thread_count=-1,
            verbose=100,
            allow_writing_files=False,
        )
        score_model.fit(
            Pool(
                augmented_score_train[feature_columns],
                augmented_score_train.score_label,
                cat_features=categorical,
                weight=augmented_score_train.sample_weight,
            ),
            eval_set=Pool(
                augmented_score_validation[feature_columns],
                augmented_score_validation.score_label,
                cat_features=categorical,
                weight=augmented_score_validation.sample_weight,
            ),
            early_stopping_rounds=100,
        )
        score_model.save_model(output / "bo3_score_catboost.cbm")
        class_order = list(score_model.classes_)
        class_to_index = {label: index for index, label in enumerate(class_order)}
        direct_score_probability = score_model.predict_proba(
            score_test[feature_columns]
        )
        mirrored_score_test = _mirror(score_test, feature_columns)
        mirror_probability = score_model.predict_proba(
            mirrored_score_test[feature_columns]
        )
        swap = {"2-0": "0-2", "2-1": "1-2", "1-2": "2-1", "0-2": "2-0"}
        inverse_probability = np.column_stack(
            [
                mirror_probability[:, class_to_index[swap[label]]]
                for label in class_order
            ]
        )
        score_probability = (direct_score_probability + inverse_probability) / 2.0
        true_index = np.array(
            [class_to_index[label] for label in score_test.score_label]
        )
        predicted = np.array(class_order)[score_probability.argmax(axis=1)]
        exact_accuracy = float(np.mean(predicted == score_test.score_label.to_numpy()))
        score_logloss = float(
            -np.mean(
                np.log(
                    np.clip(
                        score_probability[np.arange(len(score_test)), true_index],
                        1e-7,
                        1,
                    )
                )
            )
        )
        implied_winner_probability = sum(
            score_probability[:, class_to_index[label]] for label in ("2-0", "2-1")
        )
        train_score_prior = score_train.score_label.value_counts(normalize=True)
        prior_probability = np.column_stack(
            [
                np.full(len(score_test), train_score_prior.get(label, 0.0))
                for label in class_order
            ]
        )
        prior_probability /= prior_probability.sum(axis=1, keepdims=True)
        prior_logloss = float(
            -np.mean(
                np.log(
                    np.clip(
                        prior_probability[np.arange(len(score_test)), true_index],
                        1e-7,
                        1,
                    )
                )
            )
        )
        metrics["bo3_exact_score"] = {
            "rows": len(score_test),
            "classes": class_order,
            "accuracy": exact_accuracy,
            "logloss": score_logloss,
            "implied_winner": _binary_metrics(
                score_test.team1_win, implied_winner_probability
            ),
            "best_iteration": score_model.get_best_iteration(),
            "train_prior_baseline": {
                "accuracy": float(
                    np.mean(score_test.score_label == train_score_prior.idxmax())
                ),
                "logloss": prior_logloss,
                "majority_class": str(train_score_prior.idxmax()),
            },
        }
        score_predictions = score_test[
            ["match_id", "start_at", "team1_name", "team2_name", "score_label"]
        ].copy()
        score_predictions["predicted_score"] = predicted
        for label in class_order:
            score_predictions[f"probability_{label}"] = score_probability[
                :, class_to_index[label]
            ]
        score_predictions.to_csv(output / "bo3_score_test_predictions.csv", index=False)

    ratio_train = train[train.round_share_target.notna()].copy()
    ratio_validation = validation[validation.round_share_target.notna()].copy()
    ratio_test = test[test.round_share_target.notna()].copy()
    if len(ratio_train) and len(ratio_validation) and len(ratio_test):
        augmented_ratio_train = pd.concat(
            [ratio_train, _mirror(ratio_train, feature_columns)], ignore_index=True
        )
        augmented_ratio_validation = pd.concat(
            [ratio_validation, _mirror(ratio_validation, feature_columns)],
            ignore_index=True,
        )
        ratio_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=iterations,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=7.0,
            random_strength=0.5,
            random_seed=44,
            thread_count=-1,
            verbose=100,
            allow_writing_files=False,
        )
        ratio_model.fit(
            Pool(
                augmented_ratio_train[feature_columns],
                augmented_ratio_train.round_share_target,
                cat_features=categorical,
                weight=augmented_ratio_train.sample_weight,
            ),
            eval_set=Pool(
                augmented_ratio_validation[feature_columns],
                augmented_ratio_validation.round_share_target,
                cat_features=categorical,
                weight=augmented_ratio_validation.sample_weight,
            ),
            early_stopping_rounds=100,
        )
        ratio_model.save_model(output / "round_share_catboost.cbm")
        direct_ratio = ratio_model.predict(ratio_test[feature_columns])
        mirrored_ratio_test = _mirror(ratio_test, feature_columns)
        inverse_ratio = 1.0 - ratio_model.predict(mirrored_ratio_test[feature_columns])
        ratio_prediction = np.clip((direct_ratio + inverse_ratio) / 2.0, 0.0, 1.0)
        target_direction = ratio_test.round_share_target.to_numpy() >= 0.5
        winner = ratio_test.team1_win.to_numpy(dtype=int)
        metrics["round_share_regression"] = {
            **_regression_metrics(
                ratio_test.round_share_target.to_numpy(), ratio_prediction
            ),
            "best_iteration": ratio_model.get_best_iteration(),
            "winner_accuracy_at_0_5": float(
                np.mean((ratio_prediction >= 0.5) == winner)
            ),
            "target_winner_direction_agreement": float(
                np.mean(target_direction == winner)
            ),
            "constant_0_5": _regression_metrics(
                ratio_test.round_share_target.to_numpy(),
                np.full(len(ratio_test), 0.5),
            ),
        }
        ratio_predictions = ratio_test[
            [
                "match_id",
                "start_at",
                "team1_name",
                "team2_name",
                "team1_win",
                "round_share_target",
            ]
        ].copy()
        ratio_predictions["round_share_prediction"] = ratio_prediction
        ratio_predictions.to_csv(
            output / "round_share_test_predictions.csv", index=False
        )

    predictions = test[
        [
            "match_id",
            "start_at",
            "team1_name",
            "team2_name",
            "team1_win",
            "tournament_tier",
        ]
    ].copy()
    predictions["team1_win_probability"] = probability
    predictions.to_csv(output / "test_predictions.csv", index=False)
    importance = model.get_feature_importance(prettified=True)
    importance.to_csv(output / "winner_feature_importance.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def walk_forward_catboost_backtest(
    features_csv: str | Path,
    output_dir: str | Path,
    *,
    test_from: str = "2026-01-01",
    validation_days: int = 90,
    iterations: int = 900,
    feature_set: str = "core",
    veto_known_only: bool = False,
) -> dict[str, object]:
    """Simulate a monthly production retrain without consuming future labels."""
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    frame = pd.read_csv(features_csv, low_memory=False)
    _parse_feature_times(frame, pd)
    veto_known_only = veto_known_only or feature_set == "core-veto"
    if veto_known_only:
        if "veto_known" not in frame:
            raise ValueError("veto-known filtering requires a veto_known column")
        frame = frame[frame.veto_known == 1].copy()
    test_cut = pd.Timestamp(test_from, tz="UTC")
    if test_cut.day != 1:
        raise ValueError("test_from must be the first day of a month")
    if validation_days < 1:
        raise ValueError("validation_days must be positive")

    excluded = {
        "match_id",
        "start_at",
        "known_at",
        "veto_known",
        "team1_name",
        "team2_name",
        "team1_win",
        "score_label",
        "maps_played",
        "team1_rounds",
        "team2_rounds",
        "rounds_known",
        "round_share_target",
        "sample_weight",
    }
    feature_columns = _select_feature_columns(frame, excluded, feature_set)
    categorical = _categorical_feature_columns(feature_columns)
    for column in categorical:
        frame[column] = frame[column].fillna("missing").astype(str)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict[str, object]] = []
    fold_predictions: list[Any] = []
    latest_model: Any | None = None
    last_match_at = frame.start_at.max()
    fold_starts = pd.date_range(test_cut, last_match_at, freq="MS")

    def new_model(model_iterations: int, seed: int) -> Any:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=model_iterations,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=6.0,
            random_strength=0.5,
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )

    for fold_number, fold_start in enumerate(fold_starts, start=1):
        fold_end = fold_start + pd.offsets.MonthBegin(1)
        history = _labels_known_before(frame, fold_start).copy()
        validation_start = fold_start - pd.Timedelta(days=validation_days)
        tuning_train = _labels_known_before(history, validation_start).copy()
        tuning_validation = history[
            (history.start_at >= validation_start) & (history.start_at < fold_start)
        ].copy()
        fold_test = frame[
            (frame.start_at >= fold_start) & (frame.start_at < fold_end)
        ].copy()
        if min(len(tuning_train), len(tuning_validation), len(fold_test)) == 0:
            continue

        augmented_train = pd.concat(
            [tuning_train, _mirror(tuning_train, feature_columns)],
            ignore_index=True,
        )
        augmented_validation = pd.concat(
            [tuning_validation, _mirror(tuning_validation, feature_columns)],
            ignore_index=True,
        )
        tuning_model = new_model(iterations, 100 + fold_number)
        tuning_model.fit(
            Pool(
                augmented_train[feature_columns],
                augmented_train.team1_win,
                cat_features=categorical,
                weight=augmented_train.sample_weight,
            ),
            eval_set=Pool(
                augmented_validation[feature_columns],
                augmented_validation.team1_win,
                cat_features=categorical,
                weight=augmented_validation.sample_weight,
            ),
            early_stopping_rounds=100,
        )
        best_iteration = tuning_model.get_best_iteration()
        selected_iterations = best_iteration + 1 if best_iteration >= 0 else iterations

        # Once the tree count is selected only from past data, refit on every
        # result available before the prediction month, including validation.
        augmented_history = pd.concat(
            [history, _mirror(history, feature_columns)], ignore_index=True
        )
        latest_model = new_model(selected_iterations, 200 + fold_number)
        latest_model.fit(
            Pool(
                augmented_history[feature_columns],
                augmented_history.team1_win,
                cat_features=categorical,
                weight=augmented_history.sample_weight,
            )
        )
        direct = latest_model.predict_proba(fold_test[feature_columns])[:, 1]
        mirrored_test = _mirror(fold_test, feature_columns)
        inverse = 1.0 - latest_model.predict_proba(mirrored_test[feature_columns])[:, 1]
        probability = (direct + inverse) / 2.0
        current_metrics = _binary_metrics(fold_test.team1_win, probability)
        fold_metrics.append(
            {
                "month": fold_start.strftime("%Y-%m"),
                "history_rows": len(history),
                "validation_rows": len(tuning_validation),
                "selected_iterations": selected_iterations,
                **current_metrics,
            }
        )
        predictions = fold_test[
            [
                "match_id",
                "start_at",
                "team1_name",
                "team2_name",
                "team1_win",
                "tournament_tier",
            ]
        ].copy()
        predictions["team1_win_probability"] = probability
        predictions["training_cutoff"] = fold_start.isoformat()
        fold_predictions.append(predictions)
        print(
            f"walk-forward {fold_start:%Y-%m}: "
            f"rows={len(fold_test)} accuracy={current_metrics['accuracy']:.4f} "
            f"iterations={selected_iterations}",
            file=sys.stderr,
            flush=True,
        )

    if not fold_predictions or latest_model is None:
        raise ValueError("walk-forward split produced no folds")
    predictions = pd.concat(fold_predictions, ignore_index=True)
    overall = _binary_metrics(predictions.team1_win, predictions.team1_win_probability)
    metrics: dict[str, object] = {
        "protocol": {
            "test_from": test_from,
            "retrain": "monthly_full_refit",
            "validation_days": validation_days,
            "future_labels_used": False,
            "feature_set": feature_set,
            "feature_count": len(feature_columns),
            "veto_known_only": veto_known_only,
        },
        "overall": overall,
        "confidence_slices": _confidence_slices(
            predictions.team1_win, predictions.team1_win_probability
        ),
        "folds": fold_metrics,
    }
    latest_model.save_model(output / "winner_catboost_walk_forward_latest.cbm")
    predictions.to_csv(output / "walk_forward_test_predictions.csv", index=False)
    (output / "walk_forward_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics
