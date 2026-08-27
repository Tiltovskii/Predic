from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .counters import canonical_map_name

FORMAT_VERSION = "predic-light-argus-v2"
DEFAULT_STREAM = "bo3-history-2020-2026-v2"

EVENT_NUMERIC_FIELDS = (
    "kills_per_round",
    "deaths_per_round",
    "assists_per_round",
    "damage_per_round",
    "adr",
    "kast",
    "player_rating",
    "headshot_share",
    "opening_kills_per_round",
    "opening_deaths_per_round",
    "trade_kills_per_round",
    "trade_deaths_per_round",
    "hit_rate",
    "clutches_per_round",
    "log_equipment_per_round",
    "log_money_spent_per_round",
    "won_map",
    "participation_fraction",
    "rounds_participated",
)

SIDE_NUMERIC_SUFFIXES = (
    "elo",
    "log_matches",
    "career_win_rate",
    "career_map_win_rate",
    "days_since_match",
    "win_rate_30d",
    "map_win_rate_30d",
    "round_share_30d",
    "opponent_elo_30d",
    "win_rate_90d",
    "map_win_rate_90d",
    "round_share_90d",
    "opponent_elo_90d",
    "player_elo_mean",
    "player_elo_min",
    "player_elo_max",
    "player_matches_mean",
    "player_win_rate_mean",
    "player_days_inactive_mean",
    "lineup_prior_matches",
    "pair_experience_mean",
    "vrs_global_rank",
    "vrs_global_points",
    "vrs_global_age_days",
    "vrs_global_missing",
    "counter_player_inactivity_max",
    "counter_player_matches_min",
    "target_map_elo",
    "target_map_log_matches",
    "target_map_win_rate",
    "target_map_round_share",
    "target_map_days_since",
    "target_map_win_rate_90d",
    "target_map_round_share_90d",
)
SIDE_NUMERIC_FIELDS = SIDE_NUMERIC_SUFFIXES + ("target_map_pick",)
SHARED_NUMERIC_FIELDS = (
    "log_prize",
    "tournament_tier_rank",
    "round_index",
    "is_decider",
    "month_sin",
    "month_cos",
)

_TARGET_ARRAY_SPECS = {
    "target_history_indices": ("int32", (10,)),
    "target_players": ("int32", (10,)),
    "target_teams": ("int32", (2,)),
    "target_side_numeric": ("float32", (2, len(SIDE_NUMERIC_FIELDS))),
    "target_shared_numeric": ("float32", (len(SHARED_NUMERIC_FIELDS),)),
    "target_start_ts": ("int64", ()),
    "target_known_ts": ("int64", ()),
    "target_label": ("uint8", ()),
    "target_weight": ("float32", ()),
    "target_map": ("int16", ()),
    "target_tier": ("int16", ()),
    "target_event_type": ("int16", ()),
    "target_version": ("int16", ()),
    "target_bo_type": ("int16", ()),
    "target_role": ("int16", ()),
    "target_slot": ("int16", ()),
    "target_match_id": ("int64", ()),
}


@dataclass(frozen=True)
class _MatchContext:
    team1_id: int
    team2_id: int
    team1_name: str
    team2_name: str
    team1_roster: tuple[int, ...]
    team2_roster: tuple[int, ...]
    tier: str
    event_type: str
    game_version: str


def _timestamp(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value!r}")
    return int(parsed.astimezone(UTC).timestamp())


def _roster(value: str) -> tuple[int, ...]:
    raw = json.loads(value)
    if not isinstance(raw, list):
        return ()
    players: list[int] = []
    for item in raw:
        player_id = str(item).split(":", 1)[0]
        if not player_id.isdigit():
            return ()
        players.append(int(player_id))
    return tuple(players)


def _float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _ratio(numerator: object, denominator: object) -> float:
    top, bottom = _float(numerator), _float(denominator)
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0:
        return math.nan
    return top / bottom


def _vocab(values: set[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(["<UNK>", *sorted(values)])}


def _load_match_contexts(matches_csv: Path) -> dict[int, _MatchContext]:
    contexts: dict[int, _MatchContext] = {}
    with matches_csv.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            contexts[int(row["match_id"])] = _MatchContext(
                team1_id=int(row["team1_id"]),
                team2_id=int(row["team2_id"]),
                team1_name=row["team1_name"],
                team2_name=row["team2_name"],
                team1_roster=_roster(row["team1_roster"]),
                team2_roster=_roster(row["team2_roster"]),
                tier=(row.get("tournament_tier") or "unknown").casefold(),
                event_type=(row.get("event_type") or "unknown").casefold(),
                game_version=str(row.get("game_version") or "unknown"),
            )
    return contexts


def _event_numeric(row: dict[str, Any], rounds: float) -> list[float]:
    profile = row.get("steam_profile")
    participation = None
    if isinstance(profile, dict):
        raw_rounds = profile.get("game_round_steam_profiles")
        if isinstance(raw_rounds, list):
            participation = float(len(raw_rounds))
    played = participation if participation and participation > 0 else rounds
    equipment_per_round = _ratio(row.get("total_equipment_value"), played)
    money_per_round = _ratio(row.get("money_spent"), played)
    win = _float(row.get("win"))
    return [
        _ratio(row.get("kills"), played),
        _ratio(row.get("death"), played),
        _ratio(row.get("assists"), played),
        _ratio(row.get("damage"), played),
        _float(row.get("adr")),
        _float(row.get("kast")),
        _float(row.get("player_rating")),
        _ratio(row.get("headshots"), row.get("kills")),
        _ratio(row.get("first_kills"), played),
        _ratio(row.get("first_death"), played),
        _ratio(row.get("trade_kills"), played),
        _ratio(row.get("trade_death"), played),
        _ratio(row.get("hits"), row.get("shots")),
        _ratio(row.get("clutches"), played),
        math.log1p(max(0.0, equipment_per_round))
        if math.isfinite(equipment_per_round)
        else math.nan,
        math.log1p(max(0.0, money_per_round))
        if math.isfinite(money_per_round)
        else math.nan,
        win,
        played / rounds if rounds > 0 else math.nan,
        played,
    ]


def _grow(array: Any, capacity: int) -> Any:
    import numpy as np

    shape = (capacity, *array.shape[1:])
    grown = np.empty(shape, dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def _history_indices(
    player_offsets: Any,
    event_known_ts: Any,
    event_match_id: Any,
    player_ids: list[int],
    *,
    target_start_ts: int,
    target_match_id: int,
    max_history: int,
) -> Any:
    import numpy as np

    result = np.full((len(player_ids), max_history), -1, dtype=np.int32)
    for position, player_id in enumerate(player_ids):
        begin, finish = (
            int(player_offsets[player_id]),
            int(player_offsets[player_id + 1]),
        )
        eligible_end = begin + int(
            np.searchsorted(event_known_ts[begin:finish], target_start_ts, side="right")
        )
        selected = np.arange(
            max(begin, eligible_end - max_history), eligible_end, dtype=np.int32
        )
        if selected.size and np.any(event_match_id[selected] == target_match_id):
            raise ValueError(
                f"current match {target_match_id} leaked into a player history"
            )
        if selected.size:
            result[position, -len(selected) :] = selected
    return result


def _event_history_index_matrix(
    player_offsets: Any,
    event_known_ts: Any,
    event_start_ts: Any,
    event_match_id: Any,
    *,
    max_history: int,
) -> Any:
    """Build histories available at each event's pre-series prediction point."""
    import numpy as np

    result = np.full((len(event_known_ts), max_history), -1, dtype=np.int32)
    steps = np.arange(max_history, 0, -1, dtype=np.int64)
    for player_id in range(len(player_offsets) - 1):
        begin = int(player_offsets[player_id])
        finish = int(player_offsets[player_id + 1])
        if begin == finish:
            continue
        known = event_known_ts[begin:finish]
        starts = event_start_ts[begin:finish]
        eligible_ends = np.searchsorted(known, starts, side="right")
        local_indices = eligible_ends[:, None] - steps[None, :]
        valid = local_indices >= 0
        selected = np.where(valid, begin + local_indices, -1).astype(np.int32)
        selected_safe = np.maximum(selected, 0)
        if np.any(
            valid
            & (event_match_id[selected_safe] == event_match_id[begin:finish, None])
        ):
            raise ValueError("current match leaked into an event pretrain history")
        result[begin:finish] = selected
    return result


def _save_array(output: Path, name: str, value: Any) -> None:
    import numpy as np

    temporary = output / f"{name}.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(output / f"{name}.npy")


def build_light_argus_dataset(
    state_db: str | Path,
    matches_csv: str | Path,
    map_features_csv: str | Path,
    output_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
    stream: str = DEFAULT_STREAM,
    max_history: int = 32,
) -> dict[str, object]:
    """Build a compact causal player-history dataset for map prediction."""
    import numpy as np

    if max_history < 1:
        raise ValueError("max_history must be positive")
    state = Path(state_db).resolve()
    matches_path = Path(matches_csv).resolve()
    maps_path = Path(map_features_csv).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contexts = _load_match_contexts(matches_path)

    connection = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    job = connection.execute(
        "SELECT output_dir FROM bo3_capture_job WHERE stream = ?", (stream,)
    ).fetchone()
    if job is None:
        raise ValueError(f"unknown BO3 stream: {stream}")
    raw_root = Path(raw_dir).resolve() if raw_dir else Path(job[0]).resolve()

    player_values = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT steam_profile_id FROM bo3_player_map_index "
            "WHERE stream = ?",
            (stream,),
        )
    }
    team_values = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT team_id FROM bo3_player_map_index WHERE stream = ?",
            (stream,),
        )
    }
    for context in contexts.values():
        player_values.update(str(value) for value in context.team1_roster)
        player_values.update(str(value) for value in context.team2_roster)
        team_values.update((str(context.team1_id), str(context.team2_id)))
    player_vocab = _vocab(player_values)
    team_vocab = _vocab(team_values)
    map_values = {
        name
        for row in connection.execute(
            "SELECT DISTINCT map_name FROM bo3_game_index "
            "WHERE stream = ? AND map_name IS NOT NULL",
            (stream,),
        )
        if (name := canonical_map_name(row[0])) is not None
    }
    map_vocab = _vocab(map_values)
    tier_vocab = _vocab({context.tier for context in contexts.values()})
    event_type_vocab = _vocab({context.event_type for context in contexts.values()})
    version_vocab = _vocab({context.game_version for context in contexts.values()})
    bo_vocab = _vocab({"1", "3"})
    role_vocab = _vocab({"pick", "decider"})

    game_rows = connection.execute(
        """
        SELECT g.game_id, g.match_id, g.map_name, g.rounds_count,
               m.start_date, m.end_date, p.snapshot_id, s.object_path,
               p.expected_rows
        FROM bo3_game_index AS g
        JOIN bo3_match_index AS m
          ON m.stream = g.stream AND m.match_id = g.match_id
        JOIN (
            SELECT stream, game_id, MIN(snapshot_id) AS snapshot_id,
                   COUNT(*) AS expected_rows
            FROM bo3_player_map_index
            WHERE stream = ? AND training_metrics_complete = 1
            GROUP BY stream, game_id
        ) AS p ON p.stream = g.stream AND p.game_id = g.game_id
        JOIN bo3_snapshot AS s ON s.snapshot_id = p.snapshot_id
        WHERE g.stream = ? AND g.players_complete = 1
          AND m.end_date IS NOT NULL
        ORDER BY m.start_date, g.match_id, g.map_number, g.game_id
        """,
        (stream, stream),
    ).fetchall()
    event_capacity = int(sum(int(row["expected_rows"]) for row in game_rows))
    event_numeric = np.empty(
        (event_capacity, len(EVENT_NUMERIC_FIELDS)), dtype=np.float32
    )
    event_player = np.empty(event_capacity, dtype=np.int32)
    event_team = np.empty(event_capacity, dtype=np.int32)
    event_opponent = np.empty(event_capacity, dtype=np.int32)
    event_map = np.empty(event_capacity, dtype=np.int16)
    event_tier = np.empty(event_capacity, dtype=np.int16)
    event_event_type = np.empty(event_capacity, dtype=np.int16)
    event_version = np.empty(event_capacity, dtype=np.int16)
    event_start_ts = np.empty(event_capacity, dtype=np.int64)
    event_known_ts = np.empty(event_capacity, dtype=np.int64)
    event_match_id = np.empty(event_capacity, dtype=np.int64)
    event_game_id = np.empty(event_capacity, dtype=np.int64)
    event_count = 0
    skipped_events: dict[str, int] = {}

    def skip_event(reason: str, count: int = 1) -> None:
        skipped_events[reason] = skipped_events.get(reason, 0) + count

    for game_number, game in enumerate(game_rows, start=1):
        context = contexts.get(int(game["match_id"]))
        if context is None:
            skip_event("match_not_in_baseline", int(game["expected_rows"]))
            continue
        object_path = (raw_root / str(game["object_path"])).resolve()
        if not object_path.is_relative_to(raw_root):
            raise ValueError(f"snapshot path escapes raw root: {object_path}")
        try:
            with object_path.open(encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, json.JSONDecodeError):
            skip_event("unreadable_snapshot", int(game["expected_rows"]))
            continue
        if not isinstance(payload, list):
            skip_event("invalid_snapshot", int(game["expected_rows"]))
            continue
        teams = {
            int(clan["team_id"])
            for raw in payload
            if isinstance(raw, dict)
            and isinstance((clan := raw.get("team_clan")), dict)
            and isinstance(clan.get("team_id"), int)
        }
        if len(teams) != 2:
            skip_event("not_two_teams", len(payload))
            continue
        start_ts = _timestamp(str(game["start_date"]))
        known_ts = _timestamp(str(game["end_date"]))
        if known_ts <= start_ts:
            skip_event("non_positive_match_duration", len(payload))
            continue
        map_name = canonical_map_name(game["map_name"])
        rounds = float(game["rounds_count"] or 0)
        if map_name is None or rounds <= 0:
            skip_event("missing_map_or_rounds", len(payload))
            continue
        for raw in payload:
            if not isinstance(raw, dict):
                skip_event("invalid_player_row")
                continue
            required = ("kills", "death", "assists", "damage", "adr")
            if any(raw.get(field) is None for field in required):
                skip_event("missing_training_metric")
                continue
            profile_id = raw.get("steam_profile_id")
            clan = raw.get("team_clan")
            team_id = clan.get("team_id") if isinstance(clan, dict) else None
            if not isinstance(profile_id, int) or not isinstance(team_id, int):
                skip_event("missing_identity")
                continue
            opponents = teams - {team_id}
            if len(opponents) != 1:
                skip_event("invalid_opponent")
                continue
            if event_count >= event_capacity:
                raise ValueError("event allocation was smaller than parsed payloads")
            event_numeric[event_count] = _event_numeric(raw, rounds)
            event_player[event_count] = player_vocab.get(str(profile_id), 0)
            event_team[event_count] = team_vocab.get(str(team_id), 0)
            event_opponent[event_count] = team_vocab.get(str(opponents.pop()), 0)
            event_map[event_count] = map_vocab.get(map_name, 0)
            event_tier[event_count] = tier_vocab.get(context.tier, 0)
            event_event_type[event_count] = event_type_vocab.get(context.event_type, 0)
            event_version[event_count] = version_vocab.get(context.game_version, 0)
            event_start_ts[event_count] = start_ts
            event_known_ts[event_count] = known_ts
            event_match_id[event_count] = int(game["match_id"])
            event_game_id[event_count] = int(game["game_id"])
            event_count += 1
        if game_number % 10_000 == 0:
            print(
                f"light-argus events: games={game_number}/{len(game_rows)} "
                f"rows={event_count}",
                file=sys.stderr,
                flush=True,
            )
    connection.close()

    arrays = {
        "event_numeric": event_numeric[:event_count],
        "event_player": event_player[:event_count],
        "event_team": event_team[:event_count],
        "event_opponent": event_opponent[:event_count],
        "event_map": event_map[:event_count],
        "event_tier": event_tier[:event_count],
        "event_event_type": event_event_type[:event_count],
        "event_version": event_version[:event_count],
        "event_start_ts": event_start_ts[:event_count],
        "event_known_ts": event_known_ts[:event_count],
        "event_match_id": event_match_id[:event_count],
        "event_game_id": event_game_id[:event_count],
    }
    order = np.lexsort(
        (
            arrays["event_game_id"],
            arrays["event_match_id"],
            arrays["event_known_ts"],
            arrays["event_player"],
        )
    )
    for name in arrays:
        arrays[name] = arrays[name][order]
    player_offsets = np.zeros(len(player_vocab) + 1, dtype=np.int64)
    counts = np.bincount(arrays["event_player"], minlength=len(player_vocab)).astype(
        np.int64
    )
    player_offsets[1:] = np.cumsum(counts)
    arrays["player_offsets"] = player_offsets
    arrays["event_history_indices"] = _event_history_index_matrix(
        player_offsets,
        arrays["event_known_ts"],
        arrays["event_start_ts"],
        arrays["event_match_id"],
        max_history=max_history,
    )
    for name, value in arrays.items():
        _save_array(output, name, value)

    target_capacity = 100_000
    target_arrays: dict[str, Any] = {}
    for name, (dtype, suffix) in _TARGET_ARRAY_SPECS.items():
        shape = (target_capacity, *suffix)
        if name == "target_history_indices":
            shape = (target_capacity, 10, max_history)
        target_arrays[name] = np.empty(shape, dtype=dtype)
    target_count = 0
    skipped_targets: dict[str, int] = {}
    metadata_tmp = output / "target_metadata.jsonl.tmp"

    def skip_target(reason: str) -> None:
        skipped_targets[reason] = skipped_targets.get(reason, 0) + 1

    with (
        maps_path.open(encoding="utf-8", newline="") as source,
        metadata_tmp.open("w", encoding="utf-8") as metadata_stream,
    ):
        reader = csv.DictReader(source)
        required_columns = {
            f"{side}_{suffix}"
            for side in ("team1", "team2")
            for suffix in SIDE_NUMERIC_SUFFIXES
        }
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "map feature table misses Light Argus columns: "
                + ", ".join(sorted(missing_columns))
            )
        for row_number, row in enumerate(reader, start=1):
            match_id = int(row["match_id"])
            context = contexts.get(match_id)
            if context is None:
                skip_target("match_not_in_baseline")
                continue
            if len(context.team1_roster) != 5 or len(context.team2_roster) != 5:
                skip_target("not_full_5v5_roster")
                continue
            raw_players = [*context.team1_roster, *context.team2_roster]
            players = [player_vocab.get(str(value), 0) for value in raw_players]
            if any(value == 0 for value in players):
                skip_target("unknown_player")
                continue
            if not row.get("known_at"):
                skip_target("label_known_at_missing")
                continue
            start_ts = _timestamp(row["start_at"])
            known_ts = _timestamp(row["known_at"])
            if known_ts <= start_ts:
                raise ValueError(
                    f"target {row['map_row_id']} is known before it starts"
                )
            if target_count >= target_capacity:
                new_capacity = target_capacity * 2
                for name, value in target_arrays.items():
                    target_arrays[name] = _grow(value, new_capacity)
                target_capacity = new_capacity
            histories = _history_indices(
                player_offsets,
                arrays["event_known_ts"],
                arrays["event_match_id"],
                players,
                target_start_ts=start_ts,
                target_match_id=match_id,
                max_history=max_history,
            )
            target_arrays["target_history_indices"][target_count] = histories
            target_arrays["target_players"][target_count] = players
            target_arrays["target_teams"][target_count] = [
                team_vocab.get(str(context.team1_id), 0),
                team_vocab.get(str(context.team2_id), 0),
            ]
            for side_index, side in enumerate(("team1", "team2")):
                values = [
                    _float(row[f"{side}_{suffix}"]) for suffix in SIDE_NUMERIC_SUFFIXES
                ]
                values.append(_float(row[f"{side}_target_map_pick"]))
                target_arrays["target_side_numeric"][target_count, side_index] = values
            at = datetime.fromtimestamp(start_ts, UTC)
            target_arrays["target_shared_numeric"][target_count] = [
                _float(row.get("log_prize")),
                _float(row.get("tournament_tier_rank")),
                _float(row.get("round_index")),
                _float(row.get("is_decider")),
                math.sin(2.0 * math.pi * at.month / 12.0),
                math.cos(2.0 * math.pi * at.month / 12.0),
            ]
            map_name = canonical_map_name(row["target_map_name"]) or "<UNK>"
            target_arrays["target_start_ts"][target_count] = start_ts
            target_arrays["target_known_ts"][target_count] = known_ts
            target_arrays["target_label"][target_count] = int(row["team1_win"])
            target_arrays["target_weight"][target_count] = _float(
                row.get("sample_weight") or 1.0
            )
            target_arrays["target_map"][target_count] = map_vocab.get(map_name, 0)
            target_arrays["target_tier"][target_count] = tier_vocab.get(context.tier, 0)
            target_arrays["target_event_type"][target_count] = event_type_vocab.get(
                context.event_type, 0
            )
            target_arrays["target_version"][target_count] = version_vocab.get(
                context.game_version, 0
            )
            target_arrays["target_bo_type"][target_count] = bo_vocab.get(
                str(row["bo_type"]), 0
            )
            target_arrays["target_role"][target_count] = role_vocab.get(
                row["target_map_role"], 0
            )
            target_arrays["target_slot"][target_count] = int(row["target_map_slot"])
            target_arrays["target_match_id"][target_count] = match_id
            metadata_stream.write(
                json.dumps(
                    {
                        "row_index": target_count,
                        "map_row_id": row["map_row_id"],
                        "match_id": match_id,
                        "start_at": row["start_at"],
                        "known_at": row["known_at"],
                        "team1_id": context.team1_id,
                        "team2_id": context.team2_id,
                        "team1_name": context.team1_name,
                        "team2_name": context.team2_name,
                        "team1_win": int(row["team1_win"]),
                        "bo_type": int(row["bo_type"]),
                        "tournament_tier": context.tier,
                        "event_type": context.event_type,
                        "target_map_name": map_name,
                        "target_map_slot": int(row["target_map_slot"]),
                        "target_map_role": row["target_map_role"],
                        "team1_target_map_pick": int(row["team1_target_map_pick"]),
                        "team2_target_map_pick": int(row["team2_target_map_pick"]),
                        "target_map_decider": int(row["target_map_decider"]),
                        "diff_elo": _float(row["diff_elo"]),
                        "diff_target_map_elo": _float(row["diff_target_map_elo"]),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            target_count += 1
            if row_number % 10_000 == 0:
                print(
                    f"light-argus targets: source={row_number} rows={target_count}",
                    file=sys.stderr,
                    flush=True,
                )
    metadata_tmp.replace(output / "target_metadata.jsonl")
    for name, value in target_arrays.items():
        _save_array(output, name, value[:target_count])

    vocabularies = {
        "player": player_vocab,
        "team": team_vocab,
        "map": map_vocab,
        "tier": tier_vocab,
        "event_type": event_type_vocab,
        "version": version_vocab,
        "bo_type": bo_vocab,
        "role": role_vocab,
    }
    (output / "vocabularies.json").write_text(
        json.dumps(vocabularies, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stream": stream,
        "max_history": max_history,
        "events": event_count,
        "targets": target_count,
        "players": len(player_vocab) - 1,
        "teams": len(team_vocab) - 1,
        "maps": len(map_vocab) - 1,
        "event_numeric_fields": list(EVENT_NUMERIC_FIELDS),
        "side_numeric_fields": list(SIDE_NUMERIC_FIELDS),
        "shared_numeric_fields": list(SHARED_NUMERIC_FIELDS),
        "skipped_events": skipped_events,
        "skipped_targets": skipped_targets,
        "causal_contract": {
            "history_cutoff": "event known_at <= target start_at",
            "pretrain_history_cutoff": "history known_at <= event start_at",
            "current_series_excluded": True,
            "history_order": "known_at, match_id, game_id",
            "candidate_contains_outcome": False,
            "same_series_maps_share_pre_series_history": True,
        },
    }
    (output / "dataset.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
