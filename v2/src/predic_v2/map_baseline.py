from __future__ import annotations

import csv
import heapq
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .baseline import (
    _TIER_WEIGHTS,
    _binary_metrics,
    _categorical_feature_columns,
    _confidence_slices,
    _labels_known_before,
    _mirror,
    _parse_feature_times,
    _parse_time,
    _select_feature_columns,
)
from .counters import (
    RoundStyleOutcome,
    canonical_map_name,
    round_stats_for_team,
    round_style_features,
    strict_veto_complete,
)
from .weighting import (
    TIER_WEIGHT_PROFILES,
    effective_tier_weight_mass,
    tier_weight,
)

_MAP_WINDOWS = (30, 90, 180, 365)
_MAP_LAST_COUNTS = (5, 10)
_MAP_HALF_LIVES = (30, 90)
_SERIES_EXCLUDED = {
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
_MAP_EXCLUDED = {
    "map_row_id",
    "match_id",
    "start_at",
    "known_at",
    "team1_name",
    "team2_name",
    "team1_win",
    "team1_map_score",
    "team2_map_score",
    "map_scores_known",
    "sample_weight",
}


def _smoothed_rate(successes: float, total: float, prior: float = 0.5) -> float:
    return (successes + 4.0 * prior) / (total + 4.0)


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _age_days(at: datetime, previous: datetime | None) -> float:
    if previous is None:
        return 999.0
    return min(999.0, max(0.0, (at - previous).total_seconds() / 86_400.0))


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


@dataclass(frozen=True)
class _TargetMap:
    map_name: str
    veto_slot: int
    role: str
    picked_by_team1: int
    picked_by_team2: int


@dataclass(frozen=True)
class _PlayedMap:
    map_name: str
    winner_team_id: int
    loser_team_id: int
    winner_score: int | None
    loser_score: int | None
    winner_round_stats: dict[str, float] | None = None
    loser_round_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class _MapOutcome:
    at: datetime
    win: float
    round_share: float | None
    round_stats: dict[str, float] | None


@dataclass
class _MapState:
    elo: float = 1500.0
    matches: int = 0
    wins: float = 0.0
    round_share_sum: float = 0.0
    round_known: int = 0
    last_at: datetime | None = None
    outcomes: deque[_MapOutcome] = field(default_factory=deque)

    def add(
        self,
        at: datetime,
        win: float,
        round_share: float | None,
        elo_delta: float,
        round_stats: dict[str, float] | None = None,
    ) -> None:
        self.elo += elo_delta
        self.matches += 1
        self.wins += win
        if round_share is not None:
            self.round_share_sum += round_share
            self.round_known += 1
        self.last_at = at
        self.outcomes.append(_MapOutcome(at, win, round_share, round_stats))
        while self.outcomes and _age_days(at, self.outcomes[0].at) >= 365:
            self.outcomes.popleft()


def _strict_target_maps(match: dict[str, str]) -> tuple[_TargetMap, ...]:
    if not strict_veto_complete(match):
        return ()
    try:
        actions = json.loads(match.get("veto_actions") or "[]")
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(actions, list):
        return ()
    team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
    selected: list[tuple[int, int, int | None, str]] = []
    for raw in sorted(actions, key=lambda item: int(item["order"])):
        choice_type = _strict_int(raw.get("choice_type"))
        if choice_type not in {1, 3}:
            continue
        map_name = canonical_map_name(raw.get("map_name"))
        if map_name is None:
            return ()
        selected.append(
            (
                int(raw["order"]),
                choice_type,
                _strict_int(raw.get("team_id")),
                map_name,
            )
        )
    targets: list[_TargetMap] = []
    for slot, (_, choice_type, team_id, map_name) in enumerate(selected, start=1):
        targets.append(
            _TargetMap(
                map_name=map_name,
                veto_slot=slot,
                role="pick" if choice_type == 1 else "decider",
                picked_by_team1=int(choice_type == 1 and team_id == team1_id),
                picked_by_team2=int(choice_type == 1 and team_id == team2_id),
            )
        )
    bo_type = int(match["bo_type"])
    expected = 1 if bo_type == 1 else 3 if bo_type == 3 else 0
    return tuple(targets) if len(targets) == expected else ()


def _played_maps(match: dict[str, str]) -> tuple[_PlayedMap, ...]:
    try:
        raw_results = json.loads(match.get("map_results") or "[]")
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(raw_results, list):
        return ()
    team_ids = {int(match["team1_id"]), int(match["team2_id"])}
    results: list[_PlayedMap] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        map_name = canonical_map_name(raw.get("map_name"))
        winner = _strict_int(raw.get("winner_team_id"))
        loser = _strict_int(raw.get("loser_team_id"))
        if map_name is None or {winner, loser} != team_ids:
            continue
        winner_score = _strict_int(raw.get("winner_score"))
        loser_score = _strict_int(raw.get("loser_score"))
        if winner_score is None or loser_score is None:
            winner_score = loser_score = None
        results.append(
            _PlayedMap(
                map_name,
                int(winner),
                int(loser),
                winner_score,
                loser_score,
                round_stats_for_team(raw, int(winner)),
                round_stats_for_team(raw, int(loser)),
            )
        )
    return tuple(results)


def _eligible_target_rows(
    match: dict[str, str],
) -> tuple[tuple[_TargetMap, _PlayedMap], ...]:
    targets = _strict_target_maps(match)
    results = _played_maps(match)
    if not targets or len(results) != int(match.get("maps_played") or 0):
        return ()
    if len({result.map_name for result in results}) != len(results):
        return ()
    bo_type = int(match["bo_type"])
    if bo_type == 1 and len(results) != 1:
        return ()
    if bo_type == 3 and len(results) not in {2, 3}:
        return ()
    expected = targets if len(results) == len(targets) else targets[:2]
    if {result.map_name for result in results} != {
        target.map_name for target in expected
    }:
        return ()
    result_by_name = {result.map_name: result for result in results}
    return tuple((target, result_by_name[target.map_name]) for target in expected)


class _CausalMapStore:
    def __init__(self) -> None:
        self.states: dict[int, dict[str, _MapState]] = defaultdict(dict)

    def _state(self, team_id: int, map_name: str) -> _MapState | None:
        return self.states.get(team_id, {}).get(map_name)

    @staticmethod
    def _side_features(state: _MapState | None, at: datetime) -> dict[str, float]:
        if state is None:
            state = _MapState()
        outcomes = list(state.outcomes)
        result = {
            "target_map_elo": state.elo,
            "target_map_matches": float(state.matches),
            "target_map_log_matches": math.log1p(state.matches),
            "target_map_win_rate": _smoothed_rate(state.wins, state.matches),
            "target_map_round_share": _smoothed_rate(
                state.round_share_sum, state.round_known
            ),
            "target_map_round_known": float(state.round_known),
            "target_map_days_since": _age_days(at, state.last_at),
        }
        for window in _MAP_WINDOWS:
            recent = [item for item in outcomes if _age_days(at, item.at) < window]
            rounds = [
                item.round_share for item in recent if item.round_share is not None
            ]
            result.update(
                {
                    f"target_map_matches_{window}d": float(len(recent)),
                    f"target_map_win_rate_{window}d": _smoothed_rate(
                        sum(item.win for item in recent), len(recent)
                    ),
                    f"target_map_round_share_{window}d": _smoothed_rate(
                        sum(rounds), len(rounds)
                    ),
                    f"target_map_round_known_{window}d": float(len(rounds)),
                    f"target_map_round_share_std_{window}d": _std(rounds),
                }
            )
        for count in _MAP_LAST_COUNTS:
            recent = outcomes[-count:]
            rounds = [
                item.round_share for item in recent if item.round_share is not None
            ]
            result.update(
                {
                    f"target_map_last_{count}_available": float(len(recent)),
                    f"target_map_win_rate_last_{count}": _smoothed_rate(
                        sum(item.win for item in recent), len(recent)
                    ),
                    f"target_map_round_share_last_{count}": _smoothed_rate(
                        sum(rounds), len(rounds)
                    ),
                    f"target_map_round_known_last_{count}": float(len(rounds)),
                }
            )
        for half_life in _MAP_HALF_LIVES:
            weighted = [
                (0.5 ** (_age_days(at, item.at) / half_life), item)
                for item in outcomes
                if _age_days(at, item.at) < 6 * half_life
            ]
            weight = sum(value for value, _ in weighted)
            round_weight = sum(
                value for value, item in weighted if item.round_share is not None
            )
            result.update(
                {
                    f"target_map_ewm_effective_games_{half_life}d": weight,
                    f"target_map_ewm_win_rate_{half_life}d": (
                        sum(value * item.win for value, item in weighted) / weight
                        if weight
                        else 0.5
                    ),
                    f"target_map_ewm_round_share_{half_life}d": (
                        sum(
                            value * float(item.round_share)
                            for value, item in weighted
                            if item.round_share is not None
                        )
                        / round_weight
                        if round_weight
                        else 0.5
                    ),
                }
            )
        style = round_style_features(
            [
                RoundStyleOutcome(item.at, item.round_stats)
                for item in outcomes
                if item.round_stats is not None
            ],
            at,
        )
        result.update({f"target_map_{key}": value for key, value in style.items()})
        return result

    def features(
        self, team1_id: int, team2_id: int, map_name: str, at: datetime
    ) -> dict[str, float]:
        left = self._side_features(self._state(team1_id, map_name), at)
        right = self._side_features(self._state(team2_id, map_name), at)
        result: dict[str, float] = {}
        for key in left:
            result[f"team1_{key}"] = left[key]
            result[f"team2_{key}"] = right[key]
            result[f"diff_{key}"] = left[key] - right[key]
        return result

    def update(
        self,
        team1_id: int,
        team2_id: int,
        results: tuple[_PlayedMap, ...],
        at: datetime,
        tier_weight: float,
    ) -> None:
        for game in results:
            if {game.winner_team_id, game.loser_team_id} != {team1_id, team2_id}:
                continue
            winner = self.states[game.winner_team_id].setdefault(
                game.map_name, _MapState()
            )
            loser = self.states[game.loser_team_id].setdefault(
                game.map_name, _MapState()
            )
            expected = 1.0 / (1.0 + 10.0 ** ((loser.elo - winner.elo) / 400.0))
            delta = 24.0 * (0.75 + 0.5 * tier_weight) * (1.0 - expected)
            total = (
                game.winner_score + game.loser_score
                if game.winner_score is not None and game.loser_score is not None
                else 0
            )
            winner_share = game.winner_score / total if total else None
            loser_share = game.loser_score / total if total else None
            winner.add(
                at,
                1.0,
                winner_share,
                delta,
                round_stats=game.winner_round_stats,
            )
            loser.add(
                at,
                0.0,
                loser_share,
                -delta,
                round_stats=game.loser_round_stats,
            )


def _map_counter_feature_names() -> list[str]:
    dummy = _CausalMapStore().features(1, 2, "mirage", datetime(2020, 1, 1, tzinfo=UTC))
    return list(dummy)


def build_map_feature_table(
    matches_csv: str | Path,
    series_features_csv: str | Path,
    output_csv: str | Path,
    *,
    series_feature_set: str = "core-veto",
) -> dict[str, object]:
    """Expand causal series features into strict after-veto played-map rows."""
    store = _CausalMapStore()
    pending_updates: list[
        tuple[datetime, int, int, int, tuple[_PlayedMap, ...], float]
    ] = []
    skipped: dict[str, int] = defaultdict(int)
    written = 0

    matches_path = Path(matches_csv)
    series_path = Path(series_features_csv)
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    with (
        matches_path.open(encoding="utf-8", newline="") as matches_source,
        series_path.open(encoding="utf-8", newline="") as features_source,
        temporary.open("w", encoding="utf-8", newline="") as target,
    ):
        match_reader = csv.DictReader(matches_source)
        feature_reader = csv.DictReader(features_source)
        if not feature_reader.fieldnames:
            raise ValueError("series feature table has no header")
        header = SimpleNamespace(columns=feature_reader.fieldnames)
        series_columns = _select_feature_columns(
            header, _SERIES_EXCLUDED, series_feature_set
        )
        metadata = [
            "map_row_id",
            "match_id",
            "start_at",
            "known_at",
            "team1_name",
            "team2_name",
            "team1_win",
            "team1_map_score",
            "team2_map_score",
            "map_scores_known",
            "sample_weight",
        ]
        target_columns = [
            "target_map_name",
            "target_map_slot",
            "target_map_role",
            "team1_target_map_pick",
            "team2_target_map_pick",
            "target_map_decider",
        ]
        fieldnames = metadata + series_columns + target_columns
        fieldnames += _map_counter_feature_names()
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("map feature table contains duplicate columns")
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()

        feature_rows = iter(feature_reader)
        match_count = 0
        for index, match in enumerate(match_reader, start=1):
            match_count = index
            try:
                series = next(feature_rows)
            except StopIteration as error:
                raise ValueError("series feature table ended before matches") from error
            if match["match_id"] != series["match_id"]:
                raise ValueError(
                    "matches and series features are not aligned: "
                    f"{match['match_id']} != {series['match_id']}"
                )
            at = _parse_time(match["start_at"])
            while pending_updates and pending_updates[0][0] <= at:
                end_at, _, team1_id, team2_id, results, tier_weight = heapq.heappop(
                    pending_updates
                )
                store.update(
                    team1_id, team2_id, results, end_at, tier_weight=tier_weight
                )

            team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
            target_rows = _eligible_target_rows(match)
            if not strict_veto_complete(match):
                skipped["veto_not_strict"] += 1
            elif int(match["bo_type"]) not in {1, 3}:
                skipped["unsupported_veto_format"] += 1
            elif not target_rows:
                skipped["played_maps_do_not_match_veto"] += 1
            for target_map, result in target_rows:
                scores_known = int(
                    result.winner_score is not None and result.loser_score is not None
                )
                team1_score: int | str = ""
                team2_score: int | str = ""
                if scores_known:
                    team1_score = (
                        result.winner_score
                        if result.winner_team_id == team1_id
                        else result.loser_score
                    )
                    team2_score = (
                        result.winner_score
                        if result.winner_team_id == team2_id
                        else result.loser_score
                    )
                row: dict[str, object] = {
                    "map_row_id": f"{match['match_id']}:{target_map.veto_slot}",
                    "match_id": int(match["match_id"]),
                    "start_at": series["start_at"],
                    "known_at": series["known_at"],
                    "team1_name": series["team1_name"],
                    "team2_name": series["team2_name"],
                    "team1_win": int(result.winner_team_id == team1_id),
                    "team1_map_score": team1_score,
                    "team2_map_score": team2_score,
                    "map_scores_known": scores_known,
                    "sample_weight": series["sample_weight"],
                }
                row.update({column: series[column] for column in series_columns})
                row.update(
                    {
                        "target_map_name": target_map.map_name,
                        "target_map_slot": str(target_map.veto_slot),
                        "target_map_role": target_map.role,
                        "team1_target_map_pick": target_map.picked_by_team1,
                        "team2_target_map_pick": target_map.picked_by_team2,
                        "target_map_decider": int(target_map.role == "decider"),
                    }
                )
                row.update(store.features(team1_id, team2_id, target_map.map_name, at))
                writer.writerow(row)
                written += 1

            results = _played_maps(match)
            if len({result.map_name for result in results}) != len(results):
                results = ()
            known_at_raw = series.get("known_at") or ""
            if results and known_at_raw:
                known_at = _parse_time(known_at_raw)
                if known_at > at:
                    tier_weight = _TIER_WEIGHTS.get(
                        match["tournament_tier"], _TIER_WEIGHTS["unknown"]
                    )
                    heapq.heappush(
                        pending_updates,
                        (
                            known_at,
                            index,
                            team1_id,
                            team2_id,
                            results,
                            tier_weight,
                        ),
                    )
            if index % 10_000 == 0:
                print(
                    f"built map rows from {index} series: rows={written}",
                    file=sys.stderr,
                    flush=True,
                )
        try:
            extra = next(feature_rows)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError("series feature table has more rows than matches")

    temporary.replace(output)
    return {
        "output_csv": str(output),
        "source_matches": match_count,
        "rows": written,
        "features": len(fieldnames),
        "series_feature_set": series_feature_set,
        "skipped": dict(skipped),
    }


def _map_feature_columns(frame: Any) -> list[str]:
    return [column for column in frame.columns if column not in _MAP_EXCLUDED]


def _map_categorical_columns(feature_columns: list[str]) -> list[str]:
    result = _categorical_feature_columns(feature_columns)
    for column in ("target_map_name", "target_map_slot", "target_map_role"):
        if column in feature_columns:
            result.append(column)
    return result


def _slice_metrics(predictions: Any) -> dict[str, object]:
    probability = predictions.team1_win_probability

    def metrics(mask: Any) -> dict[str, float] | None:
        if int(mask.sum()) == 0:
            return None
        return _binary_metrics(predictions.loc[mask, "team1_win"], probability[mask])

    slices: dict[str, object] = {
        "bo1": metrics(predictions.bo_type.astype(str) == "1"),
        "bo3": metrics(predictions.bo_type.astype(str) == "3"),
        "bo3_map1": metrics(
            (predictions.bo_type.astype(str) == "3")
            & (predictions.target_map_slot.astype(str) == "1")
        ),
        "bo3_map2": metrics(
            (predictions.bo_type.astype(str) == "3")
            & (predictions.target_map_slot.astype(str) == "2")
        ),
        "bo3_decider": metrics(
            (predictions.bo_type.astype(str) == "3")
            & (predictions.target_map_slot.astype(str) == "3")
        ),
        "tier_s_a": metrics(predictions.tournament_tier.isin(["s", "a"])),
        "tier_s": metrics(predictions.tournament_tier.eq("s")),
        "tier_a": metrics(predictions.tournament_tier.eq("a")),
        "tier_b": metrics(predictions.tournament_tier.eq("b")),
        "tier_c_d": metrics(predictions.tournament_tier.isin(["c", "d"])),
        "lan": metrics(predictions.event_type.str.casefold() == "lan"),
        "online": metrics(predictions.event_type.str.casefold() == "online"),
        "team1_pick": metrics(predictions.team1_target_map_pick == 1),
        "team2_pick": metrics(predictions.team2_target_map_pick == 1),
        "all_deciders": metrics(predictions.target_map_decider == 1),
        "pick_maps": metrics(
            predictions.team1_target_map_pick.eq(1)
            | predictions.team2_target_map_pick.eq(1)
        ),
    }
    by_map: dict[str, dict[str, float]] = {}
    for map_name, group in predictions.groupby("target_map_name"):
        if len(group) >= 50:
            by_map[str(map_name)] = _binary_metrics(
                group.team1_win, group.team1_win_probability
            )
    slices["by_map"] = by_map
    return slices


def _cohort_map_row_ids(metadata_jsonl: str | Path) -> set[str]:
    path = Path(metadata_jsonl).resolve()
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
                map_row_id = str(row["map_row_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid cohort metadata at {path}:{line_number}"
                ) from error
            if map_row_id in result:
                raise ValueError(f"duplicate cohort map_row_id: {map_row_id}")
            result.add(map_row_id)
    if not result:
        raise ValueError("cohort metadata is empty")
    return result


def _merge_argus_embeddings(frame: Any, embeddings_csv: str | Path, pd: Any):
    embeddings = pd.read_csv(embeddings_csv, low_memory=False)
    if (
        "map_row_id" not in embeddings
        or embeddings.map_row_id.astype(str).duplicated().any()
    ):
        raise ValueError("Argus embeddings require unique map_row_id values")
    feature_columns = [
        column for column in embeddings.columns if column != "map_row_id"
    ]
    if not feature_columns or any(
        not (
            column == "argus_oof_available"
            or column.startswith(
                (
                    "team1_argus_oof_",
                    "team2_argus_oof_",
                    "diff_argus_oof_",
                    "team1_argus_aux_",
                    "team2_argus_aux_",
                    "diff_argus_aux_",
                )
            )
        )
        for column in feature_columns
    ):
        raise ValueError("Argus embedding table has an unexpected feature schema")
    frame = frame.copy()
    frame["map_row_id"] = frame.map_row_id.astype(str)
    embeddings["map_row_id"] = embeddings.map_row_id.astype(str)
    merged = frame.merge(embeddings, on="map_row_id", how="left", validate="one_to_one")
    if "argus_oof_available" not in merged:
        merged["argus_oof_available"] = 1.0
        feature_columns.append("argus_oof_available")
    merged["argus_oof_available"] = merged.argus_oof_available.fillna(0.0)
    matched = int((merged.argus_oof_available > 0).sum())
    if matched == 0:
        raise ValueError("Argus embeddings do not match the map feature table")
    return merged, feature_columns, matched


def walk_forward_map_catboost_backtest(
    features_csv: str | Path,
    output_dir: str | Path,
    *,
    test_from: str = "2026-01-01",
    test_until: str | None = None,
    validation_days: int = 90,
    iterations: int = 900,
    cohort_metadata_jsonl: str | Path | None = None,
    argus_embeddings_csv: str | Path | None = None,
    embedding_feature_mode: str = "combined",
    argus_feature_kind: str = "all",
    tier_weight_profile: str = "dataset",
) -> dict[str, object]:
    """Monthly point-in-time CatBoost backtest for individual map winners."""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    frame = pd.read_csv(features_csv, low_memory=False)
    if tier_weight_profile not in {"dataset", *TIER_WEIGHT_PROFILES}:
        choices = ", ".join(("dataset", *sorted(TIER_WEIGHT_PROFILES)))
        raise ValueError(f"unknown tier weight profile; choose {choices}")
    if tier_weight_profile != "dataset":
        frame["sample_weight"] = [
            tier_weight(tier, tier_weight_profile) for tier in frame.tournament_tier
        ]
    cohort_rows: int | None = None
    if cohort_metadata_jsonl is not None:
        if frame.map_row_id.astype(str).duplicated().any():
            raise ValueError("map feature table has duplicate map_row_id values")
        cohort_ids = _cohort_map_row_ids(cohort_metadata_jsonl)
        available = set(frame.map_row_id.astype(str))
        missing = cohort_ids - available
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise ValueError(f"cohort map_row_id values are missing: {sample}")
        frame = frame[frame.map_row_id.astype(str).isin(cohort_ids)].copy()
        cohort_rows = len(frame)
    embedding_columns: list[str] = []
    embedding_rows: int | None = None
    if embedding_feature_mode not in {"combined", "only"}:
        raise ValueError("embedding_feature_mode must be combined or only")
    if argus_feature_kind not in {
        "all",
        "raw",
        "raw-diff",
        "auxiliary",
        "auxiliary-diff",
    }:
        raise ValueError("unsupported argus_feature_kind")
    if argus_embeddings_csv is not None:
        frame, embedding_columns, embedding_rows = _merge_argus_embeddings(
            frame, argus_embeddings_csv, pd
        )
    elif embedding_feature_mode == "only":
        raise ValueError("embedding_feature_mode=only requires Argus embeddings")
    if argus_embeddings_csv is None and argus_feature_kind != "all":
        raise ValueError("argus_feature_kind requires Argus embeddings")
    _parse_feature_times(frame, pd)
    test_cut = pd.Timestamp(test_from, tz="UTC")
    if test_cut.day != 1:
        raise ValueError("test_from must be the first day of a month")
    test_end = pd.Timestamp(test_until, tz="UTC") if test_until else None
    if test_end is not None and (test_end.day != 1 or test_end <= test_cut):
        raise ValueError("test_until must be a later first day of a month")
    if validation_days < 1:
        raise ValueError("validation_days must be positive")

    def selected_argus_column(column: str) -> bool:
        if column == "argus_oof_available" or argus_feature_kind == "all":
            return True
        family = "raw" if "_argus_oof_" in column else "auxiliary"
        requested_family = argus_feature_kind.removesuffix("-diff")
        if family != requested_family:
            return False
        return not argus_feature_kind.endswith("-diff") or column.startswith("diff_")

    selected_embedding_columns = [
        column for column in embedding_columns if selected_argus_column(column)
    ]
    if embedding_feature_mode == "only":
        feature_columns = selected_embedding_columns
    else:
        excluded_embeddings = set(embedding_columns) - set(selected_embedding_columns)
        feature_columns = [
            column
            for column in _map_feature_columns(frame)
            if column not in excluded_embeddings
        ]
    categorical = _map_categorical_columns(feature_columns)
    for column in categorical:
        frame[column] = frame[column].fillna("missing").astype(str)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict[str, object]] = []
    fold_predictions: list[Any] = []
    latest_model: Any | None = None
    last_match_at = (
        min(frame.start_at.max(), test_end - pd.Timedelta(seconds=1))
        if test_end is not None
        else frame.start_at.max()
    )
    fold_starts = pd.date_range(test_cut, last_match_at, freq="MS")

    def new_model(model_iterations: int, seed: int) -> Any:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=model_iterations,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=7.0,
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
            [tuning_train, _mirror(tuning_train, feature_columns)], ignore_index=True
        )
        augmented_validation = pd.concat(
            [tuning_validation, _mirror(tuning_validation, feature_columns)],
            ignore_index=True,
        )
        tuning_model = new_model(iterations, 300 + fold_number)
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

        augmented_history = pd.concat(
            [history, _mirror(history, feature_columns)], ignore_index=True
        )
        latest_model = new_model(selected_iterations, 400 + fold_number)
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
                "map_row_id",
                "match_id",
                "start_at",
                "team1_name",
                "team2_name",
                "team1_win",
                "bo_type",
                "tournament_tier",
                "event_type",
                "target_map_name",
                "target_map_slot",
                "target_map_role",
                "team1_target_map_pick",
                "team2_target_map_pick",
                "target_map_decider",
                "diff_target_map_elo",
                "diff_elo",
            ]
        ].copy()
        predictions["team1_win_probability"] = probability
        predictions["training_cutoff"] = fold_start.isoformat()
        fold_predictions.append(predictions)
        print(
            f"map walk-forward {fold_start:%Y-%m}: "
            f"rows={len(fold_test)} accuracy={current_metrics['accuracy']:.4f} "
            f"iterations={selected_iterations}",
            file=sys.stderr,
            flush=True,
        )

    if not fold_predictions or latest_model is None:
        raise ValueError("map walk-forward split produced no folds")
    predictions = pd.concat(fold_predictions, ignore_index=True)
    overall = _binary_metrics(predictions.team1_win, predictions.team1_win_probability)
    map_elo_probability = 1.0 / (
        1.0 + 10.0 ** (-predictions.diff_target_map_elo.to_numpy(float) / 400.0)
    )
    team_elo_probability = 1.0 / (
        1.0 + 10.0 ** (-predictions.diff_elo.to_numpy(float) / 400.0)
    )
    pick_mask = predictions.team1_target_map_pick.eq(
        1
    ) | predictions.team2_target_map_pick.eq(1)
    pick_owner_won = np.where(
        predictions.loc[pick_mask, "team1_target_map_pick"].eq(1),
        predictions.loc[pick_mask, "team1_win"],
        1 - predictions.loc[pick_mask, "team1_win"],
    )
    metrics: dict[str, object] = {
        "protocol": {
            "test_from": test_from,
            "test_until": test_until,
            "retrain": "monthly_full_refit",
            "validation_days": validation_days,
            "future_labels_used": False,
            "prediction_point": "retrospective_assumed_after_veto",
            "feature_count": len(feature_columns),
            "cohort_filter": cohort_metadata_jsonl is not None,
            "cohort_rows": cohort_rows,
            "argus_embeddings": argus_embeddings_csv is not None,
            "argus_embedding_rows": embedding_rows,
            "embedding_feature_mode": embedding_feature_mode,
            "embedding_feature_count": len(embedding_columns),
            "selected_embedding_feature_count": len(selected_embedding_columns),
            "argus_feature_kind": argus_feature_kind,
            "tier_weight_profile": tier_weight_profile,
            "effective_tier_weight_mass": effective_tier_weight_mass(
                history.tournament_tier.tolist(),
                history.sample_weight.astype(float).tolist(),
            ),
        },
        "overall": overall,
        "confidence_slices": _confidence_slices(
            predictions.team1_win, predictions.team1_win_probability
        ),
        "slices": _slice_metrics(predictions),
        "baselines": {
            "constant_0_5": _binary_metrics(
                predictions.team1_win, np.full(len(predictions), 0.5)
            ),
            "target_map_elo": _binary_metrics(
                predictions.team1_win, map_elo_probability
            ),
            "series_team_elo": _binary_metrics(
                predictions.team1_win, team_elo_probability
            ),
        },
        "pick_maps": {
            "rows": int(pick_mask.sum()),
            "pick_owner_win_rate": float(np.mean(pick_owner_won)),
        },
        "folds": fold_metrics,
    }
    latest_model.save_model(output / "map_winner_catboost_walk_forward_latest.cbm")
    predictions.to_csv(output / "map_walk_forward_test_predictions.csv", index=False)
    latest_model.get_feature_importance(prettified=True).to_csv(
        output / "map_feature_importance.csv", index=False
    )
    (output / "map_feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2) + "\n", encoding="utf-8"
    )
    (output / "map_walk_forward_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics
