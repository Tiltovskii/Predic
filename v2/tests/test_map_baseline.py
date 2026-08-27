from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predic_v2.baseline import _mirror
from predic_v2.map_baseline import (
    _CausalMapStore,
    _cohort_map_row_ids,
    _eligible_target_rows,
    _map_categorical_columns,
    _PlayedMap,
    build_map_feature_table,
)


def _veto_actions(team1_id: int = 1, team2_id: int = 2) -> list[dict[str, object]]:
    patterns = (2, 2, 1, 1, 2, 2, 3)
    actors = (team1_id, team2_id, team1_id, team2_id, team1_id, team2_id, None)
    names = ("nuke", "inferno", "de_cache", "mirage", "ancient", "anubis", "vertigo")
    return [
        {
            "order": order,
            "choice_type": choice_type,
            "team_id": actor,
            "map_name": map_name,
        }
        for order, (choice_type, actor, map_name) in enumerate(
            zip(patterns, actors, names), start=1
        )
    ]


def _match(
    match_id: int,
    start_at: datetime,
    *,
    winner_by_map: tuple[int, int] = (1, 2),
) -> dict[str, str]:
    results = [
        {
            "map_name": name,
            "winner_team_id": winner,
            "loser_team_id": 2 if winner == 1 else 1,
            "winner_score": 13,
            "loser_score": 9,
        }
        for name, winner in zip(("cache", "mirage"), winner_by_map)
    ]
    return {
        "match_id": str(match_id),
        "start_at": start_at.isoformat(),
        "end_at": (start_at + timedelta(hours=2)).isoformat(),
        "team1_id": "1",
        "team1_name": "Left",
        "team2_id": "2",
        "team2_name": "Right",
        "bo_type": "3",
        "tournament_tier": "a",
        "maps_played": "2",
        "map_results": json.dumps(results),
        "veto_actions": json.dumps(_veto_actions()),
    }


def _series(match: dict[str, str]) -> dict[str, object]:
    return {
        "match_id": match["match_id"],
        "start_at": match["start_at"],
        "known_at": match["end_at"],
        "veto_known": 1,
        "team1_name": match["team1_name"],
        "team2_name": match["team2_name"],
        "team1_win": 1,
        "score_label": "2-0",
        "maps_played": 2,
        "team1_rounds": 26,
        "team2_rounds": 18,
        "rounds_known": 1,
        "round_share_target": 26 / 44,
        "sample_weight": 1.0,
        "team1_id": match["team1_id"],
        "team2_id": match["team2_id"],
        "bo_type": match["bo_type"],
        "game_version": "2",
        "tournament_tier": "a",
        "event_type": "lan",
        "bracket_type": "upper",
        "diff_elo": 25.0,
        "team1_veto_pick_1_map": "cache",
        "team2_veto_pick_1_map": "mirage",
        "veto_decider_map": "vertigo",
        "diff_counter_veto_selected_map_matchup_mean_180d": 0.1,
    }


class MapBaselineTest(unittest.TestCase):
    def test_cohort_metadata_requires_unique_map_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl"
            path.write_text(
                '{"map_row_id":"1:1"}\n{"map_row_id":"2:1"}\n',
                encoding="utf-8",
            )
            self.assertEqual({"1:1", "2:1"}, _cohort_map_row_ids(path))
            path.write_text(
                '{"map_row_id":"1:1"}\n{"map_row_id":"1:1"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate cohort"):
                _cohort_map_row_ids(path)

    def test_target_rows_use_veto_identity_and_pick_owner(self) -> None:
        match = _match(1, datetime(2025, 1, 1, tzinfo=UTC))

        rows = _eligible_target_rows(match)

        self.assertEqual(["cache", "mirage"], [target.map_name for target, _ in rows])
        self.assertEqual([1, 0], [target.picked_by_team1 for target, _ in rows])
        self.assertEqual([0, 1], [target.picked_by_team2 for target, _ in rows])
        self.assertEqual(["pick", "pick"], [target.role for target, _ in rows])

    def test_target_rows_reject_played_map_outside_veto(self) -> None:
        match = _match(1, datetime(2025, 1, 1, tzinfo=UTC))
        results = json.loads(match["map_results"])
        results[1]["map_name"] = "dust2"
        match["map_results"] = json.dumps(results)

        self.assertEqual((), _eligible_target_rows(match))

    def test_map_store_is_causal_and_side_symmetric(self) -> None:
        store = _CausalMapStore()
        at = datetime(2025, 1, 1, tzinfo=UTC)
        before = store.features(1, 2, "cache", at)
        store.update(
            1,
            2,
            (_PlayedMap("cache", 1, 2, 13, 9),),
            at,
            tier_weight=1.0,
        )
        after = store.features(1, 2, "cache", at + timedelta(days=1))
        reverse = store.features(2, 1, "cache", at + timedelta(days=1))

        self.assertEqual(0.0, before["team1_target_map_matches"])
        self.assertEqual(1.0, after["team1_target_map_matches"])
        self.assertGreater(after["diff_target_map_elo"], 0.0)
        for key, value in after.items():
            if key.startswith("team1_"):
                self.assertAlmostEqual(
                    value, reverse[key.replace("team1_", "team2_", 1)]
                )
            elif key.startswith("team2_"):
                self.assertAlmostEqual(
                    value, reverse[key.replace("team2_", "team1_", 1)]
                )
            elif key.startswith("diff_"):
                self.assertAlmostEqual(value, -reverse[key])

    def test_builder_does_not_expose_current_series_maps_to_each_other(self) -> None:
        first = _match(1, datetime(2025, 1, 1, tzinfo=UTC))
        second = _match(2, datetime(2025, 1, 2, tzinfo=UTC))
        matches = [first, second]
        series = [_series(first), _series(second)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches_path = root / "matches.csv"
            series_path = root / "series.csv"
            output_path = root / "maps.csv"
            with matches_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(matches[0]))
                writer.writeheader()
                writer.writerows(matches)
            with series_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(series[0]))
                writer.writeheader()
                writer.writerows(series)

            result = build_map_feature_table(matches_path, series_path, output_path)
            with output_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(4, result["rows"])
        self.assertEqual(
            ["1:1", "1:2", "2:1", "2:2"], [row["map_row_id"] for row in rows]
        )
        self.assertEqual("0.0", rows[0]["team1_target_map_matches"])
        self.assertEqual("0.0", rows[1]["team1_target_map_matches"])
        self.assertEqual("1.0", rows[2]["team1_target_map_matches"])
        self.assertEqual("1.0", rows[3]["team1_target_map_matches"])

    def test_map_categories_and_mirror_are_consistent(self) -> None:
        import pandas as pd

        columns = [
            "team1_id",
            "target_map_name",
            "target_map_slot",
            "target_map_role",
            "diff_target_map_elo",
        ]
        self.assertEqual(columns[:4], _map_categorical_columns(columns))
        frame = pd.DataFrame(
            {
                "team1_win": [1],
                "team1_target_map_pick": [1],
                "team2_target_map_pick": [0],
                "diff_target_map_elo": [20.0],
                "target_map_name": ["cache"],
            }
        )
        mirrored = _mirror(
            frame,
            [
                "team1_target_map_pick",
                "team2_target_map_pick",
                "diff_target_map_elo",
                "target_map_name",
            ],
        )
        self.assertEqual(0, mirrored.loc[0, "team1_win"])
        self.assertEqual(0, mirrored.loc[0, "team1_target_map_pick"])
        self.assertEqual(1, mirrored.loc[0, "team2_target_map_pick"])
        self.assertEqual(-20.0, mirrored.loc[0, "diff_target_map_elo"])
        self.assertEqual("cache", mirrored.loc[0, "target_map_name"])


if __name__ == "__main__":
    unittest.main()
