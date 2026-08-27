from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from predic_v2.baseline import (
    ExternalRankingIndex,
    _labels_known_before,
    _load_lineups,
    _parse_feature_times,
    _select_feature_columns,
    build_point_in_time_features,
)


class ExternalRankingIndexTest(unittest.TestCase):
    def test_training_history_requires_label_known_before_cutoff(self) -> None:
        import pandas as pd

        cutoff = pd.Timestamp("2026-02-01", tz="UTC")
        frame = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "known_at": [
                    cutoff - pd.Timedelta(seconds=1),
                    cutoff,
                    pd.NaT,
                ],
            }
        )
        history = _labels_known_before(frame, cutoff)
        self.assertEqual([1], history.match_id.tolist())

    def test_mixed_iso_precision_keeps_valid_known_at(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "start_at": [
                    "2024-01-01T12:00:00.000+00:00",
                    "2024-01-02T12:00:00+00:00",
                    "2024-01-03T12:00:00Z",
                ],
                "known_at": [
                    "2024-01-01T14:00:00.000+00:00",
                    "2024-01-02T14:00:00+00:00",
                    "",
                ],
            }
        )
        _parse_feature_times(frame, pd)
        self.assertEqual(0, frame.start_at.isna().sum())
        self.assertEqual(1, frame.known_at.isna().sum())

    def test_feature_sets_keep_counter_ablation_reproducible(self) -> None:
        frame = type(
            "Frame",
            (),
            {
                "columns": [
                    "base",
                    "team1_counter_win_rate_30d",
                    "diff_counter_win_rate_30d",
                    "team1_counter_matches_30d",
                    "counter_map_pool_overlap_180d",
                    "target",
                ]
            },
        )()
        self.assertEqual(["base"], _select_feature_columns(frame, {"target"}, "base"))
        self.assertEqual(
            [
                "base",
                "diff_counter_win_rate_30d",
                "team1_counter_matches_30d",
                "counter_map_pool_overlap_180d",
            ],
            _select_feature_columns(frame, {"target"}, "core"),
        )

    def test_uses_strictly_previous_snapshot_and_roster_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.csv"
            path.write_text(
                "ranking_system,region,published_at,rank,points,team_name,roster,roster_signature,source_commit,source_path\n"
                'valve_global,global,2025-01-01,1,2000,Old Org,"[""a"", ""b"", ""c"", ""d"", ""e""]",a,b,c\n',
                encoding="utf-8",
            )
            index = ExternalRankingIndex(path)
            same_day = index.lookup(
                "valve_global",
                __import__("datetime").datetime(
                    2025, 1, 1, 12, tzinfo=__import__("datetime").timezone.utc
                ),
                "New Org",
                ("a", "b", "c", "d", "e"),
            )
            self.assertEqual(1.0, same_day["missing"])
            later = index.lookup(
                "valve_global",
                __import__("datetime").datetime(
                    2025, 1, 2, tzinfo=__import__("datetime").timezone.utc
                ),
                "New Org",
                ("a", "b", "c", "d", "e"),
            )
            self.assertEqual(0.0, later["missing"])
            self.assertEqual(5.0, later["roster_overlap"])

    def test_features_are_frozen_before_current_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory) / "matches.csv"
            features = Path(directory) / "features.csv"
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
                "prize",
                "maps_played",
                "team1_map_wins",
                "team2_map_wins",
                "team1_rounds",
                "team2_rounds",
                "rounds_known",
                "score_label",
            ]
            base = {
                "team1_id": "1",
                "team1_name": "A",
                "team1_roster": json.dumps(["11:a", "12:b", "13:c", "14:d", "15:e"]),
                "team2_id": "2",
                "team2_name": "B",
                "team2_roster": json.dumps(["21:f", "22:g", "23:h", "24:i", "25:j"]),
                "winner_team_id": "1",
                "team1_win": "1",
                "bo_type": "3",
                "game_version": "2",
                "tournament_id": "1",
                "tournament_name": "Test",
                "tournament_tier": "s",
                "tournament_tier_rank": "1",
                "event_type": "lan",
                "prize": "1",
                "maps_played": "2",
                "team1_map_wins": "2",
                "team2_map_wins": "0",
                "team1_rounds": "26",
                "team2_rounds": "10",
                "rounds_known": "1",
                "score_label": "2-0",
            }
            with matches.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        **base,
                        "match_id": "1",
                        "start_at": "2024-01-01T12:00:00Z",
                        "end_at": "2024-01-01T14:00:00Z",
                    }
                )
                writer.writerow(
                    {
                        **base,
                        "match_id": "2",
                        "start_at": "2024-01-02T12:00:00Z",
                        "end_at": "2024-01-02T14:00:00Z",
                    }
                )
            build_point_in_time_features(matches, features)
            with features.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("0.0", rows[0]["team1_matches"])
            self.assertEqual("1.0", rows[1]["team1_matches"])
            self.assertEqual("1500.0", rows[0]["team1_elo"])
            self.assertGreater(float(rows[1]["team1_elo"]), 1500.0)
            self.assertAlmostEqual(26 / 36, float(rows[0]["round_share_target"]))
            self.assertEqual("5.0", rows[0]["team1_roster_size"])
            self.assertEqual("1.0", rows[1]["team1_round_share_matches_30d"])
            self.assertAlmostEqual(
                (26 / 36 + 2) / 5,
                float(rows[1]["team1_round_share_30d"]),
            )
            self.assertAlmostEqual(
                1 - float(rows[1]["team1_h2h_win_rate_365d"]),
                float(rows[1]["team2_h2h_win_rate_365d"]),
            )

    def test_overlapping_match_does_not_see_unfinished_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory) / "matches.csv"
            features = Path(directory) / "features.csv"
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
                "prize",
                "maps_played",
                "team1_map_wins",
                "team2_map_wins",
                "team1_rounds",
                "team2_rounds",
                "rounds_known",
                "map_results",
                "score_label",
            ]
            base = {
                "team1_id": "1",
                "team1_name": "A",
                "team1_roster": "[]",
                "team2_id": "2",
                "team2_name": "B",
                "team2_roster": "[]",
                "winner_team_id": "1",
                "team1_win": "1",
                "bo_type": "3",
                "game_version": "2",
                "tournament_id": "1",
                "tournament_name": "Test",
                "tournament_tier": "s",
                "tournament_tier_rank": "1",
                "event_type": "lan",
                "prize": "1",
                "maps_played": "2",
                "team1_map_wins": "2",
                "team2_map_wins": "0",
                "team1_rounds": "26",
                "team2_rounds": "10",
                "rounds_known": "1",
                "map_results": "[]",
                "score_label": "2-0",
            }
            with matches.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        **base,
                        "match_id": "1",
                        "start_at": "2024-01-01T12:00:00Z",
                        "end_at": "2024-01-02T12:00:00Z",
                    }
                )
                writer.writerow(
                    {
                        **base,
                        "match_id": "2",
                        "start_at": "2024-01-01T20:00:00Z",
                        "end_at": "2024-01-01T22:00:00Z",
                    }
                )
            build_point_in_time_features(matches, features)
            with features.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("0.0", rows[0]["team1_matches"])
            self.assertEqual("0.0", rows[1]["team1_matches"])
            self.assertEqual("2024-01-02T12:00:00+00:00", rows[0]["known_at"])

    def test_non_positive_duration_never_updates_causal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory) / "matches.csv"
            features = Path(directory) / "features.csv"
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
                "prize",
                "maps_played",
                "team1_map_wins",
                "team2_map_wins",
                "team1_rounds",
                "team2_rounds",
                "rounds_known",
                "map_results",
                "score_label",
            ]
            base = {
                "team1_id": "1",
                "team1_name": "A",
                "team1_roster": "[]",
                "team2_id": "2",
                "team2_name": "B",
                "team2_roster": "[]",
                "winner_team_id": "1",
                "team1_win": "1",
                "bo_type": "3",
                "game_version": "2",
                "tournament_id": "1",
                "tournament_name": "Test",
                "tournament_tier": "s",
                "tournament_tier_rank": "1",
                "event_type": "lan",
                "prize": "1",
                "maps_played": "2",
                "team1_map_wins": "2",
                "team2_map_wins": "0",
                "team1_rounds": "26",
                "team2_rounds": "10",
                "rounds_known": "1",
                "map_results": "[]",
                "score_label": "2-0",
            }
            with matches.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        **base,
                        "match_id": "1",
                        "start_at": "2024-01-01T12:00:00Z",
                        "end_at": "2024-01-01T11:00:00Z",
                    }
                )
                writer.writerow(
                    {
                        **base,
                        "match_id": "2",
                        "start_at": "2024-01-03T12:00:00Z",
                        "end_at": "2024-01-03T14:00:00Z",
                    }
                )
            build_point_in_time_features(matches, features)
            with features.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("", rows[0]["known_at"])
            self.assertEqual("0.0", rows[1]["team1_matches"])

    def test_historical_player_is_not_removed_by_current_coach_status(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE bo3_game_index (
                stream TEXT, game_id INTEGER, match_id INTEGER
            );
            CREATE TABLE bo3_player_map_index (
                stream TEXT, game_id INTEGER, team_id INTEGER,
                steam_profile_id INTEGER, nickname TEXT,
                rounds_participated INTEGER, current_is_coach INTEGER
            );
            INSERT INTO bo3_game_index VALUES ('s', 10, 100);
            INSERT INTO bo3_player_map_index VALUES
                ('s', 10, 1, 1, 'a', 20, 1),
                ('s', 10, 1, 2, 'b', 20, 0),
                ('s', 10, 1, 3, 'c', 20, 0),
                ('s', 10, 1, 4, 'd', 20, 0),
                ('s', 10, 1, 5, 'e', 20, 0);
            """
        )
        lineups = _load_lineups(connection, "s")
        self.assertEqual(5, len(lineups[(100, 1)]))
        self.assertIn("1:a", lineups[(100, 1)])


if __name__ == "__main__":
    unittest.main()
