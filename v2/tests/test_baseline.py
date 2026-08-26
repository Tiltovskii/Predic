from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from predic_v2.baseline import ExternalRankingIndex, build_point_in_time_features


class ExternalRankingIndexTest(unittest.TestCase):
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
                    {**base, "match_id": "1", "start_at": "2024-01-01T12:00:00Z"}
                )
                writer.writerow(
                    {**base, "match_id": "2", "start_at": "2024-01-02T12:00:00Z"}
                )
            build_point_in_time_features(matches, features)
            with features.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("0.0", rows[0]["team1_matches"])
            self.assertEqual("1.0", rows[1]["team1_matches"])
            self.assertEqual("1500.0", rows[0]["team1_elo"])
            self.assertGreater(float(rows[1]["team1_elo"]), 1500.0)


if __name__ == "__main__":
    unittest.main()
