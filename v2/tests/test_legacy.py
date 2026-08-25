from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from predic_v2.audit import audit
from predic_v2.db import connect, initialize
from predic_v2.legacy import import_legacy_csv


class LegacyImportTest(unittest.TestCase):
    def test_mirrored_rows_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "legacy.csv"
            fields = [
                "date",
                "team_1",
                "team_2",
                "score_1",
                "score_2",
                "players1",
                "players2",
                "Map",
                "team_rank_1",
                "team_rank_2",
                "right_pick",
                "mean_rating1",
                "mean_rating2",
                "score",
            ]
            row = {
                "date": "2018-01-02",
                "team_1": "Alpha",
                "team_2": "Beta",
                "score_1": "16",
                "score_2": "10",
                "players1": "['a1', 'a2', 'a3', 'a4', 'a5']",
                "players2": "['b1', 'b2', 'b3', 'b4', 'b5']",
                "Map": "Inferno",
                "team_rank_1": "10",
                "team_rank_2": "20",
                "right_pick": "1",
                "mean_rating1": "1.1",
                "mean_rating2": "0.9",
                "score": "0.7",
            }
            mirrored = dict(row)
            mirrored.update(
                team_1="Beta",
                team_2="Alpha",
                score_1="10",
                score_2="16",
                players1=row["players2"],
                players2=row["players1"],
                team_rank_1="20",
                team_rank_2="10",
                right_pick="0",
                mean_rating1="0.9",
                mean_rating2="1.1",
                score="0.3",
            )
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([row, mirrored])

            connection = connect(root / "test.sqlite3")
            self.addCleanup(connection.close)
            initialize(connection)
            result = import_legacy_csv(connection, csv_path, date(2018, 1, 1))

            self.assertEqual(1, result["maps"])
            self.assertEqual(10, result["lineup_members"])
            self.assertEqual(2, result["rankings"])
            source = connection.execute(
                "SELECT point_in_time_eligible FROM source_snapshot"
            ).fetchone()
            self.assertEqual(0, source["point_in_time_eligible"])
            legacy_known_at = connection.execute(
                "SELECT known_at FROM map_game"
            ).fetchone()
            self.assertIsNone(legacy_known_at["known_at"])
            self.assertTrue(audit(connection)["ok"])


if __name__ == "__main__":
    unittest.main()
