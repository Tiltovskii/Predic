from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from predic_v2.db import connect, initialize
from predic_v2.hltv_offline import parse_file, records_to_jsonl
from predic_v2.materialize import (
    BatchLimitError,
    materialize_raw_stream,
    materialize_records,
)
from predic_v2.raw_jsonl import import_jsonl


FIXTURES = Path(__file__).parent / "fixtures"
OBSERVED_AT = "2026-08-25T12:34:56+00:00"
KNOWN_AT = "2024-01-02T10:00:00Z"


class HltvMaterializerTest(unittest.TestCase):
    def _database(self, root: Path):
        connection = connect(root / "test.sqlite3")
        initialize(connection)
        self.addCleanup(connection.close)
        return connection

    def _records(self) -> list[dict[str, object]]:
        match_records = parse_file(
            FIXTURES / "hltv_match.html", observed_at=OBSERVED_AT
        )
        stats_records = parse_file(
            FIXTURES / "hltv_map_stats.html", observed_at=OBSERVED_AT
        )
        # The materializer must preserve known_at exactly, rather than use the
        # capture time as a surrogate.  Only this one source field is asserted
        # to have been historically available in this synthetic fixture.
        stats_records[0]["known_at"] = KNOWN_AT
        return [*match_records, *stats_records]

    def _ingest(
        self,
        connection,
        root: Path,
        records: list[dict[str, object]],
        *,
        stream: str = "fixture-hltv",
    ) -> str:
        path = root / f"{stream}.jsonl"
        path.write_text(records_to_jsonl(records), encoding="utf-8")
        import_jsonl(
            connection,
            path,
            source="authorized-hltv-capture",
            stream=stream,
        )
        return str(
            connection.execute(
                """
                SELECT DISTINCT source_snapshot_id
                FROM raw_ingest_record
                WHERE source = ? AND stream = ?
                """,
                ("authorized-hltv-capture", stream),
            ).fetchone()[0]
        )

    def test_typed_records_materialize_with_stable_ids_and_time_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            self._ingest(connection, root, self._records())

            result = materialize_raw_stream(
                connection,
                source="authorized-hltv-capture",
                stream="fixture-hltv",
                max_records=100,
            )

            self.assertEqual(
                {
                    "teams": 2,
                    "players": 2,
                    "series": 1,
                    "maps": 1,
                    "rankings": 1,
                    "player_map_stats": 2,
                },
                result["inserted"],
            )
            self.assertEqual(3, result["quarantined_count"])
            self.assertEqual(
                {"lineup_not_map_scoped", "map_not_finished"},
                {item["reason"] for item in result["quarantined"]},
            )
            self.assertFalse(result["has_more"])

            series = connection.execute(
                "SELECT * FROM series WHERE series_id = 'hltv:series:9001'"
            ).fetchone()
            self.assertIsNotNone(series)
            self.assertEqual("9001", series["source_series_id"])
            self.assertIsNone(series["known_at"])
            self.assertEqual(OBSERVED_AT, series["observed_at"])

            game = connection.execute(
                "SELECT * FROM map_game WHERE map_id = 'hltv:map:501'"
            ).fetchone()
            self.assertEqual("501", game["source_map_id"])
            self.assertEqual("hltv:team:10", game["team_a_id"])
            self.assertEqual("hltv:team:20", game["team_b_id"])
            self.assertEqual("hltv:team:10", game["winner_team_id"])
            self.assertEqual("hltv:team:10", game["picked_by_team_id"])
            self.assertEqual(13, game["score_a"])
            self.assertEqual(8, game["score_b"])
            self.assertIsNone(game["known_at"])
            self.assertEqual("2024-01-02T12:00:00Z", game["started_at"])

            stats = connection.execute(
                """
                SELECT pms.*, p.canonical_nickname
                FROM player_map_stats AS pms
                JOIN player AS p ON p.player_id = pms.player_id
                WHERE pms.map_id = 'hltv:map:501'
                ORDER BY pms.player_id
                """
            ).fetchall()
            self.assertEqual(2, len(stats))
            self.assertEqual("hltv:player:101", stats[0]["player_id"])
            self.assertEqual("a1", stats[0]["canonical_nickname"])
            self.assertEqual(KNOWN_AT, stats[0]["known_at"])
            self.assertIsNone(stats[1]["known_at"])
            self.assertEqual(2.35, stats[0]["swing"])
            self.assertEqual(1.31, stats[0]["rating"])
            self.assertEqual(
                "501", json.loads(stats[0]["metrics_json"])["map_stats_id"]
            )
            self.assertEqual(
                0,
                connection.execute("SELECT COUNT(*) FROM lineup_member").fetchone()[0],
            )

    def test_replay_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            self._ingest(connection, root, self._records())

            materialize_raw_stream(
                connection,
                source="authorized-hltv-capture",
                stream="fixture-hltv",
                max_records=100,
            )
            replay = materialize_raw_stream(
                connection,
                source="authorized-hltv-capture",
                stream="fixture-hltv",
                max_records=100,
            )

            self.assertEqual(
                {
                    "teams": 0,
                    "players": 0,
                    "series": 0,
                    "maps": 0,
                    "rankings": 0,
                    "player_map_stats": 0,
                },
                replay["inserted"],
            )
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM series").fetchone()[0])
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(
                2, connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0]
            )

    def test_orphan_map_stats_are_quarantined_without_inventing_a_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            stats = parse_file(
                FIXTURES / "hltv_map_stats.html", observed_at=OBSERVED_AT
            )
            self._ingest(connection, root, stats)

            result = materialize_raw_stream(
                connection,
                source="authorized-hltv-capture",
                stream="fixture-hltv",
                max_records=100,
            )

            self.assertEqual(2, result["quarantined_count"])
            self.assertEqual(
                {"missing_map_link"}, {item["reason"] for item in result["quarantined"]}
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(
                0, connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0]
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM player").fetchone()[0])

    def test_invalid_map_team_link_is_quarantined_without_creating_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            records = parse_file(
                FIXTURES / "hltv_match.html", observed_at=OBSERVED_AT
            )
            map_record = next(record for record in records if record["kind"] == "map")
            payload = map_record["payload"]
            self.assertIsInstance(payload, dict)
            payload["team_ids"] = ["10", "999"]
            payload["team_names"] = ["Alpha", "Imposter"]
            snapshot_id = self._ingest(connection, root, records)

            result = materialize_records(
                connection,
                records,
                source_snapshot_id=snapshot_id,
            )

            self.assertIn(
                "map_teams_do_not_match_series",
                {item["reason"] for item in result["quarantined"]},
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM team_core WHERE team_id = 'hltv:team:999'"
                ).fetchone()
            )

    def test_later_map_and_ranking_phases_recheck_existing_series_participants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            records = self._records()
            snapshot_id = self._ingest(connection, root, records)
            series = next(record for record in records if record["kind"] == "series")
            map_record = next(record for record in records if record["kind"] == "map")
            ranking = next(record for record in records if record["kind"] == "ranking")

            materialize_records(
                connection,
                [series],
                source_snapshot_id=snapshot_id,
            )
            self.assertEqual(
                2,
                connection.execute(
                    "SELECT COUNT(*) FROM series_participant WHERE series_id = 'hltv:series:9001'"
                ).fetchone()[0],
            )

            bad_map = json.loads(json.dumps(map_record))
            bad_map["payload"]["team_ids"] = ["10", "999"]
            bad_map["payload"]["team_names"] = ["Alpha", "Imposter"]
            map_result = materialize_records(
                connection,
                [bad_map],
                source_snapshot_id=snapshot_id,
            )

            bad_ranking = json.loads(json.dumps(ranking))
            bad_ranking["payload"]["team_id"] = "999"
            bad_ranking["payload"]["team_name"] = "Imposter"
            ranking_result = materialize_records(
                connection,
                [bad_ranking],
                source_snapshot_id=snapshot_id,
            )

            self.assertEqual(
                {"map_teams_do_not_match_series"},
                {item["reason"] for item in map_result["quarantined"]},
            )
            self.assertEqual(
                {"ranking_team_not_in_series"},
                {item["reason"] for item in ranking_result["quarantined"]},
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM team_core WHERE team_id = 'hltv:team:999'"
                ).fetchone()
            )

    def test_global_map_stats_id_cannot_be_rebound_in_a_later_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            original = self._records()
            first_snapshot = self._ingest(connection, root, original, stream="first")
            first_series = next(record for record in original if record["kind"] == "series")
            first_map = next(record for record in original if record["kind"] == "map")
            materialize_records(
                connection,
                [first_series, first_map],
                source_snapshot_id=first_snapshot,
            )

            conflicting_series = json.loads(json.dumps(first_series))
            conflicting_series["record_id"] = "series-9002"
            conflicting_series["payload"]["match_id"] = "9002"
            conflicting_map = json.loads(json.dumps(first_map))
            conflicting_map["record_id"] = "map-9002-501"
            conflicting_map["payload"]["match_id"] = "9002"
            second_snapshot = self._ingest(
                connection,
                root,
                [conflicting_series, conflicting_map],
                stream="second",
            )

            result = materialize_records(
                connection,
                [conflicting_series, conflicting_map],
                source_snapshot_id=second_snapshot,
            )

            self.assertEqual(
                {"map_identity_conflict"},
                {item["reason"] for item in result["quarantined"]},
            )
            game = connection.execute(
                "SELECT series_id FROM map_game WHERE map_id = 'hltv:map:501'"
            ).fetchone()
            self.assertEqual("hltv:series:9001", game["series_id"])
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])

    def test_player_map_stats_cannot_move_one_player_to_the_other_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            original = self._records()
            first_snapshot = self._ingest(connection, root, original, stream="first")
            materialize_records(
                connection,
                original,
                source_snapshot_id=first_snapshot,
            )

            first_stat = next(
                record for record in original if record["kind"] == "player_map_stats"
            )
            conflicting_stat = json.loads(json.dumps(first_stat))
            conflicting_stat["record_id"] = "player-map-501-101-other-team"
            conflicting_stat["payload"]["team_id"] = "20"
            second_snapshot = self._ingest(
                connection,
                root,
                [conflicting_stat],
                stream="second",
            )

            result = materialize_records(
                connection,
                [conflicting_stat],
                source_snapshot_id=second_snapshot,
            )

            self.assertEqual(
                {"conflicting_player_map_stats_team"},
                {item["reason"] for item in result["quarantined"]},
            )
            stats = connection.execute(
                """
                SELECT team_id FROM player_map_stats
                WHERE map_id = 'hltv:map:501' AND player_id = 'hltv:player:101'
                """
            ).fetchone()
            self.assertEqual("hltv:team:10", stats["team_id"])

    def test_direct_batch_limit_fails_before_any_normalized_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            records = self._records()
            snapshot_id = self._ingest(connection, root, records)

            with self.assertRaises(BatchLimitError):
                materialize_records(
                    connection,
                    records[:2],
                    source_snapshot_id=snapshot_id,
                    max_records=1,
                )

            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM team_core").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM series").fetchone()[0])

    def test_sqlite_failure_rolls_back_the_entire_valid_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            records = self._records()
            snapshot_id = self._ingest(connection, root, records)
            connection.execute(
                """
                CREATE TRIGGER fail_map_materialization
                BEFORE INSERT ON map_game
                BEGIN
                    SELECT RAISE(ABORT, 'test map write failure');
                END
                """
            )

            with self.assertRaises(sqlite3.IntegrityError):
                materialize_records(
                    connection,
                    records,
                    source_snapshot_id=snapshot_id,
                )

            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM team_core").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM series").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])

    def test_raw_window_is_bounded_and_returns_a_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = self._database(root)
            self._ingest(connection, root, self._records())

            result = materialize_raw_stream(
                connection,
                source="authorized-hltv-capture",
                stream="fixture-hltv",
                max_records=1,
            )

            self.assertEqual(1, result["input_records"])
            self.assertTrue(result["has_more"])
            self.assertIsInstance(result["next_raw_record_id"], str)


if __name__ == "__main__":
    unittest.main()
