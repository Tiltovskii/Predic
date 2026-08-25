from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from predic_v2.db import connect, initialize
from predic_v2.raw_jsonl import (
    RecordConflictError,
    SourceChangedError,
    import_jsonl,
)


class RawJsonlImportTest(unittest.TestCase):
    def _database(self, root: Path):
        connection = connect(root / "test.sqlite3")
        initialize(connection)
        self.addCleanup(connection.close)
        return connection

    def test_resume_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.jsonl"
            records = [
                {
                    "record_id": f"m-{number}",
                    "kind": "match",
                    "event_at": f"2018-01-0{number}T00:00:00Z",
                }
                for number in range(1, 4)
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            connection = self._database(root)

            first = import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
                max_records=1,
            )
            second = import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
            )
            third = import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
            )

            self.assertFalse(first["eof"])
            self.assertEqual(1, first["imported"])
            self.assertTrue(second["eof"])
            self.assertEqual(2, second["imported"])
            self.assertEqual(0, third["imported"])
            self.assertEqual(
                3,
                connection.execute(
                    "SELECT COUNT(*) FROM raw_ingest_record"
                ).fetchone()[0],
            )
            self.assertEqual(
                1,
                connection.execute("SELECT COUNT(*) FROM source_snapshot").fetchone()[0],
            )
            checkpoint = connection.execute(
                "SELECT cursor, high_watermark, metadata_json FROM ingestion_checkpoint"
            ).fetchone()
            self.assertEqual(path.stat().st_size, int(checkpoint["cursor"]))
            self.assertEqual("2018-01-03T00:00:00Z", checkpoint["high_watermark"])
            self.assertEqual("complete", json.loads(checkpoint["metadata_json"])["status"])

    def test_changed_file_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.jsonl"
            path.write_text('{"record_id":"one"}\n', encoding="utf-8")
            connection = self._database(root)
            import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
            )
            checkpoint_before = dict(
                connection.execute(
                    "SELECT cursor, metadata_json FROM ingestion_checkpoint"
                ).fetchone()
            )
            path.write_text(
                '{"record_id":"one"}\n{"record_id":"two"}\n', encoding="utf-8"
            )

            with self.assertRaises(SourceChangedError):
                import_jsonl(
                    connection,
                    path,
                    source="test-provider",
                    stream="matches",
                )
            checkpoint_after = dict(
                connection.execute(
                    "SELECT cursor, metadata_json FROM ingestion_checkpoint"
                ).fetchone()
            )
            self.assertEqual(checkpoint_before, checkpoint_after)
            self.assertEqual(
                1,
                connection.execute(
                    "SELECT COUNT(*) FROM raw_ingest_record"
                ).fetchone()[0],
            )

    def test_duplicate_record_ids_do_not_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.jsonl"
            path.write_text(
                '{"record_id":"same","value":1}\n'
                '{"record_id":"same","value":1}\n',
                encoding="utf-8",
            )
            connection = self._database(root)
            result = import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
                batch_size=1,
            )

            self.assertTrue(result["eof"])
            self.assertEqual(1, result["imported"])
            self.assertEqual(2, result["seen"])

    def test_record_capture_time_is_preserved_as_observed_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.jsonl"
            captured_at = "2026-08-25T12:34:56+00:00"
            path.write_text(
                json.dumps({"record_id": "one", "observed_at": captured_at}) + "\n",
                encoding="utf-8",
            )
            connection = self._database(root)

            import_jsonl(
                connection,
                path,
                source="test-provider",
                stream="matches",
            )

            observed_at = connection.execute(
                "SELECT observed_at FROM raw_ingest_record"
            ).fetchone()[0]
            self.assertEqual(captured_at, observed_at)

    def test_conflict_keeps_last_committed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.jsonl"
            path.write_text(
                '{"record_id":"same","value":1}\n'
                '{"record_id":"same","value":2}\n',
                encoding="utf-8",
            )
            connection = self._database(root)

            with self.assertRaises(RecordConflictError):
                import_jsonl(
                    connection,
                    path,
                    source="test-provider",
                    stream="matches",
                    batch_size=1,
                )

            self.assertEqual(
                1,
                connection.execute(
                    "SELECT COUNT(*) FROM raw_ingest_record"
                ).fetchone()[0],
            )
            checkpoint = connection.execute(
                "SELECT cursor, metadata_json FROM ingestion_checkpoint"
            ).fetchone()
            first_line_size = len(b'{"record_id":"same","value":1}\n')
            self.assertEqual(first_line_size, int(checkpoint["cursor"]))
            self.assertEqual(
                "running", json.loads(checkpoint["metadata_json"])["status"]
            )


if __name__ == "__main__":
    unittest.main()
