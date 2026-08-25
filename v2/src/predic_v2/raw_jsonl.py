from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARSER_VERSION = "generic-jsonl-v1"


class SourceChangedError(ValueError):
    """Raised when a checkpoint points into a different file revision."""


class RecordConflictError(ValueError):
    """Raised when one stable record ID has two different payloads."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_text(record: dict[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _decode_record(raw_line: bytes, offset: int) -> tuple[object, ...]:
    try:
        decoded = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL record at byte offset {offset}: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"JSONL record at byte offset {offset} must be an object")

    canonical = json.dumps(
        decoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source_record_id = decoded.get("record_id", content_hash)
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError(f"record_id at byte offset {offset} must be a non-empty string")
    record_kind = decoded.get("kind", "unknown")
    if not isinstance(record_kind, str) or not record_kind:
        raise ValueError(f"kind at byte offset {offset} must be a non-empty string")
    return (
        source_record_id,
        record_kind,
        _optional_text(decoded, "event_at"),
        _optional_text(decoded, "known_at"),
        content_hash,
        canonical,
        _optional_text(decoded, "observed_at"),
    )


def import_jsonl(
    connection: sqlite3.Connection,
    jsonl_path: str | Path,
    *,
    source: str,
    stream: str,
    batch_size: int = 1000,
    max_records: int | None = None,
    point_in_time_eligible: bool = False,
    license_ref: str | None = None,
) -> dict[str, object]:
    """Ingest an authorized JSONL export with transactional byte checkpoints.

    Every non-empty line is an envelope with optional ``record_id``, ``kind``,
    ``event_at`` and ``known_at`` fields. The complete JSON object is retained.
    A batch and its byte offset are committed in the same SQLite transaction,
    so a crash can replay at most an uncommitted batch without duplicating rows.
    """

    source = source.strip()
    stream = stream.strip()
    if not source or not stream:
        raise ValueError("source and stream must be non-empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_records is not None and max_records < 0:
        raise ValueError("max_records cannot be negative")

    path = Path(jsonl_path).resolve()
    content_hash = _file_sha256(path)
    file_size = path.stat().st_size
    checkpoint = connection.execute(
        """
        SELECT cursor, high_watermark, metadata_json
        FROM ingestion_checkpoint
        WHERE source = ? AND stream = ?
        """,
        (source, stream),
    ).fetchone()
    if checkpoint is None:
        offset = 0
        previous_high_watermark = None
        records_seen = 0
        records_inserted = 0
    else:
        metadata = json.loads(checkpoint["metadata_json"])
        checkpoint_hash = metadata.get("content_sha256")
        if checkpoint_hash != content_hash:
            raise SourceChangedError(
                "the JSONL file changed after this stream was checkpointed; "
                "use a new stream name for the new immutable revision"
            )
        offset = int(checkpoint["cursor"] or 0)
        if offset > file_size:
            raise SourceChangedError("checkpoint byte offset is beyond the file size")
        previous_high_watermark = checkpoint["high_watermark"]
        records_seen = int(metadata.get("records_seen", 0))
        records_inserted = int(metadata.get("records_inserted", 0))

    observed_at = _utc_now()
    snapshot_id = f"jsonl:{_digest(source, content_hash, PARSER_VERSION)[:32]}"
    connection.execute("PRAGMA synchronous = FULL")
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO source_snapshot (
                snapshot_id, source, source_locator, observed_at,
                content_sha256, parser_version, license_ref,
                point_in_time_eligible, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                source,
                str(path),
                observed_at,
                content_hash,
                PARSER_VERSION,
                license_ref,
                int(point_in_time_eligible),
                json.dumps({"format": "jsonl", "stream": stream}, sort_keys=True),
            ),
        )

    imported_this_run = 0
    seen_this_run = 0
    high_watermark = previous_high_watermark
    eof = False

    with path.open("rb") as input_stream:
        input_stream.seek(offset)
        while not eof:
            batch: list[tuple[object, ...]] = []
            batch_hashes: dict[str, str] = {}
            batch_lines = 0
            batch_end = input_stream.tell()
            while batch_lines < batch_size:
                if max_records is not None and seen_this_run >= max_records:
                    break
                line_offset = input_stream.tell()
                raw_line = input_stream.readline()
                if not raw_line:
                    eof = True
                    break
                batch_end = input_stream.tell()
                batch_lines += 1
                if not raw_line.strip():
                    continue
                decoded = _decode_record(raw_line, line_offset)
                (
                    source_record_id,
                    record_kind,
                    event_at,
                    known_at,
                    record_hash,
                    canonical,
                    record_observed_at,
                ) = decoded
                record_hash = str(record_hash)
                previous_hash = batch_hashes.get(str(source_record_id))
                if previous_hash is not None and previous_hash != record_hash:
                    raise RecordConflictError(
                        f"record_id {source_record_id!r} has conflicting payloads "
                        f"inside the same batch"
                    )
                batch_hashes[str(source_record_id)] = record_hash
                raw_record_id = f"raw:{_digest(source, stream, source_record_id)[:40]}"
                batch.append(
                    (
                        raw_record_id,
                        source,
                        stream,
                        source_record_id,
                        record_kind,
                        event_at,
                        known_at,
                        record_hash,
                        canonical,
                        record_observed_at or observed_at,
                        snapshot_id,
                    )
                )
                seen_this_run += 1
                candidate = known_at or event_at
                if candidate is not None and (
                    high_watermark is None or candidate > high_watermark
                ):
                    high_watermark = candidate

            if batch_lines == 0 and not batch:
                break

            before_changes = connection.total_changes
            with connection:
                for record in batch:
                    existing = connection.execute(
                        """
                        SELECT content_sha256
                        FROM raw_ingest_record
                        WHERE source = ? AND stream = ? AND source_record_id = ?
                        """,
                        (record[1], record[2], record[3]),
                    ).fetchone()
                    if existing is not None and existing["content_sha256"] != record[7]:
                        raise RecordConflictError(
                            f"record_id {record[3]!r} was already ingested with "
                            "different content"
                        )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO raw_ingest_record (
                        raw_record_id, source, stream, source_record_id,
                        record_kind, event_at, known_at, content_sha256,
                        payload_json, observed_at, source_snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                inserted_in_batch = connection.total_changes - before_changes
                new_records_seen = records_seen + seen_this_run
                new_records_inserted = (
                    records_inserted + imported_this_run + inserted_in_batch
                )
                checkpoint_metadata = json.dumps(
                    {
                        "content_sha256": content_hash,
                        "file_size": file_size,
                        "records_inserted": new_records_inserted,
                        "records_seen": new_records_seen,
                        "source_locator": str(path),
                        "status": "complete" if batch_end == file_size else "running",
                    },
                    sort_keys=True,
                )
                connection.execute(
                    """
                    INSERT INTO ingestion_checkpoint (
                        source, stream, cursor, high_watermark, updated_at,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, stream) DO UPDATE SET
                        cursor = excluded.cursor,
                        high_watermark = excluded.high_watermark,
                        updated_at = excluded.updated_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        source,
                        stream,
                        str(batch_end),
                        high_watermark,
                        _utc_now(),
                        checkpoint_metadata,
                    ),
                )
            imported_this_run += inserted_in_batch
            offset = batch_end
            if max_records is not None and seen_this_run >= max_records:
                break

    return {
        "cursor": offset,
        "eof": offset == file_size,
        "file_size": file_size,
        "imported": imported_this_run,
        "seen": seen_this_run,
        "source": source,
        "stream": stream,
    }
