from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

import fcntl

from .hltv_capture import (
    AuthorizationWindowError,
    CaptureBusyError,
    CapturePolicy,
    CapturePolicyError,
    ResponseLengthMismatchError,
    ResponseTooLargeError,
    load_policy,
)


CAPTURE_VERSION = "bo3-api-capture-v1"
API_ROOT = "https://api.bo3.gg/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
PROFILE_LEVEL = {"catalog": 0, "core": 1, "rich": 2, "exhaustive": 3}
TASK_PRIORITY = {
    "catalog": 10,
    "match": 20,
    "game": 30,
    "game_players": 40,
    "game_kills_matrix": 50,
    "game_flashes_matrix": 51,
    "game_grenades_stats": 52,
    "game_hit_group_stats": 53,
    "game_weapons_stats": 54,
    "round_players": 60,
}
PERMANENT_BLOCK_CODES = {401, 403, 406, 418, 451}
RETRYABLE_CODES = {408, 425, 429, 500, 502, 503, 504}
CORE_PLAYER_FIELDS = ("kills", "death", "assists", "damage", "adr", "kast")


class Bo3CaptureError(RuntimeError):
    """Base error for the BO3 raw capture pipeline."""


class Bo3SourceChangedError(Bo3CaptureError):
    """Raised when one immutable crawl stream is resumed with new inputs."""


class Bo3QualityError(Bo3CaptureError):
    """Raised when HTTP succeeds but the JSON does not prove the requested entity."""


class Bo3HostCircuitOpenError(Bo3CaptureError):
    """Raised before network use while a persistent host stop is active."""


class Bo3StorageError(Bo3CaptureError):
    """Raised when a response cannot be durably stored."""


@dataclass(frozen=True)
class _Task:
    stream: str
    task_key: str
    kind: str
    source_id: str
    url: str
    attempts: int
    metadata: dict[str, Any]


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        return None


_STATE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS bo3_state_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bo3_capture_job (
    stream TEXT PRIMARY KEY,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    statuses TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    page_limit INTEGER NOT NULL,
    profile TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    authorization_ref TEXT NOT NULL,
    capture_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bo3_window (
    stream TEXT NOT NULL REFERENCES bo3_capture_job(stream),
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    expected_count INTEGER,
    expected_pages INTEGER,
    first_page_sha256 TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (stream, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS bo3_task (
    stream TEXT NOT NULL REFERENCES bo3_capture_job(stream),
    task_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    parent_task_key TEXT,
    url TEXT NOT NULL,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_eligible_at TEXT,
    last_status_code INTEGER,
    last_error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (stream, task_key)
);

CREATE INDEX IF NOT EXISTS idx_bo3_task_pending
ON bo3_task(stream, status, priority, next_eligible_at, task_key);

CREATE TABLE IF NOT EXISTS bo3_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    stream TEXT NOT NULL,
    task_key TEXT NOT NULL,
    source_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    content_type TEXT,
    content_sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    object_path TEXT NOT NULL,
    response_headers_json TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    quality_error TEXT,
    FOREIGN KEY (stream, task_key) REFERENCES bo3_task(stream, task_key)
);

CREATE INDEX IF NOT EXISTS idx_bo3_snapshot_task
ON bo3_snapshot(stream, task_key, observed_at);

CREATE TABLE IF NOT EXISTS bo3_match_index (
    stream TEXT NOT NULL REFERENCES bo3_capture_job(stream),
    match_id INTEGER NOT NULL,
    slug TEXT NOT NULL,
    status TEXT,
    parsed_status TEXT,
    start_date TEXT NOT NULL,
    end_date TEXT,
    bo_type INTEGER,
    game_version INTEGER,
    team1_id INTEGER,
    team2_id INTEGER,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    detail_complete INTEGER NOT NULL DEFAULT 0,
    last_snapshot_id TEXT NOT NULL,
    PRIMARY KEY (stream, match_id),
    UNIQUE (stream, slug)
);

CREATE INDEX IF NOT EXISTS idx_bo3_match_start
ON bo3_match_index(stream, start_date, match_id);

CREATE TABLE IF NOT EXISTS bo3_game_index (
    stream TEXT NOT NULL REFERENCES bo3_capture_job(stream),
    game_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL,
    map_number INTEGER,
    map_name TEXT,
    status TEXT,
    rounds_count INTEGER,
    stats_expected INTEGER NOT NULL DEFAULT 0,
    game_detail_complete INTEGER NOT NULL DEFAULT 0,
    player_rows INTEGER,
    distinct_players INTEGER,
    distinct_teams INTEGER,
    players_complete INTEGER NOT NULL DEFAULT 0,
    player_quality_error TEXT,
    last_snapshot_id TEXT NOT NULL,
    PRIMARY KEY (stream, game_id),
    FOREIGN KEY (stream, match_id) REFERENCES bo3_match_index(stream, match_id)
);

CREATE INDEX IF NOT EXISTS idx_bo3_game_match
ON bo3_game_index(stream, match_id, map_number);

CREATE TABLE IF NOT EXISTS bo3_player_map_index (
    stream TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    steam_profile_id INTEGER NOT NULL,
    steam_id_64 TEXT,
    team_id INTEGER NOT NULL,
    nickname TEXT,
    metrics_complete INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    PRIMARY KEY (stream, game_id, steam_profile_id),
    FOREIGN KEY (stream, game_id) REFERENCES bo3_game_index(stream, game_id),
    FOREIGN KEY (snapshot_id) REFERENCES bo3_snapshot(snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_bo3_player_timeline
ON bo3_player_map_index(stream, steam_profile_id, game_id);

CREATE TABLE IF NOT EXISTS bo3_host_state (
    authority TEXT PRIMARY KEY,
    last_request_at TEXT NOT NULL,
    blocked_reason TEXT,
    not_before_at TEXT
);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Bo3QualityError(f"{field} is not an ISO datetime: {value!r}") from error
    if parsed.tzinfo is None:
        raise Bo3QualityError(f"{field} must have a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _digest(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(_STATE_SCHEMA)
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(bo3_game_index)")
    }
    if "stats_expected" not in columns:
        connection.execute(
            "ALTER TABLE bo3_game_index "
            "ADD COLUMN stats_expected INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            """
            UPDATE bo3_game_index
            SET stats_expected = CASE WHEN map_name IS NOT NULL THEN 1 ELSE 0 END
            """
        )
    connection.execute(
        """
        INSERT INTO bo3_state_meta (key, value) VALUES ('schema_version', '1')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )
    connection.commit()
    return connection


def _open_existing_state(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"BO3 capture state does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _acquire_lock(state_path: Path) -> BinaryIO:
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise CaptureBusyError(
            f"another BO3 capture process is already using {state_path}"
        ) from error
    return handle


def _release_lock(handle: BinaryIO) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _canonical_statuses(statuses: Iterable[str]) -> tuple[str, ...]:
    allowed = {"finished", "defwin", "upcoming", "current", "cancelled"}
    result = tuple(dict.fromkeys(item.strip().lower() for item in statuses if item.strip()))
    if not result:
        raise ValueError("at least one BO3 match status is required")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"unsupported BO3 statuses: {', '.join(unknown)}")
    return result


def _catalog_url(
    window_start: date,
    window_end: date,
    statuses: tuple[str, ...],
    offset: int,
    limit: int,
) -> str:
    lower_bound = datetime.combine(
        window_start, datetime.min.time(), timezone.utc
    ) - timedelta(microseconds=1)
    params = [
        ("scope", "widget-map-pool"),
        ("page[offset]", str(offset)),
        ("page[limit]", str(limit)),
        ("sort", "start_date,id"),
        ("filter[matches.status][in]", ",".join(statuses)),
        (
            "filter[matches.start_date][gt]",
            lower_bound.isoformat().replace("+00:00", "Z"),
        ),
        ("filter[matches.start_date][lt]", f"{window_end.isoformat()}T00:00:00Z"),
        ("filter[matches.discipline_id][eq]", "1"),
        ("with", "teams,tournament,ai_predictions,games,streams,match_maps"),
    ]
    return f"{API_ROOT}/matches?{urlencode(params)}"


def _match_url(slug: str) -> str:
    params = urlencode(
        {
            "scope": "show-match",
            "with": (
                "games,streams,teams,tournament_deep,stage,"
                "ai_predictions,match_maps"
            ),
        }
    )
    return f"{API_ROOT}/matches/{slug}?{params}"


def _task_url(kind: str, source_id: str) -> str:
    if kind == "game":
        return f"{API_ROOT}/games/{source_id}"
    if kind == "game_players":
        return f"{API_ROOT}/games/{source_id}/players_stats"
    suffixes = {
        "game_kills_matrix": "kills_matrix",
        "game_flashes_matrix": "flashes_matrix",
        "game_grenades_stats": "grenades_stats",
        "game_hit_group_stats": "hit_group_stats",
        "game_weapons_stats": "weapons_stats",
    }
    if kind in suffixes:
        return f"{API_ROOT}/games/{source_id}/{suffixes[kind]}"
    if kind == "round_players":
        game_id, round_number = source_id.split(":", 1)
        return (
            f"{API_ROOT}/games/{game_id}/rounds/{round_number}/players_stats"
        )
    raise AssertionError(kind)


def _insert_task(
    connection: sqlite3.Connection,
    *,
    stream: str,
    task_key: str,
    kind: str,
    source_id: str,
    url: str,
    now: str,
    metadata: Mapping[str, Any] | None = None,
    parent_task_key: str | None = None,
) -> None:
    existing = connection.execute(
        "SELECT kind, source_id, url FROM bo3_task WHERE stream = ? AND task_key = ?",
        (stream, task_key),
    ).fetchone()
    if existing is not None:
        expected = (kind, source_id, url)
        actual = (existing["kind"], existing["source_id"], existing["url"])
        if actual != expected:
            raise Bo3SourceChangedError(
                f"task {task_key!r} changed inside immutable stream {stream!r}"
            )
        return
    connection.execute(
        """
        INSERT INTO bo3_task (
            stream, task_key, kind, source_id, parent_task_key, url,
            priority, status, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """,
        (
            stream,
            task_key,
            kind,
            source_id,
            parent_task_key,
            url,
            TASK_PRIORITY[kind],
            json.dumps(dict(metadata or {}), sort_keys=True),
            now,
            now,
        ),
    )


def _seed_job(
    connection: sqlite3.Connection,
    *,
    stream: str,
    start_date: date,
    end_date: date,
    statuses: tuple[str, ...],
    window_days: int,
    page_limit: int,
    profile: str,
    output_dir: Path,
    policy: CapturePolicy,
    now: datetime,
) -> None:
    if start_date >= end_date:
        raise ValueError("BO3 end_date must be after start_date (end is exclusive)")
    if window_days < 1 or window_days > 90:
        raise ValueError("window_days must be between 1 and 90")
    if page_limit < 1 or page_limit > 100:
        raise ValueError("page_limit must be between 1 and 100")
    if profile not in PROFILE_LEVEL:
        raise ValueError(f"unknown BO3 profile: {profile}")

    output = str(output_dir.resolve())
    statuses_text = ",".join(statuses)
    row = connection.execute(
        "SELECT * FROM bo3_capture_job WHERE stream = ?", (stream,)
    ).fetchone()
    now_text = _iso(now)
    immutable = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "statuses": statuses_text,
        "window_days": window_days,
        "page_limit": page_limit,
        "output_dir": output,
        "authorization_ref": policy.authorization_ref,
        "capture_version": CAPTURE_VERSION,
    }
    if row is None:
        connection.execute(
            """
            INSERT INTO bo3_capture_job (
                stream, start_date, end_date, statuses, window_days,
                page_limit, profile, output_dir, authorization_ref,
                capture_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stream,
                immutable["start_date"],
                immutable["end_date"],
                statuses_text,
                window_days,
                page_limit,
                profile,
                output,
                policy.authorization_ref,
                CAPTURE_VERSION,
                now_text,
                now_text,
            ),
        )
    else:
        changed = [key for key, value in immutable.items() if row[key] != value]
        if changed:
            raise Bo3SourceChangedError(
                f"stream {stream!r} changed immutable fields: {', '.join(changed)}; "
                "use a new stream"
            )
        old_level = PROFILE_LEVEL[str(row["profile"])]
        new_level = PROFILE_LEVEL[profile]
        effective_profile = profile if new_level > old_level else str(row["profile"])
        connection.execute(
            "UPDATE bo3_capture_job SET profile = ?, updated_at = ? WHERE stream = ?",
            (effective_profile, now_text, stream),
        )

    cursor = start_date
    while cursor < end_date:
        stop = min(cursor + timedelta(days=window_days), end_date)
        connection.execute(
            """
            INSERT OR IGNORE INTO bo3_window (
                stream, window_start, window_end, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (stream, cursor.isoformat(), stop.isoformat(), now_text),
        )
        task_key = f"catalog:{cursor.isoformat()}:{stop.isoformat()}:0"
        _insert_task(
            connection,
            stream=stream,
            task_key=task_key,
            kind="catalog",
            source_id=f"{cursor.isoformat()}:{stop.isoformat()}:0",
            url=_catalog_url(cursor, stop, statuses, 0, page_limit),
            now=now_text,
            metadata={
                "window_start": cursor.isoformat(),
                "window_end": stop.isoformat(),
                "offset": 0,
                "limit": page_limit,
            },
        )
        cursor = stop
    connection.commit()


def plan_bo3_capture(
    policy_path: str | Path,
    *,
    start_date: date,
    end_date: date,
    statuses: Iterable[str] = ("finished", "defwin"),
    window_days: int = 7,
    page_limit: int = 100,
    profile: str = "core",
) -> dict[str, object]:
    policy, _ = load_policy(policy_path, require_live=False)
    canonical_statuses = _canonical_statuses(statuses)
    if start_date >= end_date:
        raise ValueError("BO3 end_date must be after start_date")
    if window_days < 1 or window_days > 90:
        raise ValueError("window_days must be between 1 and 90")
    if page_limit < 1 or page_limit > 100:
        raise ValueError("page_limit must be between 1 and 100")
    if profile not in PROFILE_LEVEL:
        raise ValueError(f"unknown BO3 profile: {profile}")
    windows = math.ceil((end_date - start_date).days / window_days)
    sample_url = _catalog_url(
        start_date,
        min(start_date + timedelta(days=window_days), end_date),
        canonical_statuses,
        0,
        page_limit,
    )
    policy.validate_url(sample_url)
    return {
        "api_root": API_ROOT,
        "capture_version": CAPTURE_VERSION,
        "end_date_exclusive": end_date.isoformat(),
        "initial_catalog_requests": windows,
        "min_interval_seconds": policy.min_interval_seconds,
        "profile": profile,
        "start_date": start_date.isoformat(),
        "statuses": list(canonical_statuses),
        "window_days": window_days,
        "written_authorization_required_for_live": True,
    }


def _authority(url: str) -> str:
    parsed = urlsplit(url)
    authority = str(parsed.hostname).lower()
    if parsed.port not in (None, 443):
        authority = f"{authority}:{parsed.port}"
    return authority


def _wait_for_slot(
    connection: sqlite3.Connection,
    authority: str,
    interval: float,
    *,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[float], None],
) -> datetime:
    while True:
        now = now_fn().astimezone(timezone.utc)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM bo3_host_state WHERE authority = ?", (authority,)
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO bo3_host_state (authority, last_request_at) VALUES (?, ?)",
                (authority, _iso(now)),
            )
            connection.commit()
            return now
        if row["blocked_reason"] is not None:
            not_before = row["not_before_at"]
            if not_before is None or now < _parse_datetime(not_before, "not_before_at"):
                connection.rollback()
                raise Bo3HostCircuitOpenError(str(row["blocked_reason"]))
            connection.execute(
                """
                UPDATE bo3_host_state
                SET blocked_reason = NULL, not_before_at = NULL
                WHERE authority = ?
                """,
                (authority,),
            )
        last = _parse_datetime(row["last_request_at"], "last_request_at")
        remaining = interval - (now - last).total_seconds()
        if remaining <= 0:
            connection.execute(
                "UPDATE bo3_host_state SET last_request_at = ? WHERE authority = ?",
                (_iso(now), authority),
            )
            connection.commit()
            return now
        connection.rollback()
        sleep_fn(remaining)


def _safe_headers(headers: Any) -> dict[str, str]:
    keep = {
        "content-type",
        "content-length",
        "content-encoding",
        "etag",
        "last-modified",
        "cache-control",
        "retry-after",
        "location",
        "date",
    }
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if str(key).lower() in keep
    }


def _retry_at(headers: Any, now: datetime, fallback_seconds: float) -> datetime:
    fallback = now + timedelta(seconds=fallback_seconds)
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return fallback
    try:
        return max(fallback, now + timedelta(seconds=max(0, int(str(raw).strip()))))
    except (ValueError, OverflowError):
        try:
            parsed = parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(fallback, parsed.astimezone(timezone.utc))
        except (TypeError, ValueError, OverflowError):
            return fallback


def _read_limited(response: Any, limit: int) -> bytes:
    declared = response.headers.get("Content-Length")
    declared_size: int | None = None
    if declared is not None:
        try:
            declared_size = int(declared)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > limit:
            raise ResponseTooLargeError(
                f"declared response size {declared_size} exceeds {limit}"
            )
    chunks: list[bytes] = []
    size = 0
    while True:
        block = response.read(min(64 * 1024, limit - size + 1))
        if not block:
            break
        chunks.append(block)
        size += len(block)
        if size > limit:
            raise ResponseTooLargeError(f"response exceeds {limit} bytes")
    body = b"".join(chunks)
    if declared_size is not None and declared_size != len(body):
        raise ResponseLengthMismatchError(
            f"response has {len(body)} bytes but declared {declared_size}"
        )
    return body


def _store_json(output_dir: Path, body: bytes) -> tuple[str, Path]:
    content_hash = hashlib.sha256(body).hexdigest()
    relative = Path("objects") / content_hash[:2] / f"{content_hash}.json"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256_file(target) != content_hash:
            raise Bo3StorageError(f"corrupt content-addressed object: {target}")
        return content_hash, relative
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_hash, relative


def _request_json(opener: Any, url: str, policy: CapturePolicy, timeout: float) -> Any:
    request = Request(
        url,
        method="GET",
        headers={
            "User-Agent": policy.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
    )
    return opener.open(request, timeout=timeout)


def _select_task(
    connection: sqlite3.Connection, stream: str, now: datetime
) -> _Task | None:
    row = connection.execute(
        """
        SELECT * FROM bo3_task
        WHERE stream = ?
          AND status IN ('pending', 'retry')
          AND (next_eligible_at IS NULL OR next_eligible_at <= ?)
        ORDER BY priority, task_key
        LIMIT 1
        """,
        (stream, _iso(now)),
    ).fetchone()
    if row is None:
        return None
    return _Task(
        stream=stream,
        task_key=str(row["task_key"]),
        kind=str(row["kind"]),
        source_id=str(row["source_id"]),
        url=str(row["url"]),
        attempts=int(row["attempts"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _start_task(connection: sqlite3.Connection, task: _Task, now: datetime) -> int:
    attempts = task.attempts + 1
    with connection:
        connection.execute(
            """
            UPDATE bo3_task
            SET status = 'running', attempts = ?, next_eligible_at = NULL,
                last_error = NULL, updated_at = ?
            WHERE stream = ? AND task_key = ?
            """,
            (attempts, _iso(now), task.stream, task.task_key),
        )
    return attempts


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Bo3QualityError(f"{field} must be an integer")
    return value


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Bo3QualityError(f"{field} must be a non-empty string")
    return value


def _profile(connection: sqlite3.Connection, stream: str) -> str:
    row = connection.execute(
        "SELECT profile FROM bo3_capture_job WHERE stream = ?", (stream,)
    ).fetchone()
    assert row is not None
    return str(row[0])


def _upsert_match(
    connection: sqlite3.Connection,
    *,
    stream: str,
    payload: dict[str, Any],
    window_start: str,
    window_end: str,
    snapshot_id: str,
    detail_complete: bool,
    now: str,
    parent_task_key: str,
) -> int:
    match_id = _integer(payload.get("id"), "match.id")
    slug = _text(payload.get("slug"), "match.slug")
    start_at = _text(payload.get("start_date"), "match.start_date")
    _parse_datetime(start_at, "match.start_date")
    existing = connection.execute(
        "SELECT slug, start_date FROM bo3_match_index WHERE stream = ? AND match_id = ?",
        (stream, match_id),
    ).fetchone()
    if existing is not None and (
        existing["slug"] != slug or existing["start_date"] != start_at
    ):
        raise Bo3QualityError(f"match identity changed for BO3 id {match_id}")
    connection.execute(
        """
        INSERT INTO bo3_match_index (
            stream, match_id, slug, status, parsed_status, start_date,
            end_date, bo_type, game_version, team1_id, team2_id,
            window_start, window_end, detail_complete, last_snapshot_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stream, match_id) DO UPDATE SET
            status = excluded.status,
            parsed_status = COALESCE(excluded.parsed_status, bo3_match_index.parsed_status),
            end_date = COALESCE(excluded.end_date, bo3_match_index.end_date),
            bo_type = COALESCE(excluded.bo_type, bo3_match_index.bo_type),
            game_version = COALESCE(excluded.game_version, bo3_match_index.game_version),
            team1_id = COALESCE(excluded.team1_id, bo3_match_index.team1_id),
            team2_id = COALESCE(excluded.team2_id, bo3_match_index.team2_id),
            detail_complete = MAX(bo3_match_index.detail_complete, excluded.detail_complete),
            last_snapshot_id = excluded.last_snapshot_id
        """,
        (
            stream,
            match_id,
            slug,
            payload.get("status"),
            payload.get("parsed_status"),
            start_at,
            payload.get("end_date"),
            _optional_integer(payload.get("bo_type"), "match.bo_type"),
            _optional_integer(payload.get("game_version"), "match.game_version"),
            _optional_integer(payload.get("team1_id"), "match.team1_id"),
            _optional_integer(payload.get("team2_id"), "match.team2_id"),
            window_start,
            window_end,
            int(detail_complete),
            snapshot_id,
        ),
    )

    if PROFILE_LEVEL[_profile(connection, stream)] >= PROFILE_LEVEL["core"]:
        _insert_task(
            connection,
            stream=stream,
            task_key=f"match:{match_id}",
            kind="match",
            source_id=str(match_id),
            url=_match_url(slug),
            now=now,
            metadata={
                "match_id": match_id,
                "slug": slug,
                "window_start": window_start,
                "window_end": window_end,
            },
            parent_task_key=parent_task_key,
        )
    games = payload.get("games") or []
    if not isinstance(games, list):
        raise Bo3QualityError("match.games must be an array or null")
    for game in games:
        if not isinstance(game, dict):
            raise Bo3QualityError("match.games items must be objects")
        _upsert_game_stub(
            connection,
            stream=stream,
            match_id=match_id,
            payload=game,
            parent_finished=(
                payload.get("status") == "finished"
                or str(payload.get("parsed_status") or "") == "done"
            ),
            snapshot_id=snapshot_id,
            now=now,
            parent_task_key=parent_task_key,
        )
    return match_id


def _upsert_game_stub(
    connection: sqlite3.Connection,
    *,
    stream: str,
    match_id: int,
    payload: dict[str, Any],
    parent_finished: bool,
    snapshot_id: str,
    now: str,
    parent_task_key: str,
) -> int:
    game_id = _integer(payload.get("id"), "game.id")
    stats_expected = bool(
        parent_finished
        and (
            payload.get("map_name")
            or payload.get("rounds_count")
            or payload.get("begin_at")
            or payload.get("end_at")
        )
    )
    existing = connection.execute(
        "SELECT match_id FROM bo3_game_index WHERE stream = ? AND game_id = ?",
        (stream, game_id),
    ).fetchone()
    if existing is not None and int(existing["match_id"]) != match_id:
        raise Bo3QualityError(f"game {game_id} moved to another match")
    connection.execute(
        """
        INSERT INTO bo3_game_index (
            stream, game_id, match_id, map_number, map_name, status,
            rounds_count, stats_expected, last_snapshot_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stream, game_id) DO UPDATE SET
            map_number = COALESCE(excluded.map_number, bo3_game_index.map_number),
            map_name = COALESCE(excluded.map_name, bo3_game_index.map_name),
            status = COALESCE(excluded.status, bo3_game_index.status),
            rounds_count = COALESCE(excluded.rounds_count, bo3_game_index.rounds_count),
            stats_expected = MAX(bo3_game_index.stats_expected, excluded.stats_expected),
            last_snapshot_id = excluded.last_snapshot_id
        """,
        (
            stream,
            game_id,
            match_id,
            _optional_integer(payload.get("number"), "game.number"),
            payload.get("map_name"),
            payload.get("status"),
            _optional_integer(payload.get("rounds_count"), "game.rounds_count"),
            int(stats_expected),
            snapshot_id,
        ),
    )
    if (
        stats_expected
        and PROFILE_LEVEL[_profile(connection, stream)] >= PROFILE_LEVEL["core"]
    ):
        _insert_task(
            connection,
            stream=stream,
            task_key=f"game:{game_id}",
            kind="game",
            source_id=str(game_id),
            url=_task_url("game", str(game_id)),
            now=now,
            metadata={"game_id": game_id, "match_id": match_id},
            parent_task_key=parent_task_key,
        )
    return game_id


def _process_catalog(
    connection: sqlite3.Connection,
    task: _Task,
    payload: Any,
    snapshot_id: str,
    content_hash: str,
    now: str,
) -> None:
    if not isinstance(payload, dict):
        raise Bo3QualityError("catalog response must be an object")
    total = payload.get("total")
    results = payload.get("results")
    if not isinstance(total, dict) or not isinstance(results, list):
        raise Bo3QualityError("catalog response needs total object and results array")
    count = _integer(total.get("count"), "catalog.total.count")
    offset = _integer(total.get("offset"), "catalog.total.offset")
    limit = _integer(total.get("limit"), "catalog.total.limit")
    if offset != int(task.metadata["offset"]) or limit != int(task.metadata["limit"]):
        raise Bo3QualityError("catalog pagination metadata does not match request")
    if len(results) > limit:
        raise Bo3QualityError("catalog returned more rows than requested")
    window_start = str(task.metadata["window_start"])
    window_end = str(task.metadata["window_end"])
    row = connection.execute(
        """
        SELECT expected_count, expected_pages
        FROM bo3_window
        WHERE stream = ? AND window_start = ? AND window_end = ?
        """,
        (task.stream, window_start, window_end),
    ).fetchone()
    assert row is not None
    pages = math.ceil(count / limit) if count else 1
    if row["expected_count"] is not None and int(row["expected_count"]) != count:
        raise Bo3QualityError(
            f"catalog total changed inside closed window {window_start}..{window_end}"
        )
    connection.execute(
        """
        UPDATE bo3_window
        SET expected_count = ?, expected_pages = ?,
            first_page_sha256 = COALESCE(first_page_sha256, ?), updated_at = ?
        WHERE stream = ? AND window_start = ? AND window_end = ?
        """,
        (count, pages, content_hash, now, task.stream, window_start, window_end),
    )
    if offset == 0:
        statuses = tuple(
            str(
                connection.execute(
                    "SELECT statuses FROM bo3_capture_job WHERE stream = ?",
                    (task.stream,),
                ).fetchone()[0]
            ).split(",")
        )
        for next_offset in range(limit, count, limit):
            key = f"catalog:{window_start}:{window_end}:{next_offset}"
            _insert_task(
                connection,
                stream=task.stream,
                task_key=key,
                kind="catalog",
                source_id=f"{window_start}:{window_end}:{next_offset}",
                url=_catalog_url(
                    date.fromisoformat(window_start),
                    date.fromisoformat(window_end),
                    statuses,
                    next_offset,
                    limit,
                ),
                now=now,
                metadata={
                    "window_start": window_start,
                    "window_end": window_end,
                    "offset": next_offset,
                    "limit": limit,
                },
                parent_task_key=task.task_key,
            )
    seen_ids: set[int] = set()
    allowed_statuses = set(
        str(
            connection.execute(
                "SELECT statuses FROM bo3_capture_job WHERE stream = ?",
                (task.stream,),
            ).fetchone()[0]
        ).split(",")
    )
    lower = datetime.combine(date.fromisoformat(window_start), datetime.min.time(), timezone.utc)
    upper = datetime.combine(date.fromisoformat(window_end), datetime.min.time(), timezone.utc)
    for item in results:
        if not isinstance(item, dict):
            raise Bo3QualityError("catalog result must be an object")
        match_id = _integer(item.get("id"), "match.id")
        if match_id in seen_ids:
            raise Bo3QualityError(f"duplicate match id {match_id} inside catalog page")
        seen_ids.add(match_id)
        if item.get("status") is not None and item.get("status") not in allowed_statuses:
            raise Bo3QualityError(
                f"match {match_id} has status {item.get('status')!r} outside request"
            )
        discipline_id = item.get("discipline_id")
        if discipline_id is not None and discipline_id != 1:
            raise Bo3QualityError(f"match {match_id} is not CS/CS2 discipline 1")
        started = _parse_datetime(
            _text(item.get("start_date"), "match.start_date"),
            "match.start_date",
        )
        if not lower <= started < upper:
            raise Bo3QualityError(
                f"match {match_id} is outside requested window {window_start}..{window_end}"
            )
        _upsert_match(
            connection,
            stream=task.stream,
            payload=item,
            window_start=window_start,
            window_end=window_end,
            snapshot_id=snapshot_id,
            detail_complete=False,
            now=now,
            parent_task_key=task.task_key,
        )


def _process_match(
    connection: sqlite3.Connection,
    task: _Task,
    payload: Any,
    snapshot_id: str,
    now: str,
) -> None:
    if not isinstance(payload, dict):
        raise Bo3QualityError("match detail response must be an object")
    match_id = _integer(payload.get("id"), "match.id")
    if match_id != int(task.metadata["match_id"]):
        raise Bo3QualityError("match detail id does not match requested id")
    if payload.get("slug") != task.metadata["slug"]:
        raise Bo3QualityError("match detail slug does not match requested slug")
    allowed_statuses = set(
        str(
            connection.execute(
                "SELECT statuses FROM bo3_capture_job WHERE stream = ?",
                (task.stream,),
            ).fetchone()[0]
        ).split(",")
    )
    status = payload.get("status")
    if status not in allowed_statuses:
        raise Bo3QualityError(
            f"match detail {match_id} has status {status!r} outside request"
        )
    _upsert_match(
        connection,
        stream=task.stream,
        payload=payload,
        window_start=str(task.metadata["window_start"]),
        window_end=str(task.metadata["window_end"]),
        snapshot_id=snapshot_id,
        detail_complete=True,
        now=now,
        parent_task_key=task.task_key,
    )


def _process_game(
    connection: sqlite3.Connection,
    task: _Task,
    payload: Any,
    snapshot_id: str,
    now: str,
) -> None:
    if not isinstance(payload, dict):
        raise Bo3QualityError("game response must be an object")
    game_id = _integer(payload.get("id"), "game.id")
    if game_id != int(task.metadata["game_id"]):
        raise Bo3QualityError("game id does not match request")
    match_id = _optional_integer(payload.get("match_id"), "game.match_id")
    expected_match = int(task.metadata["match_id"])
    if match_id is not None and match_id != expected_match:
        raise Bo3QualityError("game match_id does not match parent match")
    rounds = payload.get("game_rounds") or []
    if not isinstance(rounds, list):
        raise Bo3QualityError("game_rounds must be an array or null")
    round_numbers: list[int] = []
    for item in rounds:
        if not isinstance(item, dict):
            raise Bo3QualityError("game_rounds entries must be objects")
        round_numbers.append(_integer(item.get("round_number"), "round.round_number"))
    if len(set(round_numbers)) != len(round_numbers):
        raise Bo3QualityError("game has duplicate round numbers")
    declared_rounds = _optional_integer(payload.get("rounds_count"), "game.rounds_count")
    if declared_rounds is not None and rounds and declared_rounds != len(rounds):
        raise Bo3QualityError(
            f"game {game_id} declares {declared_rounds} rounds but contains {len(rounds)}"
        )
    connection.execute(
        """
        UPDATE bo3_game_index
        SET map_name = COALESCE(?, map_name), status = COALESCE(?, status),
            rounds_count = COALESCE(?, rounds_count), stats_expected = 1,
            game_detail_complete = 1,
            last_snapshot_id = ?
        WHERE stream = ? AND game_id = ?
        """,
        (
            payload.get("map_name"),
            payload.get("status"),
            declared_rounds if declared_rounds is not None else len(rounds) or None,
            snapshot_id,
            task.stream,
            game_id,
        ),
    )
    _insert_task(
        connection,
        stream=task.stream,
        task_key=f"game_players:{game_id}",
        kind="game_players",
        source_id=str(game_id),
        url=_task_url("game_players", str(game_id)),
        now=now,
        metadata={"game_id": game_id, "match_id": expected_match},
        parent_task_key=task.task_key,
    )
    profile = _profile(connection, task.stream)
    if PROFILE_LEVEL[profile] >= PROFILE_LEVEL["rich"]:
        for kind in (
            "game_kills_matrix",
            "game_flashes_matrix",
            "game_grenades_stats",
            "game_hit_group_stats",
            "game_weapons_stats",
        ):
            _insert_task(
                connection,
                stream=task.stream,
                task_key=f"{kind}:{game_id}",
                kind=kind,
                source_id=str(game_id),
                url=_task_url(kind, str(game_id)),
                now=now,
                metadata={"game_id": game_id, "match_id": expected_match},
                parent_task_key=task.task_key,
            )
    if PROFILE_LEVEL[profile] >= PROFILE_LEVEL["exhaustive"]:
        for round_number in sorted(round_numbers):
            source_id = f"{game_id}:{round_number}"
            _insert_task(
                connection,
                stream=task.stream,
                task_key=f"round_players:{source_id}",
                kind="round_players",
                source_id=source_id,
                url=_task_url("round_players", source_id),
                now=now,
                metadata={
                    "game_id": game_id,
                    "match_id": expected_match,
                    "round_number": round_number,
                },
                parent_task_key=task.task_key,
            )


def _player_identity(row: dict[str, Any]) -> tuple[int, int, str | None, str | None, bool]:
    profile_id = _integer(row.get("steam_profile_id"), "player.steam_profile_id")
    clan = row.get("team_clan")
    if not isinstance(clan, dict):
        raise Bo3QualityError(f"player {profile_id} has no team_clan object")
    team_id = _integer(clan.get("team_id"), "player.team_clan.team_id")
    profile = row.get("steam_profile")
    if not isinstance(profile, dict):
        raise Bo3QualityError(f"player {profile_id} has no steam_profile object")
    steam_id = profile.get("steam_id_64")
    if steam_id is not None:
        steam_id = str(steam_id)
    nickname = profile.get("nickname")
    if nickname is not None and not isinstance(nickname, str):
        raise Bo3QualityError("player nickname must be a string or null")
    metrics_complete = all(row.get(field) is not None for field in CORE_PLAYER_FIELDS)
    return profile_id, team_id, steam_id, nickname, metrics_complete


def _process_game_players(
    connection: sqlite3.Connection,
    task: _Task,
    payload: Any,
    snapshot_id: str,
) -> None:
    if not isinstance(payload, list):
        raise Bo3QualityError("game players response must be an array")
    if not payload:
        raise Bo3QualityError("game players response is empty")
    identities: list[tuple[int, int, str | None, str | None, bool]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise Bo3QualityError("game players entries must be objects")
        identities.append(_player_identity(item))
    profile_ids = [item[0] for item in identities]
    if len(set(profile_ids)) != len(profile_ids):
        raise Bo3QualityError("game players response has duplicate steam_profile_id")
    team_counts = Counter(item[1] for item in identities)
    metrics_complete = all(item[4] for item in identities)
    quality_errors: list[str] = []
    if len(identities) != 10:
        quality_errors.append(f"expected 10 player rows, got {len(identities)}")
    if sorted(team_counts.values()) != [5, 5]:
        quality_errors.append(f"expected two teams of five, got {dict(team_counts)}")
    if not metrics_complete:
        quality_errors.append("one or more player rows miss core metrics")
    game_id = int(task.metadata["game_id"])
    connection.execute(
        "DELETE FROM bo3_player_map_index WHERE stream = ? AND game_id = ?",
        (task.stream, game_id),
    )
    connection.executemany(
        """
        INSERT INTO bo3_player_map_index (
            stream, game_id, steam_profile_id, steam_id_64, team_id,
            nickname, metrics_complete, snapshot_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                task.stream,
                game_id,
                profile_id,
                steam_id,
                team_id,
                nickname,
                int(complete),
                snapshot_id,
            )
            for profile_id, team_id, steam_id, nickname, complete in identities
        ],
    )
    error_text = "; ".join(quality_errors) if quality_errors else None
    connection.execute(
        """
        UPDATE bo3_game_index
        SET player_rows = ?, distinct_players = ?, distinct_teams = ?,
            players_complete = ?, player_quality_error = ?, last_snapshot_id = ?
        WHERE stream = ? AND game_id = ?
        """,
        (
            len(identities),
            len(set(profile_ids)),
            len(team_counts),
            int(not quality_errors),
            error_text,
            snapshot_id,
            task.stream,
            game_id,
        ),
    )
    if quality_errors:
        raise Bo3QualityError(error_text or "incomplete player map")


def _process_generic_enrichment(task: _Task, payload: Any) -> None:
    if task.kind == "round_players":
        if not isinstance(payload, list) or not payload:
            raise Bo3QualityError("round player stats must be a non-empty array")
        ids = [
            _integer(item.get("steam_profile_id"), "round_player.steam_profile_id")
            for item in payload
            if isinstance(item, dict)
        ]
        if len(ids) != len(payload) or len(set(ids)) != len(ids):
            raise Bo3QualityError("round player stats have invalid or duplicate players")
        return
    if not isinstance(payload, (dict, list)):
        raise Bo3QualityError(f"{task.kind} response must be an object or array")


def _process_payload(
    connection: sqlite3.Connection,
    task: _Task,
    payload: Any,
    snapshot_id: str,
    content_hash: str,
    observed_at: str,
) -> None:
    if task.kind == "catalog":
        _process_catalog(
            connection, task, payload, snapshot_id, content_hash, observed_at
        )
    elif task.kind == "match":
        _process_match(connection, task, payload, snapshot_id, observed_at)
    elif task.kind == "game":
        _process_game(connection, task, payload, snapshot_id, observed_at)
    elif task.kind == "game_players":
        _process_game_players(connection, task, payload, snapshot_id)
    else:
        _process_generic_enrichment(task, payload)


def _quality_finish(
    connection: sqlite3.Connection,
    task: _Task,
    *,
    attempts: int,
    policy: CapturePolicy,
    observed_at: datetime,
    source_url: str,
    final_url: str,
    status_code: int,
    headers: Mapping[str, str],
    body: bytes,
    output_dir: Path,
) -> tuple[bool, str | None]:
    content_type = headers.get("content-type", "")
    if "json" not in content_type.casefold():
        raise Bo3QualityError(f"unexpected content type: {content_type!r}")
    if headers.get("content-encoding", "identity").casefold() not in {"", "identity"}:
        raise Bo3QualityError("encoded responses are not accepted")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Bo3QualityError(f"invalid JSON: {error}") from error
    try:
        content_hash, relative = _store_json(output_dir, body)
    except OSError as error:
        raise Bo3StorageError(f"cannot persist BO3 response: {error}") from error
    snapshot_id = f"bo3:{_digest(task.stream, task.task_key, _iso(observed_at), content_hash)[:40]}"
    quality_error: str | None = None
    complete = True
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            INSERT INTO bo3_snapshot (
                snapshot_id, stream, task_key, source_url, final_url,
                observed_at, status_code, content_type, content_sha256,
                byte_size, object_path, response_headers_json,
                quality_status, quality_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL)
            """,
            (
                snapshot_id,
                task.stream,
                task.task_key,
                source_url,
                final_url,
                _iso(observed_at),
                status_code,
                content_type or None,
                content_hash,
                len(body),
                str(relative),
                json.dumps(dict(headers), sort_keys=True),
            ),
        )
        try:
            _process_payload(
                connection,
                task,
                payload,
                snapshot_id,
                content_hash,
                _iso(observed_at),
            )
        except Bo3QualityError as error:
            quality_error = str(error)
            complete = False
        connection.execute(
            """
            UPDATE bo3_snapshot
            SET quality_status = ?, quality_error = ?
            WHERE snapshot_id = ?
            """,
            ("complete" if complete else "incomplete", quality_error, snapshot_id),
        )
        if complete:
            task_status = "complete"
            next_at = None
        elif attempts >= policy.max_attempts_per_url:
            task_status = "quarantined"
            next_at = None
        else:
            task_status = "retry"
            next_at = _iso(
                observed_at
                + timedelta(seconds=policy.base_backoff_seconds * (2 ** (attempts - 1)))
            )
        connection.execute(
            """
            UPDATE bo3_task
            SET status = ?, next_eligible_at = ?, last_status_code = ?,
                last_error = ?, updated_at = ?
            WHERE stream = ? AND task_key = ?
            """,
            (
                task_status,
                next_at,
                status_code,
                quality_error,
                _iso(observed_at),
                task.stream,
                task.task_key,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return complete, quality_error


def _finish_http_error(
    connection: sqlite3.Connection,
    task: _Task,
    *,
    attempts: int,
    status_code: int | None,
    error: str,
    headers: Any,
    policy: CapturePolicy,
    now: datetime,
) -> str:
    if status_code in PERMANENT_BLOCK_CODES:
        task_status = "retry"
        blocked_reason = f"http_{status_code}"
        not_before = None
    elif status_code in RETRYABLE_CODES or status_code is None:
        task_status = "quarantined" if attempts >= policy.max_attempts_per_url else "retry"
        blocked_reason = None
        not_before = _retry_at(
            headers,
            now,
            policy.base_backoff_seconds * (2 ** (attempts - 1)),
        )
    elif status_code == 404:
        task_status = "not_found"
        blocked_reason = None
        not_before = None
    else:
        task_status = "http_error"
        blocked_reason = None
        not_before = None
    with connection:
        connection.execute(
            """
            UPDATE bo3_task
            SET status = ?, next_eligible_at = ?, last_status_code = ?,
                last_error = ?, updated_at = ?
            WHERE stream = ? AND task_key = ?
            """,
            (
                task_status,
                _iso(not_before) if isinstance(not_before, datetime) else None,
                status_code,
                error,
                _iso(now),
                task.stream,
                task.task_key,
            ),
        )
        if blocked_reason is not None:
            connection.execute(
                """
                UPDATE bo3_host_state
                SET blocked_reason = ?, not_before_at = NULL
                WHERE authority = ?
                """,
                (blocked_reason, _authority(task.url)),
            )
        elif status_code == 429 and isinstance(not_before, datetime):
            connection.execute(
                """
                UPDATE bo3_host_state
                SET blocked_reason = 'http_429', not_before_at = ?
                WHERE authority = ?
                """,
                (_iso(not_before), _authority(task.url)),
            )
    return task_status


def capture_bo3(
    state_db: str | Path,
    output_dir: str | Path,
    *,
    stream: str,
    policy_path: str | Path,
    start_date: date,
    end_date: date,
    statuses: Iterable[str] = ("finished", "defwin"),
    window_days: int = 7,
    page_limit: int = 100,
    profile: str = "core",
    max_requests: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: OpenerDirector | Any | None = None,
    now_fn: Callable[[], datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Capture a bounded, resumable BO3 date range into raw content-addressed JSON.

    The end date is exclusive. A live run requires the same explicit written-
    authorization policy used by the HLTV transport; this is intentional because
    BO3 currently advertises ``ai-train=no`` in its robots content signal.
    """

    state_path = Path(state_db).resolve()
    output_path = Path(output_dir).resolve()
    policy, _ = load_policy(policy_path, require_live=True)
    if policy.robots_txt_mode != "written_permission_override":
        raise CapturePolicyError(
            "BO3 live ML capture requires a written_permission_override policy"
        )
    canonical_statuses = _canonical_statuses(statuses)
    if max_requests is None:
        max_requests = policy.max_http_requests_per_run
    if max_requests < 1 or max_requests > policy.max_http_requests_per_run:
        raise ValueError(
            "max_requests must be positive and no greater than policy budget"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    for url in (
        _catalog_url(
            start_date,
            min(start_date + timedelta(days=window_days), end_date),
            canonical_statuses,
            0,
            page_limit,
        ),
        f"{API_ROOT}/games/1",
    ):
        policy.validate_url(url)
    policy.assert_authorization_active(now_fn())

    lock = _acquire_lock(state_path)
    connection: sqlite3.Connection | None = None
    requests = 0
    stopped_reason: str | None = None
    try:
        connection = _connect_state(state_path)
        _seed_job(
            connection,
            stream=stream,
            start_date=start_date,
            end_date=end_date,
            statuses=canonical_statuses,
            window_days=window_days,
            page_limit=page_limit,
            profile=profile,
            output_dir=output_path,
            policy=policy,
            now=now_fn(),
        )
        with connection:
            connection.execute(
                """
                UPDATE bo3_task
                SET status = 'retry', last_error = 'interrupted before completion'
                WHERE stream = ? AND status = 'running'
                """,
                (stream,),
            )
        request_opener = opener or build_opener(_NoRedirectHandler())
        while requests < max_requests:
            now = now_fn().astimezone(timezone.utc)
            task = _select_task(connection, stream, now)
            if task is None:
                break
            policy.validate_url(task.url)
            try:
                request_at = _wait_for_slot(
                    connection,
                    _authority(task.url),
                    policy.min_interval_seconds,
                    now_fn=now_fn,
                    sleep_fn=sleep_fn,
                )
                policy.assert_authorization_active(request_at)
            except (Bo3HostCircuitOpenError, AuthorizationWindowError) as error:
                stopped_reason = str(error)
                break
            attempts = _start_task(connection, task, request_at)
            request_counted = False
            try:
                response = _request_json(
                    request_opener, task.url, policy, timeout_seconds
                )
                requests += 1
                request_counted = True
                status_code = int(response.getcode())
                final_url = str(response.geturl())
                policy.validate_url(final_url)
                headers = _safe_headers(response.headers)
                if status_code != 200:
                    raise HTTPError(
                        final_url, status_code, "unexpected status", response.headers, None
                    )
                body = _read_limited(response, policy.max_response_bytes)
                complete, _ = _quality_finish(
                    connection,
                    task,
                    attempts=attempts,
                    policy=policy,
                    observed_at=request_at,
                    source_url=task.url,
                    final_url=final_url,
                    status_code=status_code,
                    headers=headers,
                    body=body,
                    output_dir=output_path,
                )
                if not complete:
                    stopped_reason = "quality_retry"
                    break
            except HTTPError as error:
                if not request_counted:
                    requests += 1
                status = _finish_http_error(
                    connection,
                    task,
                    attempts=attempts,
                    status_code=int(error.code),
                    error=f"HTTP {error.code}: {error.reason}",
                    headers=error.headers,
                    policy=policy,
                    now=now_fn().astimezone(timezone.utc),
                )
                if status == "retry" or int(error.code) in PERMANENT_BLOCK_CODES:
                    stopped_reason = f"http_{error.code}"
                    break
            except (URLError, TimeoutError, OSError) as error:
                requests += 1
                status = _finish_http_error(
                    connection,
                    task,
                    attempts=attempts,
                    status_code=None,
                    error=f"{type(error).__name__}: {error}",
                    headers={},
                    policy=policy,
                    now=now_fn().astimezone(timezone.utc),
                )
                if status == "retry":
                    stopped_reason = "network_retry"
                    break
            except (Bo3QualityError, ResponseTooLargeError, ResponseLengthMismatchError) as error:
                # These exceptions occur before a durable snapshot can be committed.
                status = _finish_http_error(
                    connection,
                    task,
                    attempts=attempts,
                    status_code=None,
                    error=f"{type(error).__name__}: {error}",
                    headers={},
                    policy=policy,
                    now=now_fn().astimezone(timezone.utc),
                )
                if status == "retry":
                    stopped_reason = "quality_retry"
                    break
            except Bo3StorageError as error:
                with connection:
                    connection.execute(
                        """
                        UPDATE bo3_task
                        SET status = 'quarantined', last_error = ?, updated_at = ?
                        WHERE stream = ? AND task_key = ?
                        """,
                        (
                            f"{type(error).__name__}: {error}",
                            _iso(now_fn()),
                            task.stream,
                            task.task_key,
                        ),
                    )
                stopped_reason = "storage_error"
                break
        result = audit_bo3_capture(state_path, stream=stream)
        result.update(
            {
                "requests_this_run": requests,
                "stopped_reason": stopped_reason,
                "stream": stream,
            }
        )
        return result
    finally:
        if connection is not None:
            connection.close()
        _release_lock(lock)


def audit_bo3_capture(
    state_db: str | Path, *, stream: str, max_samples: int = 20
) -> dict[str, object]:
    connection = _open_existing_state(Path(state_db).resolve())
    try:
        job = connection.execute(
            "SELECT * FROM bo3_capture_job WHERE stream = ?", (stream,)
        ).fetchone()
        if job is None:
            raise ValueError(f"unknown BO3 stream: {stream}")
        task_counts = {
            f"{row['kind']}:{row['status']}": int(row["count"])
            for row in connection.execute(
                """
                SELECT kind, status, COUNT(*) AS count
                FROM bo3_task WHERE stream = ?
                GROUP BY kind, status ORDER BY kind, status
                """,
                (stream,),
            )
        }
        window_rows = connection.execute(
            """
            SELECT w.window_start, w.window_end, w.expected_count, w.expected_pages,
                   (
                       SELECT COUNT(*) FROM bo3_task AS t
                       WHERE t.stream = w.stream AND t.kind = 'catalog'
                         AND t.status = 'complete'
                         AND json_extract(t.metadata_json, '$.window_start') = w.window_start
                         AND json_extract(t.metadata_json, '$.window_end') = w.window_end
                   ) AS complete_pages,
                   (
                       SELECT COUNT(*) FROM bo3_match_index AS m
                       WHERE m.stream = w.stream
                         AND m.window_start = w.window_start
                         AND m.window_end = w.window_end
                   ) AS discovered_matches
            FROM bo3_window AS w
            WHERE w.stream = ?
            ORDER BY w.window_start
            """,
            (stream,),
        ).fetchall()
        incomplete_windows = [
            row
            for row in window_rows
            if row["expected_count"] is None
            or int(row["complete_pages"]) != int(row["expected_pages"] or 0)
            or int(row["discovered_matches"]) != int(row["expected_count"] or 0)
        ][:max_samples]
        incomplete_games = connection.execute(
            """
            SELECT g.game_id, g.match_id, g.map_name, g.rounds_count,
                   g.game_detail_complete, g.player_rows, g.distinct_players,
                   g.distinct_teams, g.players_complete, g.player_quality_error
            FROM bo3_game_index AS g
            JOIN bo3_match_index AS m
              ON m.stream = g.stream AND m.match_id = g.match_id
            WHERE g.stream = ?
              AND g.stats_expected = 1
              AND (
                  g.map_name IS NULL
                  OR g.game_detail_complete = 0 OR g.players_complete = 0
                  OR g.player_rows <> 10 OR g.distinct_players <> 10
                  OR g.distinct_teams <> 2
              )
            ORDER BY m.start_date, g.match_id, g.map_number
            LIMIT ?
            """,
            (stream, max_samples),
        ).fetchall()
        identity_conflicts = connection.execute(
            """
            SELECT steam_profile_id, COUNT(DISTINCT steam_id_64) AS steam_ids
            FROM bo3_player_map_index
            WHERE stream = ? AND steam_id_64 IS NOT NULL
            GROUP BY steam_profile_id
            HAVING COUNT(DISTINCT steam_id_64) > 1
            LIMIT ?
            """,
            (stream, max_samples),
        ).fetchall()
        reverse_identity_conflicts = connection.execute(
            """
            SELECT steam_id_64, COUNT(DISTINCT steam_profile_id) AS profile_ids
            FROM bo3_player_map_index
            WHERE stream = ? AND steam_id_64 IS NOT NULL
            GROUP BY steam_id_64
            HAVING COUNT(DISTINCT steam_profile_id) > 1
            LIMIT ?
            """,
            (stream, max_samples),
        ).fetchall()
        totals = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM bo3_window WHERE stream = ?) AS windows,
                (SELECT COUNT(*) FROM bo3_match_index WHERE stream = ?) AS matches,
                (SELECT COUNT(*) FROM bo3_game_index WHERE stream = ?) AS games,
                (SELECT COUNT(*) FROM bo3_player_map_index WHERE stream = ?) AS player_maps,
                (SELECT COUNT(DISTINCT steam_profile_id)
                   FROM bo3_player_map_index WHERE stream = ?) AS players,
                (SELECT COUNT(*) FROM bo3_task
                   WHERE stream = ? AND status NOT IN ('complete')) AS noncomplete_tasks
            """,
            (stream, stream, stream, stream, stream, stream),
        ).fetchone()
        gaps = {
            "incomplete_window_count": sum(
                1
                for row in window_rows
                if row["expected_count"] is None
                or int(row["complete_pages"]) != int(row["expected_pages"] or 0)
                or int(row["discovered_matches"]) != int(row["expected_count"] or 0)
            ),
            "finished_game_player_gap_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM bo3_game_index AS g
                    JOIN bo3_match_index AS m
                      ON m.stream = g.stream AND m.match_id = g.match_id
                    WHERE g.stream = ?
                      AND g.stats_expected = 1
                      AND (
                          g.map_name IS NULL
                          OR g.game_detail_complete = 0 OR g.players_complete = 0
                          OR g.player_rows <> 10 OR g.distinct_players <> 10
                          OR g.distinct_teams <> 2
                      )
                    """,
                    (stream,),
                ).fetchone()[0]
            ),
            "steam_identity_conflict_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT steam_profile_id
                        FROM bo3_player_map_index
                        WHERE stream = ? AND steam_id_64 IS NOT NULL
                        GROUP BY steam_profile_id
                        HAVING COUNT(DISTINCT steam_id_64) > 1
                    )
                    """,
                    (stream,),
                ).fetchone()[0]
            ),
            "steam_id_reverse_conflict_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT steam_id_64
                        FROM bo3_player_map_index
                        WHERE stream = ? AND steam_id_64 IS NOT NULL
                        GROUP BY steam_id_64
                        HAVING COUNT(DISTINCT steam_profile_id) > 1
                    )
                    """,
                    (stream,),
                ).fetchone()[0]
            ),
        }
        ok = (
            int(totals["noncomplete_tasks"]) == 0
            and all(value == 0 for value in gaps.values())
        )
        return {
            "capture_version": str(job["capture_version"]),
            "coverage": {
                "end_date_exclusive": str(job["end_date"]),
                "start_date": str(job["start_date"]),
                "statuses": str(job["statuses"]).split(","),
            },
            "gaps": gaps,
            "identity_conflict_samples": {
                "profile_to_steam": [dict(row) for row in identity_conflicts],
                "steam_to_profile": [dict(row) for row in reverse_identity_conflicts],
            },
            "incomplete_game_samples": [dict(row) for row in incomplete_games],
            "incomplete_window_samples": [dict(row) for row in incomplete_windows],
            "ok": ok,
            "profile": str(job["profile"]),
            "task_counts": task_counts,
            "totals": {key: int(totals[key]) for key in totals.keys()},
        }
    finally:
        connection.close()


def bo3_capture_index(
    state_db: str | Path, *, stream: str
) -> Iterable[dict[str, object]]:
    connection = _open_existing_state(Path(state_db).resolve())
    try:
        rows = connection.execute(
            """
            SELECT s.*, t.kind, t.source_id, t.metadata_json
            FROM bo3_snapshot AS s
            JOIN bo3_task AS t
              ON t.stream = s.stream AND t.task_key = s.task_key
            WHERE s.stream = ?
            ORDER BY s.observed_at, s.task_key
            """,
            (stream,),
        )
        for row in rows:
            yield {
                "content_sha256": row["content_sha256"],
                "kind": row["kind"],
                "metadata": json.loads(row["metadata_json"]),
                "object_path": row["object_path"],
                "observed_at": row["observed_at"],
                "quality_error": row["quality_error"],
                "quality_status": row["quality_status"],
                "snapshot_id": row["snapshot_id"],
                "source_id": row["source_id"],
                "source_url": row["source_url"],
            }
    finally:
        connection.close()
