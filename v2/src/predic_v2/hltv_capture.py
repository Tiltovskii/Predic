from __future__ import annotations

import hashlib
from http.client import HTTPException
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

import fcntl


CAPTURE_VERSION = "hltv-authorized-capture-v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
_PERMANENT_HOST_BLOCK_CODES = {401, 403, 406, 418, 451}
_TERMINAL_STATUSES = {
    "complete",
    "not_found",
    "http_error",
    "invalid_content",
    "retry_exhausted",
}


class CapturePolicyError(ValueError):
    """Raised when a capture policy is absent, unsafe, or out of scope."""


class CaptureSourceChangedError(ValueError):
    """Raised when an existing stream is resumed with different inputs."""


class ResponseTooLargeError(ValueError):
    """Raised before an oversized response can be persisted."""


class CaptureCorruptionError(ValueError):
    """Raised when a raw object no longer matches its capture manifest."""


class CaptureIncompleteError(ValueError):
    """Raised when an export would silently omit unfinished or failed pages."""


class CaptureQualityError(ValueError):
    """Raised when HTTP succeeded but the document lacks the requested entity."""


class CaptureBusyError(RuntimeError):
    """Raised when another process already owns the capture state lock."""


class AuthorizationWindowError(CapturePolicyError):
    """Raised when a request would occur outside the written permission window."""


class HostCircuitOpenError(RuntimeError):
    """Raised before network use when a host-level stop is still active."""

    def __init__(self, reason: str, not_before_at: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.not_before_at = not_before_at


class HttpRequestBudgetError(RuntimeError):
    """Raised before a request that would exceed the run's HTTP budget."""


class ArtifactStorageError(RuntimeError):
    """Raised when a successful response cannot be durably stored locally."""


class ResponseLengthMismatchError(HTTPException):
    """Raised when a response ends before its declared Content-Length."""


@dataclass(frozen=True)
class CapturePolicy:
    live_enabled: bool
    authorization_ref: str
    authorization_scope: str
    authorization_confirmed_at: str | None
    valid_until: str | None
    allowed_schemes: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]
    allowed_query_keys: tuple[str, ...]
    user_agent: str
    contact: str
    min_interval_seconds: float
    max_pages_per_run: int
    max_http_requests_per_run: int
    max_response_bytes: int
    max_attempts_per_url: int
    base_backoff_seconds: float
    robots_txt_mode: str

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any], *, require_live: bool = True
    ) -> "CapturePolicy":
        required = {
            "live_enabled",
            "authorization_ref",
            "authorization_scope",
            "authorization_confirmed_at",
            "valid_until",
            "allowed_schemes",
            "allowed_hosts",
            "allowed_path_prefixes",
            "allowed_query_keys",
            "user_agent",
            "contact",
            "min_interval_seconds",
            "max_pages_per_run",
            "max_http_requests_per_run",
            "max_response_bytes",
            "max_attempts_per_url",
            "base_backoff_seconds",
            "robots_txt_mode",
        }
        missing = sorted(required - raw.keys())
        unknown = sorted(raw.keys() - required)
        if missing:
            raise CapturePolicyError(f"capture policy is missing: {', '.join(missing)}")
        if unknown:
            raise CapturePolicyError(
                f"capture policy has unknown fields: {', '.join(unknown)}"
            )

        def text(field: str, *, optional: bool = False) -> str | None:
            value = raw[field]
            if value is None and optional:
                return None
            if not isinstance(value, str) or not value.strip():
                raise CapturePolicyError(f"{field} must be a non-empty string")
            return value.strip()

        def text_tuple(field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
            value = raw[field]
            if (
                not isinstance(value, list)
                or (not value and not allow_empty)
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                qualifier = "an array" if allow_empty else "a non-empty array"
                raise CapturePolicyError(
                    f"{field} must be {qualifier} of non-empty strings"
                )
            return tuple(item.strip() for item in value)

        live_enabled = raw["live_enabled"]
        if not isinstance(live_enabled, bool):
            raise CapturePolicyError("live_enabled must be a boolean")
        if require_live and not live_enabled:
            raise CapturePolicyError(
                "live capture is disabled by policy; verify the written authorization "
                "and set live_enabled to true"
            )

        authorization_ref = text("authorization_ref")
        authorization_scope = text("authorization_scope")
        confirmed_at = text("authorization_confirmed_at", optional=True)
        valid_until = text("valid_until", optional=True)
        if require_live and confirmed_at is None:
            raise CapturePolicyError(
                "authorization_confirmed_at is required for live capture"
            )
        if confirmed_at is not None:
            _parse_iso_datetime(confirmed_at, "authorization_confirmed_at")
        if valid_until is not None:
            _parse_iso_datetime(valid_until, "valid_until")

        allowed_schemes = tuple(item.lower() for item in text_tuple("allowed_schemes"))
        if set(allowed_schemes) != {"https"}:
            raise CapturePolicyError("authorized capture requires HTTPS-only URLs")
        allowed_hosts = tuple(item.lower() for item in text_tuple("allowed_hosts"))
        if any("/" in item or "@" in item for item in allowed_hosts):
            raise CapturePolicyError("allowed_hosts must contain host[:port] values only")
        allowed_paths = text_tuple("allowed_path_prefixes")
        if any(not item.startswith("/") for item in allowed_paths):
            raise CapturePolicyError("every allowed path prefix must start with /")
        allowed_query_keys = text_tuple("allowed_query_keys", allow_empty=True)

        min_interval = _positive_number(
            raw, "min_interval_seconds", allow_zero=False, maximum=86_400
        )
        max_pages = _positive_integer(raw, "max_pages_per_run", maximum=100_000)
        max_http_requests = _positive_integer(
            raw, "max_http_requests_per_run", maximum=100_000
        )
        max_bytes = _positive_integer(
            raw, "max_response_bytes", maximum=100 * 1024 * 1024
        )
        max_attempts = _positive_integer(raw, "max_attempts_per_url", maximum=10)
        base_backoff = _positive_number(
            raw, "base_backoff_seconds", allow_zero=False, maximum=86_400
        )
        robots_mode = text("robots_txt_mode")
        if robots_mode not in {"respect", "written_permission_override"}:
            raise CapturePolicyError(
                "robots_txt_mode must be respect or written_permission_override"
            )

        user_agent = str(text("user_agent"))
        contact = str(text("contact"))
        if contact.casefold() not in user_agent.casefold():
            raise CapturePolicyError(
                "user_agent must include the configured contact identifier"
            )

        return cls(
            live_enabled=live_enabled,
            authorization_ref=str(authorization_ref),
            authorization_scope=str(authorization_scope),
            authorization_confirmed_at=confirmed_at,
            valid_until=valid_until,
            allowed_schemes=allowed_schemes,
            allowed_hosts=allowed_hosts,
            allowed_path_prefixes=allowed_paths,
            allowed_query_keys=allowed_query_keys,
            user_agent=user_agent,
            contact=contact,
            min_interval_seconds=min_interval,
            max_pages_per_run=max_pages,
            max_http_requests_per_run=max_http_requests,
            max_response_bytes=max_bytes,
            max_attempts_per_url=max_attempts,
            base_backoff_seconds=base_backoff,
            robots_txt_mode=str(robots_mode),
        )

    def assert_authorization_active(self, now: datetime) -> None:
        checked_at = now.astimezone(timezone.utc)
        if self.authorization_confirmed_at is None:
            raise AuthorizationWindowError("written authorization is not confirmed")
        starts_at = _parse_iso_datetime(
            self.authorization_confirmed_at, "authorization_confirmed_at"
        )
        if checked_at < starts_at:
            raise AuthorizationWindowError("written authorization is not active yet")
        if self.valid_until is not None and checked_at >= _parse_iso_datetime(
            self.valid_until, "valid_until"
        ):
            raise AuthorizationWindowError("written authorization has expired")

    def validate_url(self, url: str) -> None:
        if any(ord(character) < 33 for character in url):
            raise CapturePolicyError(f"URL contains whitespace or control characters: {url!r}")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as error:
            raise CapturePolicyError(f"invalid URL {url!r}: {error}") from error
        if parsed.scheme.lower() not in self.allowed_schemes:
            raise CapturePolicyError(f"URL scheme is outside the policy: {url}")
        if parsed.username is not None or parsed.password is not None:
            raise CapturePolicyError(f"URL credentials are forbidden: {url}")
        if parsed.fragment:
            raise CapturePolicyError(f"URL fragments are forbidden: {url}")
        if parsed.hostname is None:
            raise CapturePolicyError(f"URL has no host: {url}")
        authority = parsed.hostname.lower()
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        if port is not None and port != default_port:
            authority = f"{authority}:{port}"
        if authority not in self.allowed_hosts:
            raise CapturePolicyError(f"URL host is outside the policy: {url}")

        raw_path = parsed.path or "/"
        lowered = raw_path.lower()
        if any(token in lowered for token in ("%2f", "%5c", "%00")):
            raise CapturePolicyError(f"encoded path separators are forbidden: {url}")
        decoded_path = unquote(raw_path)
        if "\\" in decoded_path or any(
            segment in {".", ".."} for segment in decoded_path.split("/")
        ):
            raise CapturePolicyError(f"ambiguous URL path is forbidden: {url}")
        if not any(_path_is_in_scope(decoded_path, prefix) for prefix in self.allowed_path_prefixes):
            raise CapturePolicyError(f"URL path is outside the policy: {url}")
        try:
            query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        except ValueError as error:
            raise CapturePolicyError(f"malformed URL query: {url}") from error
        unknown_query_keys = sorted(
            {key for key, _ in query if key not in self.allowed_query_keys}
        )
        if unknown_query_keys:
            raise CapturePolicyError(
                "URL query keys are outside the policy: "
                + ", ".join(unknown_query_keys)
            )


@dataclass(frozen=True)
class ManifestEntry:
    ordinal: int
    record_id: str
    url: str
    page_type: str
    metadata: dict[str, object]


def _path_is_in_scope(path: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


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


_CAPTURE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS capture_job (
    job_id TEXT PRIMARY KEY,
    stream TEXT NOT NULL UNIQUE,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    policy_path TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    authorization_ref TEXT NOT NULL,
    authorization_scope TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    capture_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capture_entry (
    job_id TEXT NOT NULL REFERENCES capture_job(job_id),
    ordinal INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    url TEXT NOT NULL,
    page_type TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_eligible_at TEXT,
    last_status_code INTEGER,
    last_error TEXT,
    capture_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, record_id),
    UNIQUE (job_id, ordinal)
);

CREATE TABLE IF NOT EXISTS capture_attempt (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    outcome TEXT NOT NULL,
    status_code INTEGER,
    final_url TEXT,
    retry_after_at TEXT,
    response_headers_json TEXT,
    redirect_chain_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    UNIQUE (job_id, record_id, attempt_number),
    FOREIGN KEY (job_id, record_id) REFERENCES capture_entry(job_id, record_id)
);

CREATE TABLE IF NOT EXISTS capture_artifact (
    capture_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    content_type TEXT,
    etag TEXT,
    last_modified TEXT,
    content_sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    object_path TEXT NOT NULL,
    response_headers_json TEXT NOT NULL,
    redirect_chain_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (job_id, record_id) REFERENCES capture_entry(job_id, record_id)
);

CREATE TABLE IF NOT EXISTS capture_manifest_metadata (
    job_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (job_id, record_id),
    FOREIGN KEY (job_id, record_id) REFERENCES capture_entry(job_id, record_id)
);

CREATE TABLE IF NOT EXISTS capture_host_state (
    authority TEXT PRIMARY KEY,
    last_request_at TEXT NOT NULL,
    blocked_reason TEXT,
    not_before_at TEXT
);

CREATE TABLE IF NOT EXISTS capture_host_review (
    review_id TEXT PRIMARY KEY,
    authority TEXT NOT NULL,
    previous_blocked_reason TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    authorization_ref TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS robots_snapshot (
    authority TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    body TEXT,
    response_headers_json TEXT NOT NULL
);
"""


def _positive_number(
    raw: dict[str, Any], field: str, *, allow_zero: bool, maximum: float | None = None
) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapturePolicyError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CapturePolicyError(f"{field} must be finite")
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CapturePolicyError(f"{field} must be {qualifier}")
    if maximum is not None and number > maximum:
        raise CapturePolicyError(f"{field} must not exceed {maximum:g}")
    return number


def _positive_integer(
    raw: dict[str, Any], field: str, *, maximum: int | None = None
) -> int:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CapturePolicyError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise CapturePolicyError(f"{field} must not exceed {maximum}")
    return value


def _parse_iso_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CapturePolicyError(f"{field} must be an ISO-8601 datetime") from error
    if parsed.tzinfo is None:
        raise CapturePolicyError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_policy(path: str | Path, *, require_live: bool = True) -> tuple[CapturePolicy, str]:
    policy_path = Path(path).resolve()
    try:
        content = policy_path.read_bytes()
        raw = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapturePolicyError(f"cannot read capture policy: {error}") from error
    if not isinstance(raw, dict):
        raise CapturePolicyError("capture policy must be a JSON object")
    return (
        CapturePolicy.from_mapping(raw, require_live=require_live),
        hashlib.sha256(content).hexdigest(),
    )


def _load_manifest(path: Path, policy: CapturePolicy) -> tuple[list[ManifestEntry], str]:
    entries: list[ManifestEntry] = []
    seen: dict[str, tuple[str, str]] = {}
    seen_urls: dict[tuple[str, str, str, str], str] = {}
    seen_entities: dict[tuple[str, str], str] = {}
    content = path.read_bytes()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest JSON on line {line_number}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"manifest line {line_number} needs a non-empty url")
        policy.validate_url(url)
        page_type = raw.get("page_type", "auto")
        if not isinstance(page_type, str) or page_type.strip() not in {
            "auto",
            "match",
            "map-stats",
            "results",
        }:
            raise ValueError(
                f"manifest line {line_number} page_type must be auto, match, map-stats, or results"
            )
        record_id = raw.get("record_id", _digest(url)[:40])
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"manifest line {line_number} record_id must be a non-empty string"
            )
        record_id = record_id.strip()
        signature = (url, page_type.strip())
        if record_id in seen:
            raise ValueError(
                f"manifest record_id {record_id!r} is duplicated; every requested "
                "page must have one stable unique ID"
            )
        seen[record_id] = signature
        canonical_url = _canonical_url_key(url)
        if canonical_url in seen_urls:
            raise ValueError(
                f"manifest URL {url!r} duplicates record_id "
                f"{seen_urls[canonical_url]!r}; one page may be fetched only once"
            )
        seen_urls[canonical_url] = record_id
        entity_key = _source_entity_key(url, page_type.strip())
        if entity_key is not None and entity_key in seen_entities:
            raise ValueError(
                f"manifest URL {url!r} duplicates source entity {entity_key[0]} "
                f"{entity_key[1]} from record_id {seen_entities[entity_key]!r}"
            )
        if entity_key is not None:
            seen_entities[entity_key] = record_id
        metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"record_id", "url", "page_type"}
        }
        entries.append(
            ManifestEntry(
                ordinal=len(entries),
                record_id=record_id,
                url=url,
                page_type=page_type.strip(),
                metadata=metadata,
            )
        )
    if not entries:
        raise ValueError("capture manifest has no records")
    return entries, hashlib.sha256(content).hexdigest()


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.executescript(_CAPTURE_SCHEMA)
    connection.commit()
    return connection


def _open_existing_state(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"capture state database does not exist: {path}")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # Forward-compatible state migration.  In particular, older capture DBs
    # predate capture_manifest_metadata; never make an existing raw archive
    # unreadable merely because we added provenance preservation.
    connection.executescript(_CAPTURE_SCHEMA)
    connection.commit()
    return connection


def _initialize_job(
    connection: sqlite3.Connection,
    *,
    stream: str,
    manifest_path: Path,
    manifest_sha256: str,
    policy_path: Path,
    policy_sha256: str,
    authorization_ref: str,
    authorization_scope: str,
    output_dir: Path,
    entries: Iterable[ManifestEntry],
    now: datetime,
) -> str:
    entries = tuple(entries)
    job_id = f"capture:{_digest(stream, manifest_sha256, policy_sha256)[:40]}"
    existing = connection.execute(
        "SELECT * FROM capture_job WHERE stream = ?", (stream,)
    ).fetchone()
    if existing is not None:
        expected = {
            "manifest_sha256": manifest_sha256,
            "policy_sha256": policy_sha256,
            "output_dir": str(output_dir),
            "capture_version": CAPTURE_VERSION,
        }
        changed = [field for field, value in expected.items() if existing[field] != value]
        if changed:
            raise CaptureSourceChangedError(
                "capture stream inputs changed ("
                + ", ".join(changed)
                + "); use a new stream name for a new immutable revision"
            )
        if existing["job_id"] != job_id:
            raise CaptureSourceChangedError("capture stream resolves to a different job")
        with connection:
            connection.execute(
                """
                UPDATE capture_attempt
                SET completed_at = ?, outcome = 'abandoned_after_crash',
                    error = 'process stopped before the attempt outcome was committed'
                WHERE job_id = ? AND outcome = 'in_progress'
                """,
                (_iso(now), job_id),
            )
            connection.execute(
                """
                UPDATE capture_entry
                SET status = 'pending',
                    last_error = 'previous process stopped after reserving a request',
                    updated_at = ?
                WHERE job_id = ? AND status = 'in_progress'
                """,
                (_iso(now), job_id),
            )
        return job_id

    stamp = _iso(now)
    with connection:
        connection.execute(
            """
            INSERT INTO capture_job (
                job_id, stream, manifest_path, manifest_sha256,
                policy_path, policy_sha256, authorization_ref,
                authorization_scope, output_dir, capture_version,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                stream,
                str(manifest_path),
                manifest_sha256,
                str(policy_path),
                policy_sha256,
                authorization_ref,
                authorization_scope,
                str(output_dir),
                CAPTURE_VERSION,
                stamp,
                stamp,
            ),
        )
        connection.executemany(
            """
            INSERT INTO capture_entry (
                job_id, ordinal, record_id, url, page_type, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                (job_id, entry.ordinal, entry.record_id, entry.url, entry.page_type, stamp)
                for entry in entries
            ),
        )
        connection.executemany(
            """
            INSERT INTO capture_manifest_metadata (job_id, record_id, metadata_json)
            VALUES (?, ?, ?)
            """,
            (
                (
                    job_id,
                    entry.record_id,
                    json.dumps(
                        entry.metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
                for entry in entries
            ),
        )
    return job_id


def _authority(url: str) -> str:
    parsed = urlsplit(url)
    authority = str(parsed.hostname).lower()
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    if parsed.port is not None and parsed.port != default_port:
        authority = f"{authority}:{parsed.port}"
    return authority


def _canonical_url_key(url: str) -> tuple[str, str, str, str]:
    parsed = urlsplit(url)
    return (
        parsed.scheme.lower(),
        _authority(url),
        parsed.path or "/",
        parsed.query,
    )


def _source_entity_key(url: str, page_type: str) -> tuple[str, str] | None:
    match = re.search(r"/matches/(\d+)(?:/|$)", url)
    map_stats = re.search(r"/mapstatsid/(\d+)(?:/|$)", url)
    if page_type == "match" and match:
        return "match", match.group(1)
    if page_type == "map-stats" and map_stats:
        return "map-stats", map_stats.group(1)
    if page_type == "auto":
        if match:
            return "match", match.group(1)
        if map_stats:
            return "map-stats", map_stats.group(1)
    return None


def _wait_for_request_slot(
    connection: sqlite3.Connection,
    authority: str,
    interval_seconds: float,
    *,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[float], None],
) -> datetime:
    while True:
        now = now_fn().astimezone(timezone.utc)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT last_request_at, blocked_reason, not_before_at
            FROM capture_host_state WHERE authority = ?
            """,
            (authority,),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO capture_host_state (authority, last_request_at) VALUES (?, ?)",
                (authority, _iso(now)),
            )
            connection.commit()
            return now
        last = _parse_iso_datetime(row["last_request_at"], "last_request_at")
        blocked_reason = row["blocked_reason"]
        not_before_at = row["not_before_at"]
        if blocked_reason in {
            f"http_{code}" for code in _PERMANENT_HOST_BLOCK_CODES
        }:
            connection.rollback()
            raise HostCircuitOpenError(str(blocked_reason), not_before_at)
        if blocked_reason is not None and not_before_at is not None:
            not_before = _parse_iso_datetime(not_before_at, "not_before_at")
            if now < not_before:
                connection.rollback()
                raise HostCircuitOpenError(str(blocked_reason), not_before_at)
            connection.execute(
                """
                UPDATE capture_host_state
                SET blocked_reason = NULL, not_before_at = NULL
                WHERE authority = ?
                """,
                (authority,),
            )
        remaining = interval_seconds - (now - last).total_seconds()
        if remaining <= 0:
            connection.execute(
                "UPDATE capture_host_state SET last_request_at = ? WHERE authority = ?",
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
    result: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in keep:
            result[lowered] = str(value)
    return result


def _retry_after_at(headers: Any, now: datetime, fallback_seconds: float) -> datetime:
    local_backoff = now + timedelta(seconds=fallback_seconds)
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is not None:
        try:
            seconds = max(0, int(str(raw).strip()))
            try:
                server_backoff = now + timedelta(seconds=seconds)
            except OverflowError:
                server_backoff = datetime.max.replace(tzinfo=timezone.utc)
            return max(local_backoff, server_backoff)
        except (ValueError, OverflowError):
            try:
                parsed = parsedate_to_datetime(str(raw))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(local_backoff, parsed.astimezone(timezone.utc))
            except (TypeError, ValueError, OverflowError):
                pass
    return local_backoff


def _acquire_run_lock(state_path: Path) -> BinaryIO:
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise CaptureBusyError(
            f"another capture process is already using {state_path}"
        ) from error
    return handle


def _release_run_lock(handle: BinaryIO) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _build_opener(policy: CapturePolicy) -> OpenerDirector:
    return build_opener(_NoRedirectHandler())


def _request(opener: Any, url: str, policy: CapturePolicy, timeout: float) -> Any:
    request = Request(
        url,
        method="GET",
        headers={
            "User-Agent": policy.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.1",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
    )
    return opener.open(request, timeout=timeout)


def _request_chain(
    connection: sqlite3.Connection,
    opener: Any,
    url: str,
    policy: CapturePolicy,
    timeout: float,
    *,
    first_requested_at: datetime,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[float], None],
    request_budget: int,
    max_redirects: int = 5,
) -> tuple[int, str, Any, bytes, list[dict[str, object]], int]:
    """Perform a manually scoped redirect chain and rate-limit every hop."""

    current_url = url
    seen_urls = {url}
    redirects: list[dict[str, object]] = []
    request_count = 0
    requested_at = first_requested_at
    while True:
        if request_count >= request_budget:
            failure = HttpRequestBudgetError("HTTP request budget exhausted")
            setattr(failure, "capture_redirect_chain", redirects)
            setattr(failure, "capture_request_count", request_count)
            raise failure
        try:
            policy.assert_authorization_active(now_fn())
        except AuthorizationWindowError as error:
            setattr(error, "capture_redirect_chain", redirects)
            setattr(error, "capture_request_count", request_count)
            raise
        request_count += 1
        try:
            response = _request(opener, current_url, policy, timeout)
        except HTTPError as error:
            status_code = int(error.code)
            if status_code not in {301, 302, 303, 307, 308}:
                setattr(error, "capture_redirect_chain", redirects)
                setattr(error, "capture_request_count", request_count)
                raise
            headers = error.headers
            location = headers.get("Location")
            error.close()
            if not location:
                failure = CapturePolicyError("redirect response has no Location header")
                setattr(failure, "capture_redirect_chain", redirects)
                setattr(failure, "capture_request_count", request_count)
                raise failure
        except BaseException as error:
            setattr(error, "capture_redirect_chain", redirects)
            setattr(error, "capture_request_count", request_count)
            raise
        else:
            status_code = int(response.getcode())
            headers = response.headers
            if status_code not in {301, 302, 303, 307, 308}:
                final_url = str(response.geturl())
                try:
                    policy.validate_url(final_url)
                    body = _read_limited(response, policy.max_response_bytes)
                except BaseException as error:
                    setattr(error, "capture_redirect_chain", redirects)
                    setattr(error, "capture_request_count", request_count)
                    raise
                finally:
                    response.close()
                return status_code, final_url, headers, body, redirects, request_count
            location = headers.get("Location")
            response.close()
            if not location:
                failure = CapturePolicyError("redirect response has no Location header")
                setattr(failure, "capture_redirect_chain", redirects)
                setattr(failure, "capture_request_count", request_count)
                raise failure

        target_url = urljoin(current_url, str(location))
        try:
            policy.validate_url(target_url)
        except CapturePolicyError as error:
            redirects.append(
                {
                    "from_url": current_url,
                    "status_code": status_code,
                    "location": str(location),
                    "target_url": target_url,
                    "requested_at": _iso(requested_at),
                    "allowed": False,
                }
            )
            setattr(error, "capture_redirect_chain", redirects)
            setattr(error, "capture_request_count", request_count)
            raise
        redirects.append(
            {
                "from_url": current_url,
                "status_code": status_code,
                "location": str(location),
                "target_url": target_url,
                "requested_at": _iso(requested_at),
                "allowed": True,
            }
        )
        if len(redirects) > max_redirects:
            failure = CapturePolicyError("redirect limit exceeded")
            setattr(failure, "capture_redirect_chain", redirects)
            setattr(failure, "capture_request_count", request_count)
            raise failure
        if target_url in seen_urls:
            failure = CapturePolicyError("redirect loop detected")
            setattr(failure, "capture_redirect_chain", redirects)
            setattr(failure, "capture_request_count", request_count)
            raise failure
        seen_urls.add(target_url)
        current_url = target_url
        if request_count >= request_budget:
            failure = HttpRequestBudgetError("HTTP request budget exhausted")
            setattr(failure, "capture_redirect_chain", redirects)
            setattr(failure, "capture_request_count", request_count)
            raise failure
        try:
            requested_at = _wait_for_request_slot(
                connection,
                _authority(current_url),
                policy.min_interval_seconds,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
            )
        except BaseException as error:
            setattr(error, "capture_redirect_chain", redirects)
            setattr(error, "capture_request_count", request_count)
            raise


def _read_limited(response: Any, limit: int) -> bytes:
    declared = response.headers.get("Content-Length")
    declared_size: int | None = None
    if declared is not None:
        try:
            declared_size = int(declared)
            if declared_size > limit:
                raise ResponseTooLargeError(
                    f"declared response size {declared} exceeds limit {limit}"
                )
        except ValueError:
            declared_size = None
    chunks: list[bytes] = []
    size = 0
    while True:
        block = response.read(min(64 * 1024, limit - size + 1))
        if not block:
            break
        size += len(block)
        if size > limit:
            raise ResponseTooLargeError(f"response exceeds byte limit {limit}")
        chunks.append(block)
    body = b"".join(chunks)
    if declared_size is not None and declared_size >= 0 and size != declared_size:
        raise ResponseLengthMismatchError(
            f"response ended at {size} bytes but declared {declared_size}"
        )
    return body


def _store_artifact(output_dir: Path, body: bytes) -> tuple[str, Path]:
    content_sha256 = hashlib.sha256(body).hexdigest()
    relative = Path("objects") / content_sha256[:2] / f"{content_sha256}.html"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256_file(target) != content_sha256:
            raise IOError(f"content-addressed artifact is corrupted: {target}")
        return content_sha256, relative
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
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
    return content_sha256, relative


def _start_attempt(
    connection: sqlite3.Connection,
    job_id: str,
    record_id: str,
    started_at: datetime,
) -> tuple[int, str]:
    with connection:
        row = connection.execute(
            """
            SELECT attempts FROM capture_entry
            WHERE job_id = ? AND record_id = ?
            """,
            (job_id, record_id),
        ).fetchone()
        attempt_number = int(row["attempts"]) + 1
        attempt_id = f"attempt:{_digest(job_id, record_id, attempt_number)[:40]}"
        connection.execute(
            """
            UPDATE capture_entry
            SET status = 'in_progress', attempts = ?, updated_at = ?
            WHERE job_id = ? AND record_id = ?
            """,
            (attempt_number, _iso(started_at), job_id, record_id),
        )
        connection.execute(
            """
            INSERT INTO capture_attempt (
                attempt_id, job_id, record_id, attempt_number,
                started_at, outcome
            ) VALUES (?, ?, ?, ?, ?, 'in_progress')
            """,
            (attempt_id, job_id, record_id, attempt_number, _iso(started_at)),
        )
    return attempt_number, attempt_id


def _finish_failure(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    job_id: str,
    record_id: str,
    completed_at: datetime,
    outcome: str,
    entry_status: str,
    status_code: int | None,
    final_url: str | None,
    retry_after_at: datetime | None,
    headers: Any | None,
    redirect_chain: list[dict[str, object]],
    error: str,
    circuit_authority: str | None = None,
    circuit_reason: str | None = None,
    circuit_not_before_at: datetime | None = None,
) -> None:
    safe_headers = _safe_headers(headers) if headers is not None else {}
    with connection:
        connection.execute(
            """
            UPDATE capture_attempt
            SET completed_at = ?, outcome = ?, status_code = ?, final_url = ?,
                retry_after_at = ?, response_headers_json = ?,
                redirect_chain_json = ?, error = ?
            WHERE attempt_id = ?
            """,
            (
                _iso(completed_at),
                outcome,
                status_code,
                final_url,
                _iso(retry_after_at) if retry_after_at else None,
                json.dumps(safe_headers, sort_keys=True),
                json.dumps(redirect_chain, sort_keys=True),
                error,
                attempt_id,
            ),
        )
        if circuit_authority is not None and circuit_reason is not None:
            connection.execute(
                """
                INSERT INTO capture_host_state (
                    authority, last_request_at, blocked_reason, not_before_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(authority) DO UPDATE SET
                    last_request_at = excluded.last_request_at,
                    blocked_reason = excluded.blocked_reason,
                    not_before_at = excluded.not_before_at
                """,
                (
                    circuit_authority,
                    _iso(completed_at),
                    circuit_reason,
                    _iso(circuit_not_before_at) if circuit_not_before_at else None,
                ),
            )
        connection.execute(
            """
            UPDATE capture_entry
            SET status = ?, next_eligible_at = ?, last_status_code = ?,
                last_error = ?, updated_at = ?
            WHERE job_id = ? AND record_id = ?
            """,
            (
                entry_status,
                _iso(retry_after_at) if retry_after_at else None,
                status_code,
                error,
                _iso(completed_at),
                job_id,
                record_id,
            ),
        )


def _finish_success(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    job_id: str,
    record_id: str,
    source_url: str,
    final_url: str,
    observed_at: datetime,
    status_code: int,
    headers: Any,
    content_sha256: str,
    byte_size: int,
    object_path: Path,
    redirect_chain: list[dict[str, object]],
) -> str:
    safe_headers = _safe_headers(headers)
    capture_id = f"capture:{_digest(job_id, record_id, content_sha256)[:40]}"
    content_type = headers.get("Content-Type")
    etag = headers.get("ETag")
    last_modified = headers.get("Last-Modified")
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO capture_artifact (
                capture_id, job_id, record_id, source_url, final_url,
                observed_at, status_code, content_type, etag, last_modified,
                content_sha256, byte_size, object_path, response_headers_json
                , redirect_chain_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capture_id,
                job_id,
                record_id,
                source_url,
                final_url,
                _iso(observed_at),
                status_code,
                content_type,
                etag,
                last_modified,
                content_sha256,
                byte_size,
                str(object_path),
                json.dumps(safe_headers, sort_keys=True),
                json.dumps(redirect_chain, sort_keys=True),
            ),
        )
        connection.execute(
            """
            UPDATE capture_attempt
            SET completed_at = ?, outcome = 'complete', status_code = ?,
                final_url = ?, response_headers_json = ?, redirect_chain_json = ?
            WHERE attempt_id = ?
            """,
            (
                _iso(observed_at),
                status_code,
                final_url,
                json.dumps(safe_headers, sort_keys=True),
                json.dumps(redirect_chain, sort_keys=True),
                attempt_id,
            ),
        )
        connection.execute(
            """
            UPDATE capture_entry
            SET status = 'complete', next_eligible_at = NULL,
                last_status_code = ?, last_error = NULL, capture_id = ?,
                updated_at = ?
            WHERE job_id = ? AND record_id = ?
            """,
            (status_code, capture_id, _iso(observed_at), job_id, record_id),
        )
    return capture_id


def _entry_counts(connection: sqlite3.Connection, job_id: str) -> dict[str, int]:
    return {
        row["status"]: int(row["count"])
        for row in connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM capture_entry WHERE job_id = ? GROUP BY status
            """,
            (job_id,),
        )
    }


def plan_capture(
    manifest_path: str | Path,
    policy_path: str | Path,
    *,
    max_pages: int | None = None,
    max_http_requests: int | None = None,
) -> dict[str, object]:
    policy, policy_sha256 = load_policy(policy_path, require_live=False)
    manifest = Path(manifest_path).resolve()
    entries, manifest_sha256 = _load_manifest(manifest, policy)
    requested = policy.max_pages_per_run if max_pages is None else max_pages
    if requested < 1:
        raise ValueError("max_pages must be positive")
    effective = min(requested, policy.max_pages_per_run, len(entries))
    requested_http = (
        policy.max_http_requests_per_run
        if max_http_requests is None
        else max_http_requests
    )
    if requested_http < 1:
        raise ValueError("max_http_requests must be positive")
    effective_http = min(requested_http, policy.max_http_requests_per_run)
    return {
        "network_used": False,
        "live_enabled": policy.live_enabled,
        "authorization_ref": policy.authorization_ref,
        "authorization_scope": policy.authorization_scope,
        "manifest_entries": len(entries),
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy_sha256,
        "effective_page_limit": effective,
        "effective_http_request_limit": effective_http,
        "min_interval_seconds": policy.min_interval_seconds,
        "allowed_hosts": list(policy.allowed_hosts),
        "allowed_path_prefixes": list(policy.allowed_path_prefixes),
        "robots_txt_mode": policy.robots_txt_mode,
    }


def capture_manifest(
    state_db: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    stream: str,
    policy_path: str | Path,
    max_pages: int | None = None,
    max_http_requests: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Any | None = None,
    now_fn: Callable[[], datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run one exclusive, restartable authorized capture process."""

    state_path = Path(state_db).resolve()
    lock_handle = _acquire_run_lock(state_path)
    try:
        return _capture_manifest_locked(
            state_path,
            manifest_path,
            output_dir,
            stream=stream,
            policy_path=policy_path,
            max_pages=max_pages,
            max_http_requests=max_http_requests,
            timeout_seconds=timeout_seconds,
            opener=opener,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
    finally:
        _release_run_lock(lock_handle)


def clear_host_circuit(
    state_db: str | Path,
    *,
    authority: str,
    authorization_ref: str,
    reason: str,
    now_fn: Callable[[], datetime] = _utc_now,
) -> dict[str, str]:
    """Clear a persistent 401/403-style stop after documented human review.

    This does not override a 429 cooldown, reset the last-request timestamp, or
    retry anything.  It merely records an explicit authorization reference and
    makes a future, separately invoked capture eligible to request again.
    """

    normalized_authority = authority.strip().casefold()
    if (
        not normalized_authority
        or "/" in normalized_authority
        or "@" in normalized_authority
        or any(character.isspace() for character in normalized_authority)
    ):
        raise ValueError("authority must be a host[:port] without whitespace")
    review_ref = authorization_ref.strip()
    review_reason = reason.strip()
    if not review_ref or not review_reason:
        raise ValueError("authorization_ref and reason must be non-empty")

    state_path = Path(state_db).resolve()
    lock_handle = _acquire_run_lock(state_path)
    try:
        connection = _open_existing_state(state_path)
        try:
            row = connection.execute(
                """
                SELECT blocked_reason, not_before_at
                FROM capture_host_state
                WHERE authority = ?
                """,
                (normalized_authority,),
            ).fetchone()
            if row is None or row["blocked_reason"] is None:
                raise ValueError(f"no persistent host circuit is open for {normalized_authority}")
            blocked_reason = str(row["blocked_reason"])
            if blocked_reason not in {
                f"http_{status_code}" for status_code in _PERMANENT_HOST_BLOCK_CODES
            }:
                raise ValueError(
                    f"{normalized_authority} is not in a manually clearable permanent stop "
                    f"({blocked_reason}); respect its cooldown instead"
                )
            reviewed_at = _iso(now_fn())
            review_id = f"host-review:{_digest(normalized_authority, reviewed_at, review_ref, review_reason)[:40]}"
            with connection:
                connection.execute(
                    """
                    UPDATE capture_host_state
                    SET blocked_reason = NULL, not_before_at = NULL
                    WHERE authority = ?
                    """,
                    (normalized_authority,),
                )
                connection.execute(
                    """
                    INSERT INTO capture_host_review (
                        review_id, authority, previous_blocked_reason, reviewed_at,
                        authorization_ref, reason
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id,
                        normalized_authority,
                        blocked_reason,
                        reviewed_at,
                        review_ref,
                        review_reason,
                    ),
                )
            return {
                "authority": normalized_authority,
                "previous_blocked_reason": blocked_reason,
                "reviewed_at": reviewed_at,
                "review_id": review_id,
            }
        finally:
            connection.close()
    finally:
        _release_run_lock(lock_handle)


def _capture_manifest_locked(
    state_db: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    stream: str,
    policy_path: str | Path,
    max_pages: int | None = None,
    max_http_requests: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Any | None = None,
    now_fn: Callable[[], datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Capture explicitly authorized pages with durable, restartable state.

    The function is deliberately a URL-manifest consumer, not a discovery or
    anti-bot client. It performs sequential ordinary GET requests only. A 403
    or 429 pauses the entire run; completed records remain committed and are
    skipped on restart.
    """

    stream = stream.strip()
    if not stream:
        raise ValueError("stream must be non-empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 300:
        raise ValueError("timeout_seconds must be positive")
    policy_file = Path(policy_path).resolve()
    policy, policy_sha256 = load_policy(policy_file, require_live=True)
    now = now_fn().astimezone(timezone.utc)
    policy.assert_authorization_active(now)
    if policy.robots_txt_mode == "respect":
        raise CapturePolicyError(
            "robots_txt_mode=respect is fail-closed until the robots snapshot "
            "collector is configured; use written_permission_override only when "
            "the written authorization explicitly covers these routes"
        )

    manifest = Path(manifest_path).resolve()
    entries, manifest_sha256 = _load_manifest(manifest, policy)
    requested_limit = policy.max_pages_per_run if max_pages is None else max_pages
    if requested_limit < 1:
        raise ValueError("max_pages must be positive")
    page_limit = min(requested_limit, policy.max_pages_per_run)
    requested_http_limit = (
        policy.max_http_requests_per_run
        if max_http_requests is None
        else max_http_requests
    )
    if requested_http_limit < 1:
        raise ValueError("max_http_requests must be positive")
    http_request_limit = min(
        requested_http_limit, policy.max_http_requests_per_run
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = _connect_state(Path(state_db).resolve())
    try:
        job_id = _initialize_job(
            connection,
            stream=stream,
            manifest_path=manifest,
            manifest_sha256=manifest_sha256,
            policy_path=policy_file,
            policy_sha256=policy_sha256,
            authorization_ref=policy.authorization_ref,
            authorization_scope=policy.authorization_scope,
            output_dir=output,
            entries=entries,
            now=now,
        )
    except BaseException:
        connection.close()
        raise
    http = opener if opener is not None else _build_opener(policy)
    attempted_this_run = 0
    http_requests_this_run = 0
    captured_this_run = 0
    stopped_reason: str | None = None

    while attempted_this_run < page_limit:
        if http_requests_this_run >= http_request_limit:
            stopped_reason = "http_request_limit"
            break
        entry = connection.execute(
            """
            SELECT * FROM capture_entry
            WHERE job_id = ? AND status NOT IN ('complete', 'not_found',
                                                'http_error', 'invalid_content',
                                                'retry_exhausted')
            ORDER BY ordinal LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if entry is None:
            break
        if entry["status"] in {"blocked", "paused"}:
            if entry["status"] == "paused" and entry["next_eligible_at"]:
                eligible = _parse_iso_datetime(entry["next_eligible_at"], "next_eligible_at")
                if now_fn().astimezone(timezone.utc) >= eligible:
                    with connection:
                        connection.execute(
                            """
                            UPDATE capture_entry SET status = 'pending', updated_at = ?
                            WHERE job_id = ? AND record_id = ?
                            """,
                            (_iso(now_fn()), job_id, entry["record_id"]),
                        )
                    continue
            stopped_reason = entry["status"]
            break
        if int(entry["attempts"]) >= policy.max_attempts_per_url:
            with connection:
                connection.execute(
                    """
                    UPDATE capture_entry
                    SET status = 'retry_exhausted', next_eligible_at = NULL,
                        last_error = 'maximum attempt count reached', updated_at = ?
                    WHERE job_id = ? AND record_id = ?
                    """,
                    (_iso(now_fn()), job_id, entry["record_id"]),
                )
            stopped_reason = "attempt_limit"
            break
        if entry["next_eligible_at"]:
            eligible = _parse_iso_datetime(entry["next_eligible_at"], "next_eligible_at")
            remaining = (eligible - now_fn().astimezone(timezone.utc)).total_seconds()
            if remaining > 0:
                try:
                    sleep_fn(remaining)
                except BaseException:
                    connection.close()
                    raise

        try:
            request_started = _wait_for_request_slot(
                connection,
                _authority(entry["url"]),
                policy.min_interval_seconds,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
            )
        except HostCircuitOpenError as error:
            stopped_reason = error.reason
            break
        except BaseException:
            connection.close()
            raise
        try:
            policy.assert_authorization_active(now_fn())
        except AuthorizationWindowError:
            stopped_reason = "authorization_window_closed"
            break
        attempt_number, attempt_id = _start_attempt(
            connection, job_id, entry["record_id"], request_started
        )
        attempted_this_run += 1
        try:
            (
                status_code,
                final_url,
                headers,
                body,
                redirect_chain,
                request_count,
            ) = _request_chain(
                connection,
                http,
                entry["url"],
                policy,
                timeout_seconds,
                first_requested_at=request_started,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                request_budget=http_request_limit - http_requests_this_run,
            )
            http_requests_this_run += request_count
            observed_at = now_fn().astimezone(timezone.utc)
            if status_code != 200:
                _finish_failure(
                    connection,
                    attempt_id=attempt_id,
                    job_id=job_id,
                    record_id=entry["record_id"],
                    completed_at=observed_at,
                    outcome="unexpected_success_status",
                    entry_status="http_error",
                    status_code=status_code,
                    final_url=final_url,
                    retry_after_at=None,
                    headers=headers,
                    redirect_chain=redirect_chain,
                    error=f"expected HTTP 200, got {status_code}",
                )
                continue
            requested_entity = _source_entity_key(entry["url"], entry["page_type"])
            final_entity = _source_entity_key(final_url, entry["page_type"])
            # A slug normalisation is harmless; a numerical entity switch is
            # not.  Persisting the body under the requested manifest record
            # would make a map/match corpus claim that A was captured when the
            # server actually returned B.  Keep the failed redirect evidence
            # in the state DB and require an explicit new manifest decision.
            if requested_entity is not None and final_entity != requested_entity:
                _finish_failure(
                    connection,
                    attempt_id=attempt_id,
                    job_id=job_id,
                    record_id=entry["record_id"],
                    completed_at=observed_at,
                    outcome="redirect_entity_changed",
                    entry_status="invalid_content",
                    status_code=status_code,
                    final_url=final_url,
                    retry_after_at=None,
                    headers=headers,
                    redirect_chain=redirect_chain,
                    error=(
                        "redirect changed requested numerical entity from "
                        f"{requested_entity[0]}:{requested_entity[1]} to "
                        + (
                            f"{final_entity[0]}:{final_entity[1]}"
                            if final_entity is not None
                            else "a URL without the requested entity"
                        )
                    ),
                )
                continue
            content_type = str(headers.get("Content-Type", "")).lower()
            content_encoding = str(headers.get("Content-Encoding", "")).strip().lower()
            if content_encoding not in {"", "identity"}:
                _finish_failure(
                    connection,
                    attempt_id=attempt_id,
                    job_id=job_id,
                    record_id=entry["record_id"],
                    completed_at=observed_at,
                    outcome="invalid_content_encoding",
                    entry_status="invalid_content",
                    status_code=status_code,
                    final_url=final_url,
                    retry_after_at=None,
                    headers=headers,
                    redirect_chain=redirect_chain,
                    error=f"unexpected content encoding: {content_encoding}",
                )
                continue
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                _finish_failure(
                    connection,
                    attempt_id=attempt_id,
                    job_id=job_id,
                    record_id=entry["record_id"],
                    completed_at=observed_at,
                    outcome="invalid_content",
                    entry_status="invalid_content",
                    status_code=status_code,
                    final_url=final_url,
                    retry_after_at=None,
                    headers=headers,
                    redirect_chain=redirect_chain,
                    error=f"unexpected content type: {content_type or 'missing'}",
                )
                continue
            try:
                content_sha256, relative_path = _store_artifact(output, body)
            except OSError as error:
                raise ArtifactStorageError(str(error)) from error
            _finish_success(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                source_url=entry["url"],
                final_url=final_url,
                observed_at=observed_at,
                status_code=status_code,
                headers=headers,
                content_sha256=content_sha256,
                byte_size=len(body),
                object_path=relative_path,
                redirect_chain=redirect_chain,
            )
            captured_this_run += 1
        except HTTPError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 1))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            status_code = int(error.code)
            headers = error.headers
            final_url = str(error.geturl())
            if status_code in _PERMANENT_HOST_BLOCK_CODES:
                outcome = entry_status = "blocked"
                retry_at = None
                stopped_reason = f"http_{status_code}"
            elif status_code == 429:
                outcome = entry_status = "paused"
                backoff = policy.base_backoff_seconds * (2 ** (attempt_number - 1))
                retry_at = _retry_after_at(headers, completed_at, backoff)
                stopped_reason = "http_429"
            elif status_code in {404, 410}:
                outcome = entry_status = "not_found"
                retry_at = None
            elif 500 <= status_code <= 599:
                if attempt_number >= policy.max_attempts_per_url:
                    outcome = entry_status = "retry_exhausted"
                    retry_at = None
                    stopped_reason = "attempt_limit"
                else:
                    outcome = "retryable_http_error"
                    retry_at = _retry_after_at(
                        headers,
                        completed_at,
                        policy.base_backoff_seconds * (2 ** (attempt_number - 1)),
                    )
                    entry_status = "pending"
            else:
                outcome = entry_status = "http_error"
                retry_at = None
                stopped_reason = f"http_{status_code}"
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome=outcome,
                entry_status=entry_status,
                status_code=status_code,
                final_url=final_url,
                retry_after_at=retry_at,
                headers=headers,
                redirect_chain=redirect_chain,
                error=f"HTTP {status_code}",
                circuit_authority=(
                    _authority(final_url or entry["url"])
                    if status_code in _PERMANENT_HOST_BLOCK_CODES | {429}
                    else None
                ),
                circuit_reason=(
                    f"http_{status_code}"
                    if status_code in _PERMANENT_HOST_BLOCK_CODES | {429}
                    else None
                ),
                circuit_not_before_at=retry_at,
            )
            error.close()
            if stopped_reason is not None:
                break
        except AuthorizationWindowError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 0))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="authorization_window_closed",
                entry_status="pending",
                status_code=None,
                final_url=None,
                retry_after_at=None,
                headers=None,
                redirect_chain=redirect_chain,
                error=f"{type(error).__name__}: {error}",
            )
            stopped_reason = "authorization_window_closed"
            break
        except HttpRequestBudgetError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 0))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="http_request_limit",
                entry_status="pending",
                status_code=None,
                final_url=None,
                retry_after_at=None,
                headers=None,
                redirect_chain=redirect_chain,
                error=str(error),
            )
            stopped_reason = "http_request_limit"
            break
        except HostCircuitOpenError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 0))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            retry_at = (
                _parse_iso_datetime(error.not_before_at, "not_before_at")
                if error.not_before_at
                else None
            )
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="host_circuit_open",
                entry_status=(
                    "blocked"
                    if error.reason
                    in {f"http_{code}" for code in _PERMANENT_HOST_BLOCK_CODES}
                    else "pending"
                ),
                status_code=None,
                final_url=None,
                retry_after_at=retry_at,
                headers=None,
                redirect_chain=redirect_chain,
                error=error.reason,
            )
            stopped_reason = error.reason
            break
        except CapturePolicyError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 1))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="redirect_blocked",
                entry_status="blocked",
                status_code=None,
                final_url=None,
                retry_after_at=None,
                headers=None,
                redirect_chain=redirect_chain,
                error=f"{type(error).__name__}: {error}",
            )
            stopped_reason = "redirect_blocked"
            break
        except ArtifactStorageError as error:
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="local_storage_error",
                entry_status="pending",
                status_code=200,
                final_url=final_url,
                retry_after_at=None,
                headers=headers,
                redirect_chain=redirect_chain,
                error=f"{type(error).__name__}: {error}",
            )
            stopped_reason = "local_storage_error"
            break
        except ResponseTooLargeError as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 1))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="invalid_content",
                entry_status="invalid_content",
                status_code=None,
                final_url=None,
                retry_after_at=None,
                headers=None,
                redirect_chain=redirect_chain,
                error=f"{type(error).__name__}: {error}",
            )
        except (URLError, TimeoutError, OSError, HTTPException) as error:
            http_requests_this_run += int(getattr(error, "capture_request_count", 1))
            redirect_chain = list(getattr(error, "capture_redirect_chain", []))
            completed_at = now_fn().astimezone(timezone.utc)
            exhausted = attempt_number >= policy.max_attempts_per_url
            retry_at = (
                None
                if exhausted
                else completed_at
                + timedelta(
                    seconds=policy.base_backoff_seconds * (2 ** (attempt_number - 1))
                )
            )
            _finish_failure(
                connection,
                attempt_id=attempt_id,
                job_id=job_id,
                record_id=entry["record_id"],
                completed_at=completed_at,
                outcome="retry_exhausted" if exhausted else "retryable_transport_error",
                entry_status="retry_exhausted" if exhausted else "pending",
                status_code=None,
                final_url=None,
                retry_after_at=retry_at,
                headers=None,
                redirect_chain=redirect_chain,
                error=f"{type(error).__name__}: {error}",
            )
            if exhausted:
                stopped_reason = "attempt_limit"
                break
        except BaseException:
            connection.close()
            raise

    with connection:
        connection.execute(
            "UPDATE capture_job SET updated_at = ? WHERE job_id = ?",
            (_iso(now_fn()), job_id),
        )
    counts = _entry_counts(connection, job_id)
    pending = sum(
        count for status, count in counts.items() if status not in _TERMINAL_STATUSES
    )
    failure_count = sum(
        count
        for status, count in counts.items()
        if status in _TERMINAL_STATUSES and status != "complete"
    )
    all_captured = counts.get("complete", 0) == sum(counts.values())
    result: dict[str, object] = {
        "job_id": job_id,
        "stream": stream,
        "attempted_this_run": attempted_this_run,
        "http_requests_this_run": http_requests_this_run,
        "captured_this_run": captured_this_run,
        "counts": counts,
        "pending": pending,
        "failure_count": failure_count,
        "finished": pending == 0,
        "all_captured": all_captured,
        "complete": pending == 0,
        "stopped_reason": stopped_reason,
        "network_policy": {
            "authorization_ref": policy.authorization_ref,
            "min_interval_seconds": policy.min_interval_seconds,
            "page_limit": page_limit,
            "http_request_limit": http_request_limit,
            "robots_txt_mode": policy.robots_txt_mode,
        },
    }
    connection.close()
    return result


def capture_index(
    state_db: str | Path, *, stream: str, allow_partial: bool = False
) -> list[dict[str, object]]:
    connection = _open_existing_state(Path(state_db).resolve())
    job = connection.execute(
        "SELECT * FROM capture_job WHERE stream = ?", (stream,)
    ).fetchone()
    if job is None:
        connection.close()
        raise ValueError(f"unknown capture stream: {stream}")
    counts = _entry_counts(connection, job["job_id"])
    if not allow_partial and any(status != "complete" for status in counts):
        connection.close()
        raise CaptureIncompleteError(
            "capture stream is not all-success; use allow_partial only for explicit "
            f"diagnostics (counts={json.dumps(counts, sort_keys=True)})"
        )
    rows = connection.execute(
        """
        SELECT e.record_id, e.page_type, a.source_url, a.final_url,
               a.observed_at, a.status_code, a.content_type, a.etag,
               a.last_modified, a.content_sha256, a.byte_size,
               j.output_dir, j.policy_sha256, j.authorization_ref,
               j.authorization_scope, a.object_path, m.metadata_json
        FROM capture_job AS j
        JOIN capture_entry AS e ON e.job_id = j.job_id
        JOIN capture_artifact AS a ON a.capture_id = e.capture_id
        LEFT JOIN capture_manifest_metadata AS m
          ON m.job_id = e.job_id AND m.record_id = e.record_id
        WHERE j.stream = ? AND e.status = 'complete'
        ORDER BY e.ordinal
        """,
        (stream,),
    ).fetchall()
    connection.close()
    return [
        {
            "record_id": row["record_id"],
            "page_type": row["page_type"],
            "source_url": row["source_url"],
            "final_url": row["final_url"],
            "observed_at": row["observed_at"],
            "status_code": row["status_code"],
            "content_type": row["content_type"],
            "etag": row["etag"],
            "last_modified": row["last_modified"],
            "content_sha256": row["content_sha256"],
            "byte_size": row["byte_size"],
            "policy_sha256": row["policy_sha256"],
            "authorization_ref": row["authorization_ref"],
            "authorization_scope": row["authorization_scope"],
            "manifest_metadata": (
                json.loads(row["metadata_json"])
                if row["metadata_json"] is not None
                else {}
            ),
            "object_path": str(Path(row["output_dir"]) / row["object_path"]),
        }
        for row in rows
    ]


def index_to_jsonl(records: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for record in records
    )


def parsed_capture_records(
    state_db: str | Path, *, stream: str, allow_partial: bool = False
) -> Iterable[dict[str, object]]:
    """Verify every capture before exposing any parsed record.

    Records are staged in a bounded-memory temporary spool.  This matters for
    stdout JSONL exports: corruption or a parser-quality failure in a later
    artifact must not leave an apparently valid prefix that can be imported as
    a complete dataset.
    """

    from .hltv_offline import parse_file

    captures = capture_index(
        state_db, stream=stream, allow_partial=allow_partial
    )
    if any(str(capture["page_type"]) == "results" for capture in captures):
        raise ValueError(
            "results listing captures must be converted with "
            "extract-hltv-match-manifest, not parse-hltv-captures"
        )
    seen_entities: dict[tuple[str, str], str] = {}
    for capture in captures:
        entity_key = _source_entity_key(
            str(capture["final_url"]), str(capture["page_type"])
        )
        if entity_key is not None and entity_key in seen_entities:
            raise CaptureQualityError(
                f"multiple captures resolve to {entity_key[0]} {entity_key[1]}: "
                f"{seen_entities[entity_key]} and {capture['record_id']}"
            )
        if entity_key is not None:
            seen_entities[entity_key] = str(capture["record_id"])

    seen_record_ids: set[str] = set()
    with tempfile.SpooledTemporaryFile(
        mode="w+", encoding="utf-8", newline="\n", max_size=8 * 1024 * 1024
    ) as spool:
        for capture in captures:
            path = Path(str(capture["object_path"]))
            if not path.is_file():
                raise CaptureCorruptionError(f"captured object is missing: {path}")
            expected_hash = str(capture["content_sha256"])
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise CaptureCorruptionError(
                    f"captured object hash mismatch for {path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            records = parse_file(
                path,
                page_type=str(capture["page_type"]),
                source_url=str(capture["final_url"]),
                observed_at=str(capture["observed_at"]),
            )
            page_type = str(capture["page_type"])
            series_records = [
                record for record in records if record["kind"] == "series"
            ]
            if page_type == "match" or series_records:
                if not series_records or len(
                    series_records[0].get("payload", {}).get("teams", [])
                ) != 2:
                    raise CaptureQualityError(
                        f"captured match page lacks two identifiable teams: {path}"
                    )
            elif page_type in {"auto", "map-stats"} and not any(
                record["kind"] == "player_map_stats" for record in records
            ):
                raise CaptureQualityError(
                    f"captured map-stats page has no player statistics: {path}"
                )
            for record in records:
                parser_hash = record.get("source_document_sha256")
                if parser_hash != expected_hash:
                    raise CaptureCorruptionError(
                        "offline parser document hash disagrees with the capture manifest"
                    )
                record["capture_provenance"] = {
                    "requested_url": capture["source_url"],
                    "final_url": capture["final_url"],
                    "observed_at": capture["observed_at"],
                    "content_sha256": expected_hash,
                    "policy_sha256": capture["policy_sha256"],
                    "authorization_ref": capture["authorization_ref"],
                    "authorization_scope": capture["authorization_scope"],
                    "manifest_metadata": capture.get("manifest_metadata", {}),
                }
                record_id = str(record["record_id"])
                if record_id in seen_record_ids:
                    raise CaptureQualityError(
                        f"duplicate parsed record_id across captures: {record_id}"
                    )
                seen_record_ids.add(record_id)
                spool.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )

        spool.seek(0)
        for line in spool:
            yield json.loads(line)
