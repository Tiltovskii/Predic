from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from predic_v2.hltv_capture import (
    CaptureBusyError,
    CaptureCorruptionError,
    CaptureIncompleteError,
    CapturePolicyError,
    CaptureQualityError,
    CaptureSourceChangedError,
    clear_host_circuit,
    capture_index,
    capture_manifest,
    parsed_capture_records,
    plan_capture,
    _acquire_run_lock,
    _release_run_lock,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class FakeResponse:
    def __init__(
        self,
        url: str,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
        content_encoding: str | None = None,
    ) -> None:
        self._url = url
        self._body = io.BytesIO(body)
        self._status = status
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "ETag": '"fixture"',
        }
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self.closed = False

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[tuple[str, float]] = []
        self.request_headers: list[dict[str, str]] = []

    def open(self, request, timeout: float):
        self.requests.append((request.full_url, timeout))
        self.request_headers.append(dict(request.header_items()))
        if not self.outcomes:
            raise AssertionError("unexpected network request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class BrokenReadResponse(FakeResponse):
    def read(self, size: int = -1) -> bytes:
        raise IncompleteRead(b"partial", 100)


def http_error(url: str, status: int, headers: dict[str, str] | None = None):
    return HTTPError(url, status, f"HTTP {status}", headers or {}, io.BytesIO())


class HltvCaptureTest(unittest.TestCase):
    def _policy(self, root: Path, **overrides) -> Path:
        policy = {
            "live_enabled": True,
            "authorization_ref": "permission-email-2026-08-25",
            "authorization_scope": "research model; /matches and /stats/matches",
            "authorization_confirmed_at": "2026-08-25T10:00:00Z",
            "valid_until": None,
            "allowed_schemes": ["https"],
            "allowed_hosts": ["www.hltv.org"],
            "allowed_path_prefixes": ["/matches/", "/stats/matches/"],
            "allowed_query_keys": [],
            "user_agent": "PredicResearch/0.1 (+contact: test@example.invalid)",
            "contact": "test@example.invalid",
            "min_interval_seconds": 10,
            "max_pages_per_run": 20,
            "max_http_requests_per_run": 40,
            "max_response_bytes": 1024,
            "max_attempts_per_url": 3,
            "base_backoff_seconds": 60,
            "robots_txt_mode": "written_permission_override",
        }
        policy.update(overrides)
        path = root / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def _manifest(self, root: Path, urls: list[str]) -> Path:
        path = root / "manifest.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "record_id": f"page-{index}",
                        "page_type": "match",
                        "url": url,
                    }
                )
                + "\n"
                for index, url in enumerate(urls)
            ),
            encoding="utf-8",
        )
        return path

    def test_dry_run_validates_without_network_or_live_enablement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root, live_enabled=False, authorization_confirmed_at=None)
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a-vs-b"]
            )

            result = plan_capture(manifest, policy)

            self.assertFalse(result["network_used"])
            self.assertFalse(result["live_enabled"])
            self.assertEqual(1, result["manifest_entries"])

    def test_out_of_scope_url_is_rejected_before_state_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root)
            manifest = self._manifest(root, ["https://example.com/matches/1/a"])

            with self.assertRaises(CapturePolicyError):
                capture_manifest(
                    root / "state.sqlite3",
                    manifest,
                    root / "raw",
                    stream="test",
                    policy_path=policy,
                    opener=FakeOpener([]),
                )

            self.assertFalse((root / "state.sqlite3").exists())

    def test_unknown_query_key_is_rejected_by_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root)
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a?unexpected=1"]
            )

            with self.assertRaises(CapturePolicyError):
                plan_capture(manifest, policy)

    def test_manifest_page_type_typo_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "record_id": "one",
                        "page_type": "map_stats",
                        "url": "https://www.hltv.org/matches/1/a",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                plan_capture(manifest, policy)

    def test_contact_must_be_present_in_user_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root, user_agent="PredicResearch/0.1")
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a"]
            )

            with self.assertRaises(CapturePolicyError):
                plan_capture(manifest, policy)

    def test_default_port_variant_cannot_bypass_dedup_or_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root)
            manifest = self._manifest(
                root,
                [
                    "https://www.hltv.org/matches/1/a-vs-b",
                    "https://www.hltv.org:443/matches/1/a-vs-b",
                ],
            )

            with self.assertRaises(ValueError):
                plan_capture(manifest, policy)

    def test_two_slugs_for_same_numeric_entity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root)
            manifest = self._manifest(
                root,
                [
                    "https://www.hltv.org/matches/42/old-slug",
                    "https://www.hltv.org/matches/42/new-slug",
                ],
            )

            with self.assertRaises(ValueError):
                plan_capture(manifest, policy)

    def test_success_is_content_addressed_rate_limited_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a-vs-b",
                "https://www.hltv.org/matches/2/c-vs-d",
            ]
            policy = self._policy(root)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            first_opener = FakeOpener([FakeResponse(urls[0], b"<html>one</html>")])

            first = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                max_pages=1,
                opener=first_opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            second_opener = FakeOpener([FakeResponse(urls[1], b"<html>two</html>")])
            second = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=second_opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            third_opener = FakeOpener([])
            third = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=third_opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual(1, first["captured_this_run"])
            self.assertFalse(first["complete"])
            self.assertEqual(1, second["captured_this_run"])
            self.assertTrue(second["complete"])
            self.assertEqual(0, third["attempted_this_run"])
            self.assertEqual([10.0], clock.sleeps)
            records = capture_index(root / "state.sqlite3", stream="test")
            self.assertEqual(2, len(records))
            self.assertTrue(all(Path(row["object_path"]).is_file() for row in records))
            self.assertNotEqual(records[0]["content_sha256"], records[1]["content_sha256"])
            self.assertIn(
                "test@example.invalid",
                first_opener.request_headers[0]["User-agent"],
            )

    def test_partial_stream_requires_explicit_partial_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a",
                "https://www.hltv.org/matches/2/b",
            ]
            policy = self._policy(root)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                max_pages=1,
                opener=FakeOpener([FakeResponse(urls[0], b"<html>ok</html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            with self.assertRaises(CaptureIncompleteError):
                capture_index(root / "state.sqlite3", stream="test")
            self.assertEqual(
                1,
                len(
                    capture_index(
                        root / "state.sqlite3",
                        stream="test",
                        allow_partial=True,
                    )
                ),
            )

    def test_export_does_not_create_a_missing_state_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "missing.sqlite3"

            with self.assertRaises(FileNotFoundError):
                capture_index(state, stream="missing")

            self.assertFalse(state.exists())

    def test_same_body_is_one_blob_for_two_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a-vs-b",
                "https://www.hltv.org/matches/2/c-vs-d",
            ]
            policy = self._policy(root, min_interval_seconds=1)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            body = b"<html>same</html>"

            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [FakeResponse(urls[0], body), FakeResponse(urls[1], body)]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            records = capture_index(root / "state.sqlite3", stream="test")
            self.assertEqual(records[0]["object_path"], records[1]["object_path"])
            self.assertEqual(1, len(list((root / "raw" / "objects").glob("*/*.html"))))

    def test_partial_or_encoded_response_is_never_marked_complete(self) -> None:
        for response in (
            FakeResponse(
                "https://www.hltv.org/matches/1/a",
                b"partial",
                status=206,
            ),
            FakeResponse(
                "https://www.hltv.org/matches/1/a",
                b"compressed",
                content_encoding="gzip",
            ),
        ):
            with self.subTest(status=response.getcode(), encoding=response.headers.get("Content-Encoding")):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    url = "https://www.hltv.org/matches/1/a"
                    policy = self._policy(root)
                    manifest = self._manifest(root, [url])
                    result = capture_manifest(
                        root / "state.sqlite3",
                        manifest,
                        root / "raw",
                        stream="test",
                        policy_path=policy,
                        opener=FakeOpener([response]),
                        now_fn=FakeClock().now,
                    )

                    self.assertEqual(0, result["captured_this_run"])
                    self.assertEqual(
                        [],
                        capture_index(
                            root / "state.sqlite3",
                            stream="test",
                            allow_partial=True,
                        ),
                    )

    def test_captured_html_is_verified_parsed_and_keeps_capture_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/42/a-vs-b"
            html = b"""
            <link rel="canonical" href="https://www.hltv.org/matches/42/a-vs-b">
            <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
            <div class="match-status">Finished</div>
            """
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, html)]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            records = list(parsed_capture_records(root / "state.sqlite3", stream="test"))

            self.assertEqual(1, len(records))
            self.assertEqual("series", records[0]["kind"])
            self.assertEqual(clock.now().isoformat(), records[0]["observed_at"])
            self.assertIsNone(records[0]["known_at"])
            self.assertEqual(
                "permission-email-2026-08-25",
                records[0]["capture_provenance"]["authorization_ref"],
            )

    def test_empty_200_html_is_captured_but_fails_parse_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/42/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, b"")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            with self.assertRaises(CaptureQualityError):
                list(parsed_capture_records(root / "state.sqlite3", stream="test"))

    def test_corrupted_capture_is_refused_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/42/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, b"<html></html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            captured = capture_index(root / "state.sqlite3", stream="test")[0]
            Path(captured["object_path"]).write_bytes(b"corrupted")

            with self.assertRaises(CaptureCorruptionError):
                list(parsed_capture_records(root / "state.sqlite3", stream="test"))

    def test_late_corruption_is_detected_before_first_record_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/41/a-vs-b",
                "https://www.hltv.org/matches/42/c-vs-d",
            ]
            bodies = [
                b"""
                <link rel="canonical" href="https://www.hltv.org/matches/41/a-vs-b">
                <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
                """,
                b"""
                <link rel="canonical" href="https://www.hltv.org/matches/42/c-vs-d">
                <a href="/team/3/c">C</a><a href="/team/4/d">D</a>
                """,
            ]
            policy = self._policy(root, min_interval_seconds=1)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [
                        FakeResponse(urls[0], bodies[0]),
                        FakeResponse(urls[1], bodies[1]),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            captures = capture_index(root / "state.sqlite3", stream="test")
            Path(captures[1]["object_path"]).write_bytes(b"corrupted")

            records = iter(
                parsed_capture_records(root / "state.sqlite3", stream="test")
            )
            with self.assertRaises(CaptureCorruptionError):
                next(records)

    def test_local_storage_error_stops_without_automatic_refetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/42/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            opener = FakeOpener([FakeResponse(url, b"<html>ok</html>")])

            with patch(
                "predic_v2.hltv_capture._store_artifact",
                side_effect=OSError("disk full"),
            ):
                result = capture_manifest(
                    root / "state.sqlite3",
                    manifest,
                    root / "raw",
                    stream="test",
                    policy_path=policy,
                    opener=opener,
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )

            self.assertEqual("local_storage_error", result["stopped_reason"])
            self.assertEqual(1, result["http_requests_this_run"])
            self.assertEqual(1, len(opener.requests))

    def test_manifest_change_requires_new_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, b"<html></html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "record_id": "page-2",
                            "page_type": "match",
                            "url": "https://www.hltv.org/matches/2/c-vs-d",
                        }
                    )
                    + "\n"
                )

            with self.assertRaises(CaptureSourceChangedError):
                capture_manifest(
                    root / "state.sqlite3",
                    manifest,
                    root / "raw",
                    stream="test",
                    policy_path=policy,
                    opener=FakeOpener([]),
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )

    def test_403_stops_run_and_stays_blocked_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a-vs-b",
                "https://www.hltv.org/matches/2/c-vs-d",
            ]
            policy = self._policy(root)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            opener = FakeOpener([http_error(urls[0], 403)])

            first = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            second_opener = FakeOpener([])
            second = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=second_opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_403", first["stopped_reason"])
            self.assertEqual("blocked", second["stopped_reason"])
            self.assertEqual(1, len(opener.requests))
            self.assertEqual([], second_opener.requests)

    def test_documented_review_can_clear_a_403_host_circuit_for_a_new_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="blocked",
                policy_path=policy,
                opener=FakeOpener([http_error(url, 403)]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            review = clear_host_circuit(
                root / "state.sqlite3",
                authority="www.hltv.org",
                authorization_ref="hltv-follow-up-2026-08-25",
                reason="HLTV confirmed the requested route after the pilot stop",
                now_fn=clock.now,
            )
            resumed = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="after-review",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, b"<html>ok</html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_403", review["previous_blocked_reason"])
            self.assertEqual(1, resumed["captured_this_run"])
            connection = sqlite3.connect(root / "state.sqlite3")
            review_count = connection.execute(
                "SELECT COUNT(*) FROM capture_host_review"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(1, review_count)

    def test_429_retry_after_is_persisted_and_run_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([http_error(url, 429, {"Retry-After": "120"})]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_429", result["stopped_reason"])
            connection = sqlite3.connect(root / "state.sqlite3")
            next_eligible = connection.execute(
                "SELECT next_eligible_at FROM capture_entry"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(
                clock.now() + timedelta(seconds=120),
                datetime.fromisoformat(next_eligible),
            )

    def test_short_retry_after_never_weakens_local_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root, base_backoff_seconds=60)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([http_error(url, 429, {"Retry-After": "5"})]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            connection = sqlite3.connect(root / "state.sqlite3")
            next_eligible = connection.execute(
                "SELECT next_eligible_at FROM capture_entry"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(
                clock.now() + timedelta(seconds=60),
                datetime.fromisoformat(next_eligible),
            )

    def test_allowed_redirect_is_scoped_and_rate_limited_per_hop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "https://www.hltv.org/matches/1/old"
            target = "https://www.hltv.org/matches/1/new"
            policy = self._policy(root)
            manifest = self._manifest(root, [original])
            clock = FakeClock()
            opener = FakeOpener(
                [
                    http_error(original, 302, {"Location": "/matches/1/new"}),
                    FakeResponse(target, b"<html>ok</html>"),
                ]
            )

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual(2, result["http_requests_this_run"])
            self.assertEqual([10.0], clock.sleeps)
            self.assertEqual([original, target], [row[0] for row in opener.requests])
            connection = sqlite3.connect(root / "state.sqlite3")
            chain = json.loads(
                connection.execute(
                    "SELECT redirect_chain_json FROM capture_attempt"
                ).fetchone()[0]
            )
            connection.close()
            self.assertEqual(target, chain[0]["target_url"])
            self.assertTrue(chain[0]["allowed"])

    def test_http_request_budget_stops_before_redirect_hop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "https://www.hltv.org/matches/1/old"
            target = "https://www.hltv.org/matches/1/new"
            policy = self._policy(root, max_http_requests_per_run=1)
            manifest = self._manifest(root, [original])
            clock = FakeClock()
            opener = FakeOpener(
                [http_error(original, 302, {"Location": "/matches/1/new"})]
            )

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_request_limit", result["stopped_reason"])
            self.assertEqual(1, result["http_requests_this_run"])
            self.assertEqual([original], [row[0] for row in opener.requests])
            self.assertNotIn(target, [row[0] for row in opener.requests])

    def test_out_of_scope_redirect_stops_before_target_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "https://www.hltv.org/matches/1/old"
            target = "https://example.com/steal"
            policy = self._policy(root)
            manifest = self._manifest(root, [original])
            clock = FakeClock()
            opener = FakeOpener([http_error(original, 302, {"Location": target})])

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("redirect_blocked", result["stopped_reason"])
            self.assertEqual(1, result["http_requests_this_run"])
            self.assertEqual([original], [row[0] for row in opener.requests])

    def test_redirect_changing_match_numeric_identity_is_refused_before_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "https://www.hltv.org/matches/41/old"
            target = "https://www.hltv.org/matches/42/new"
            html = b"""
            <link rel="canonical" href="https://www.hltv.org/matches/42/new">
            <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
            """
            policy = self._policy(root)
            manifest = self._manifest(root, [original])
            clock = FakeClock()
            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [
                        http_error(original, 302, {"Location": target}),
                        FakeResponse(target, html),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual({"invalid_content": 1}, result["counts"])
            self.assertEqual(1, result["failure_count"])
            with self.assertRaises(CaptureIncompleteError):
                list(parsed_capture_records(root / "state.sqlite3", stream="test"))
            self.assertEqual(
                [],
                capture_index(
                    root / "state.sqlite3", stream="test", allow_partial=True
                ),
            )
            connection = sqlite3.connect(root / "state.sqlite3")
            try:
                attempt = connection.execute(
                    "SELECT outcome, final_url FROM capture_attempt"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(("redirect_entity_changed", target), attempt)

    def test_redirect_changing_map_stats_numeric_identity_is_refused_before_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "https://www.hltv.org/stats/matches/mapstatsid/501/old"
            target = "https://www.hltv.org/stats/matches/mapstatsid/502/new"
            html = b"""
            <link rel="canonical" href="https://www.hltv.org/stats/matches/mapstatsid/502/new">
            """
            policy = self._policy(root, min_interval_seconds=1)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "record_id": "map-stats-501",
                        "page_type": "map-stats",
                        "url": original,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            clock = FakeClock()
            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [
                        http_error(original, 302, {"Location": target}),
                        FakeResponse(target, html),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual({"invalid_content": 1}, result["counts"])
            self.assertEqual(
                [],
                capture_index(
                    root / "state.sqlite3", stream="test", allow_partial=True
                ),
            )

    def test_5xx_uses_exponential_backoff_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root, base_backoff_seconds=60)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [http_error(url, 500), FakeResponse(url, b"<html>ok</html>")]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual(1, result["captured_this_run"])
            self.assertEqual(2, result["http_requests_this_run"])
            self.assertEqual([60.0], clock.sleeps)

    def test_503_retry_after_is_stronger_than_local_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root, base_backoff_seconds=60)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [
                        http_error(url, 503, {"Retry-After": "3600"}),
                        FakeResponse(url, b"<html>ok</html>"),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual(1, result["captured_this_run"])
            self.assertEqual([3600.0], clock.sleeps)

    def test_incomplete_read_is_retryable_and_never_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root, max_attempts_per_url=1)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([BrokenReadResponse(url, b"")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("attempt_limit", result["stopped_reason"])
            self.assertEqual(0, result["captured_this_run"])
            self.assertEqual(1, result["counts"]["retry_exhausted"])

    def test_short_200_body_with_long_content_length_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root, max_attempts_per_url=1)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            response = FakeResponse(url, b"short")
            response.headers["Content-Length"] = "999"

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([response]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("attempt_limit", result["stopped_reason"])
            self.assertEqual(0, result["captured_this_run"])

    def test_exhausted_page_does_not_wedge_later_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a-vs-b",
                "https://www.hltv.org/matches/2/c-vs-d",
            ]
            policy = self._policy(
                root, max_attempts_per_url=2, base_backoff_seconds=1
            )
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            first = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([http_error(urls[0], 500), http_error(urls[0], 500)]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            second = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(urls[1], b"<html>ok</html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("attempt_limit", first["stopped_reason"])
            self.assertEqual(1, second["captured_this_run"])
            self.assertEqual(1, second["counts"]["retry_exhausted"])

    def test_403_host_circuit_blocks_a_new_stream_in_same_state_db(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="first",
                policy_path=policy,
                opener=FakeOpener([http_error(url, 403)]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )
            second_opener = FakeOpener([])

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="second",
                policy_path=policy,
                opener=second_opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_403", result["stopped_reason"])
            self.assertEqual([], second_opener.requests)

    def test_waf_style_418_stops_and_opens_host_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a",
                "https://www.hltv.org/matches/2/b",
            ]
            policy = self._policy(root)
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            opener = FakeOpener([http_error(urls[0], 418)])

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_418", result["stopped_reason"])
            self.assertEqual(1, len(opener.requests))

    def test_permission_expiry_is_rechecked_after_rate_limit_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = [
                "https://www.hltv.org/matches/1/a-vs-b",
                "https://www.hltv.org/matches/2/c-vs-d",
            ]
            policy = self._policy(root, valid_until="2026-08-25T12:00:05Z")
            manifest = self._manifest(root, urls)
            clock = FakeClock()
            opener = FakeOpener([FakeResponse(urls[0], b"<html>ok</html>")])

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=opener,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("authorization_window_closed", result["stopped_reason"])
            self.assertEqual(1, len(opener.requests))
            self.assertEqual([10.0], clock.sleeps)

    def test_huge_retry_after_pauses_without_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()

            result = capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener(
                    [http_error(url, 429, {"Retry-After": "9" * 100})]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertEqual("http_429", result["stopped_reason"])
            connection = sqlite3.connect(root / "state.sqlite3")
            next_eligible = connection.execute(
                "SELECT next_eligible_at FROM capture_entry"
            ).fetchone()[0]
            connection.close()
            self.assertTrue(next_eligible.startswith("9999-12-31"))

    def test_non_finite_policy_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root, min_interval_seconds=float("nan"))
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a-vs-b"]
            )

            with self.assertRaises(CapturePolicyError):
                plan_capture(manifest, policy)

    def test_crashed_attempt_is_closed_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/1/a-vs-b"
            policy = self._policy(root)
            manifest = self._manifest(root, [url])
            clock = FakeClock()
            with self.assertRaises(KeyboardInterrupt):
                capture_manifest(
                    root / "state.sqlite3",
                    manifest,
                    root / "raw",
                    stream="test",
                    policy_path=policy,
                    opener=FakeOpener([KeyboardInterrupt()]),
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )

            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="test",
                policy_path=policy,
                opener=FakeOpener([FakeResponse(url, b"<html>ok</html>")]),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            connection = sqlite3.connect(root / "state.sqlite3")
            outcomes = [
                row[0]
                for row in connection.execute(
                    "SELECT outcome FROM capture_attempt ORDER BY attempt_number"
                )
            ]
            connection.close()
            self.assertEqual(["abandoned_after_crash", "complete"], outcomes)

    def test_second_process_cannot_share_one_capture_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.sqlite3"
            policy = self._policy(root)
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a-vs-b"]
            )
            lock = _acquire_run_lock(state)
            try:
                with self.assertRaises(CaptureBusyError):
                    capture_manifest(
                        state,
                        manifest,
                        root / "raw",
                        stream="test",
                        policy_path=policy,
                        opener=FakeOpener([]),
                    )
            finally:
                _release_run_lock(lock)

    def test_expired_authorization_fails_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self._policy(root, valid_until="2026-08-24T00:00:00Z")
            manifest = self._manifest(
                root, ["https://www.hltv.org/matches/1/a-vs-b"]
            )
            opener = FakeOpener([])

            with self.assertRaises(CapturePolicyError):
                capture_manifest(
                    root / "state.sqlite3",
                    manifest,
                    root / "raw",
                    stream="test",
                    policy_path=policy,
                    opener=opener,
                    now_fn=FakeClock().now,
                )

            self.assertEqual([], opener.requests)


if __name__ == "__main__":
    unittest.main()
