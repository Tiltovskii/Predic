from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from predic_v2.hltv_capture import capture_manifest
from predic_v2.hltv_discovery import (
    HltvDiscoveryError,
    aggregate_match_manifest,
    derive_results_pagination_manifest,
    extract_mapstats_manifest,
    extract_match_manifest,
    generate_results_manifest,
    parse_results_html,
)


FIXTURES = Path(__file__).parent / "fixtures"


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeResponse:
    def __init__(self, url: str, body: bytes) -> None:
        self._url = url
        self._body = body
        self.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(body)),
        }

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            result, self._body = self._body, b""
            return result
        result, self._body = self._body[:size], self._body[size:]
        return result

    def close(self) -> None:
        return None


class FakeOpener:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)

    def open(self, request, timeout: float):
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


class HltvDiscoveryTest(unittest.TestCase):
    def _policy(self, root: Path) -> Path:
        payload = {
            "live_enabled": True,
            "authorization_ref": "fixture-permission",
            "authorization_scope": "fixture only",
            "authorization_confirmed_at": "2026-08-25T10:00:00Z",
            "valid_until": None,
            "allowed_schemes": ["https"],
            "allowed_hosts": ["www.hltv.org"],
            "allowed_path_prefixes": ["/results", "/matches/", "/stats/matches/"],
            "allowed_query_keys": ["offset", "startDate", "endDate"],
            "user_agent": "PredicFixture/0.1 (+fixture@example.invalid)",
            "contact": "fixture@example.invalid",
            "min_interval_seconds": 1,
            "max_pages_per_run": 20,
            "max_http_requests_per_run": 20,
            "max_response_bytes": 1024 * 1024,
            "max_attempts_per_url": 2,
            "base_backoff_seconds": 1,
            "robots_txt_mode": "written_permission_override",
        }
        path = root / "policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _manifest(self, root: Path, entries: list[dict[str, object]]) -> Path:
        path = root / "manifest.jsonl"
        path.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
        )
        return path

    def _capture(
        self,
        root: Path,
        *,
        stream: str,
        entries: list[dict[str, object]],
        bodies: list[bytes],
    ) -> None:
        manifest = self._manifest(root, entries)
        clock = FakeClock()
        capture_manifest(
            root / "state.sqlite3",
            manifest,
            root / "raw",
            stream=stream,
            policy_path=self._policy(root),
            opener=FakeOpener(
                [FakeResponse(entry["url"], body) for entry, body in zip(entries, bodies)]
            ),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )

    def _results_body(
        self,
        groups: list[tuple[str, list[str]]],
        *,
        pagination_urls: list[str] | None = None,
    ) -> bytes:
        sublists = []
        for heading, match_ids in groups:
            cards = "".join(
                "<div class='result-con'><a href='/matches/"
                + match_id
                + "/fixture'>fixture</a></div>"
                for match_id in match_ids
            )
            sublists.append(
                "<div class='results-sublist'><span class='standard-headline'>"
                + heading
                + "</span>"
                + cards
                + "</div>"
            )
        pagination = "".join(
            "<a class='pagination-next' href='" + url + "'>next</a>"
            for url in pagination_urls or []
        )
        return (
            "<html><body><div class='results-holder'>"
            + "".join(sublists)
            + "</div>"
            + pagination
            + "</body></html>"
        ).encode("utf-8")

    def test_parse_results_extracts_only_result_cards_and_pagination(self) -> None:
        listing = parse_results_html(
            (FIXTURES / "hltv_results.html").read_text(encoding="utf-8"),
            source_url="https://www.hltv.org/results",
        )

        self.assertEqual(
            (
                (
                    "9001",
                    "https://www.hltv.org/matches/9001/alpha-vs-beta",
                    1,
                    "2024-01-02",
                ),
                (
                    "9002",
                    "https://www.hltv.org/matches/9002/gamma-vs-delta",
                    2,
                    "2024-01-02",
                ),
            ),
            tuple(
                (item.match_id, item.url, item.card_position, item.listed_date.isoformat())
                for item in listing.matches
            ),
        )
        self.assertEqual(
            (
                "https://www.hltv.org/results?offset=100",
                "https://www.hltv.org/results?offset=0",
            ),
            listing.pagination_urls,
        )

    def test_result_manifest_is_bounded_and_date_partitioned(self) -> None:
        records = generate_results_manifest(
            date(2018, 1, 1),
            date(2018, 1, 10),
            window_days=7,
            url_template=(
                "https://www.hltv.org/results?startDate={start_date}&endDate={end_date}"
            ),
        )

        self.assertEqual(2, len(records))
        self.assertEqual(
            "https://www.hltv.org/results?startDate=2018-01-01&endDate=2018-01-07",
            records[0]["url"],
        )
        self.assertEqual("hltv-results-root", records[0]["discovery"]["kind"])
        self.assertEqual("2018-01-08", records[1]["discovery"]["window_start"])
        with self.assertRaises(ValueError):
            generate_results_manifest(
                date(2018, 1, 2),
                date(2018, 1, 1),
                window_days=1,
                url_template="https://www.hltv.org/results?startDate={start_date}",
            )
        with self.assertRaises(ValueError):
            generate_results_manifest(
                date(2018, 1, 1),
                date(2018, 1, 1),
                window_days=1,
                url_template=(
                    "https://www.hltv.org/results?startDate={start_date}&"
                    "endDate={end_date}&offset=100"
                ),
            )

    def test_results_cards_keep_their_own_date_group(self) -> None:
        listing = parse_results_html(
            self._results_body(
                [
                    ("Results for January 2, 2024", ["9001"]),
                    ("Results for January 3rd 2024", ["9002"]),
                ]
            ).decode("utf-8"),
            source_url="https://www.hltv.org/results",
        )

        self.assertEqual(
            [("9001", "2024-01-02"), ("9002", "2024-01-03")],
            [(item.match_id, item.listed_date.isoformat()) for item in listing.matches],
        )

    def test_inactive_pagination_control_without_href_is_not_a_graph_edge(self) -> None:
        listing = parse_results_html(
            """
            <div class='results-holder'></div>
            <a class='pagination-prev inactive'>previous</a>
            <a class='pagination-next' href='/results?offset=100'>next</a>
            """,
            source_url="https://www.hltv.org/results",
        )

        self.assertEqual(
            ("https://www.hltv.org/results?offset=100",), listing.pagination_urls
        )

    def test_date_window_rejects_pagination_that_drops_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02"
            self._capture(
                root,
                stream="results",
                entries=[
                    {
                        "record_id": "windowed-results",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "window_start": "2024-01-02",
                            "window_end": "2024-01-02",
                        },
                    }
                ],
                bodies=[(FIXTURES / "hltv_results.html").read_bytes()],
            )

            with self.assertRaises(HltvDiscoveryError):
                extract_match_manifest(root / "state.sqlite3", stream="results")

    def test_date_window_rejects_content_outside_the_window_even_when_url_keeps_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2018-01-01&endDate=2018-01-07"
            self._capture(
                root,
                stream="results",
                entries=[
                    {
                        "record_id": "windowed-results",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "window_start": "2018-01-01",
                            "window_end": "2018-01-07",
                        },
                    }
                ],
                bodies=[self._results_body([("Results for January 2, 2024", ["9001"])])],
            )

            with self.assertRaises(HltvDiscoveryError):
                extract_match_manifest(root / "state.sqlite3", stream="results")

    def test_extract_match_manifest_preserves_listing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02"
            body = (
                (FIXTURES / "hltv_results.html")
                .read_text(encoding="utf-8")
                .replace(
                    "/results?offset=100",
                    "/results?startDate=2024-01-02&amp;endDate=2024-01-02&amp;offset=100",
                )
                .replace(
                    "/results?offset=0",
                    "/results?startDate=2024-01-02&amp;endDate=2024-01-02&amp;offset=0",
                )
                .encode("utf-8")
            )
            self._capture(
                root,
                stream="results",
                entries=[
                    {
                        "record_id": "result-page",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "window_start": "2024-01-02",
                            "window_end": "2024-01-02",
                        },
                    }
                ],
                bodies=[body],
            )

            records, report = extract_match_manifest(root / "state.sqlite3", stream="results")

            self.assertEqual(["hltv-match:9001", "hltv-match:9002"], [item["record_id"] for item in records])
            self.assertEqual(
                "result-page",
                records[0]["discovery"][0]["listing_capture_record_id"],
            )
            self.assertEqual("2024-01-02", records[0]["discovery"][0]["listed_date"])
            self.assertFalse(report["coverage_complete"])
            self.assertEqual(
                [
                    "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02&offset=100"
                ],
                report["unfetched_pagination_urls"],
            )

    def test_date_window_listing_cannot_silently_redirect_to_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = (
                "https://www.hltv.org/results?startDate=2018-01-01&endDate=2018-01-07"
            )
            manifest = self._manifest(
                root,
                [
                    {
                        "record_id": "historical-results",
                        "page_type": "results",
                        "url": requested,
                        "discovery": {
                            "window_start": "2018-01-01",
                            "window_end": "2018-01-07",
                        },
                    }
                ],
            )
            capture_manifest(
                root / "state.sqlite3",
                manifest,
                root / "raw",
                stream="results",
                policy_path=self._policy(root),
                opener=FakeOpener(
                    [
                        FakeResponse(
                            "https://www.hltv.org/results",
                            (FIXTURES / "hltv_results.html").read_bytes(),
                        )
                    ]
                ),
                now_fn=FakeClock().now,
                sleep_fn=FakeClock().sleep,
            )

            with self.assertRaises(HltvDiscoveryError):
                extract_match_manifest(root / "state.sqlite3", stream="results")

    def test_pagination_cascade_aggregates_only_verified_parent_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_url = "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02"
            child_url = root_url + "&offset=100"
            root_entry = {
                "record_id": "root-page",
                "page_type": "results",
                "url": root_url,
                "discovery": {
                    "kind": "hltv-results-root",
                    "window_start": "2024-01-02",
                    "window_end": "2024-01-02",
                },
            }
            self._capture(
                root,
                stream="results-root",
                entries=[root_entry],
                bodies=[
                    self._results_body(
                        [("Results for January 2, 2024", ["9001"])],
                        pagination_urls=[child_url],
                    )
                ],
            )

            first_child, first_report = derive_results_pagination_manifest(
                root / "state.sqlite3", streams=["results-root"]
            )
            self.assertEqual(1, len(first_child))
            self.assertFalse(first_report["coverage_complete"])
            self.assertEqual(child_url, first_child[0]["url"])
            self._capture(
                root,
                stream="results-page-1",
                entries=first_child,
                bodies=[
                    self._results_body(
                        [("Results for January 2nd 2024", ["9002"])],
                        pagination_urls=[root_url],
                    )
                ],
            )

            next_child, next_report = derive_results_pagination_manifest(
                root / "state.sqlite3",
                streams=["results-root", "results-page-1"],
            )
            self.assertEqual([], next_child)
            self.assertTrue(next_report["coverage_complete"])
            records, report = aggregate_match_manifest(
                root / "state.sqlite3",
                streams=["results-root", "results-page-1"],
                expected_start=date(2024, 1, 2),
                expected_end=date(2024, 1, 2),
            )
            self.assertTrue(report["coverage_complete"])
            self.assertEqual(
                ["hltv-match:9001", "hltv-match:9002"],
                [record["record_id"] for record in records],
            )

    def test_aggregate_refuses_partial_listing_coverage_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02"
            child_url = url + "&offset=100"
            self._capture(
                root,
                stream="results-root",
                entries=[
                    {
                        "record_id": "root-page",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "kind": "hltv-results-root",
                            "window_start": "2024-01-02",
                            "window_end": "2024-01-02",
                        },
                    }
                ],
                bodies=[
                    self._results_body(
                        [("Results for January 2, 2024", ["9001"])],
                        pagination_urls=[child_url],
                    )
                ],
            )

            with self.assertRaises(HltvDiscoveryError):
                aggregate_match_manifest(
                    root / "state.sqlite3",
                    streams=["results-root"],
                    expected_start=date(2024, 1, 2),
                    expected_end=date(2024, 1, 2),
                )
            _, report = aggregate_match_manifest(
                root / "state.sqlite3",
                streams=["results-root"],
                expected_start=date(2024, 1, 2),
                expected_end=date(2024, 1, 2),
                require_complete=False,
            )
            self.assertFalse(report["coverage_complete"])

    def test_aggregate_refuses_root_windows_outside_requested_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2024-01-01&endDate=2024-01-03"
            self._capture(
                root,
                stream="results-root",
                entries=[
                    {
                        "record_id": "root-page",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "kind": "hltv-results-root",
                            "window_start": "2024-01-01",
                            "window_end": "2024-01-03",
                        },
                    }
                ],
                bodies=[self._results_body([("Results for January 2, 2024", ["9001"])])],
            )

            with self.assertRaises(HltvDiscoveryError):
                aggregate_match_manifest(
                    root / "state.sqlite3",
                    streams=["results-root"],
                    expected_start=date(2024, 1, 2),
                    expected_end=date(2024, 1, 2),
                    require_complete=False,
                )

    def test_aggregate_rejects_legacy_untyped_root_listing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/results?startDate=2024-01-02&endDate=2024-01-02"
            self._capture(
                root,
                stream="results-root",
                entries=[
                    {
                        "record_id": "legacy-root-page",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "window_start": "2024-01-02",
                            "window_end": "2024-01-02",
                        },
                    }
                ],
                bodies=[self._results_body([])],
            )

            with self.assertRaises(HltvDiscoveryError):
                aggregate_match_manifest(
                    root / "state.sqlite3",
                    streams=["results-root"],
                    expected_start=date(2024, 1, 2),
                    expected_end=date(2024, 1, 2),
                )

    def test_aggregate_rejects_nonzero_offset_declared_as_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = (
                "https://www.hltv.org/results?startDate=2024-01-02&"
                "endDate=2024-01-02&offset=100"
            )
            self._capture(
                root,
                stream="results-root",
                entries=[
                    {
                        "record_id": "wrong-root-page",
                        "page_type": "results",
                        "url": url,
                        "discovery": {
                            "kind": "hltv-results-root",
                            "window_start": "2024-01-02",
                            "window_end": "2024-01-02",
                        },
                    }
                ],
                bodies=[self._results_body([])],
            )

            with self.assertRaises(HltvDiscoveryError):
                aggregate_match_manifest(
                    root / "state.sqlite3",
                    streams=["results-root"],
                    expected_start=date(2024, 1, 2),
                    expected_end=date(2024, 1, 2),
                )

    def test_extract_mapstats_manifest_uses_exact_link_from_match_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/9001/alpha-vs-beta"
            self._capture(
                root,
                stream="matches",
                entries=[
                    {
                        "record_id": "match-page",
                        "page_type": "match",
                        "url": url,
                        "discovery": [{"listing_content_sha256": "parent-listing"}],
                    }
                ],
                bodies=[(FIXTURES / "hltv_match.html").read_bytes()],
            )

            records, report = extract_mapstats_manifest(root / "state.sqlite3", stream="matches")

            self.assertEqual(1, len(records))
            self.assertEqual("hltv-map-stats:501", records[0]["record_id"])
            self.assertEqual(
                "https://www.hltv.org/stats/matches/mapstatsid/501/alpha-vs-beta",
                records[0]["url"],
            )
            self.assertEqual("9001", records[0]["discovery"][0]["parent_match_id"])
            self.assertEqual(
                "parent-listing",
                records[0]["discovery"][0]["parent_capture"]["manifest_metadata"]["discovery"][0]["listing_content_sha256"],
            )
            self.assertEqual([], report["missing_played_map_stats"])

    def test_mapstats_manifest_refuses_a_played_map_without_exact_stats_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.hltv.org/matches/9001/alpha-vs-beta"
            body = (FIXTURES / "hltv_match.html").read_text(encoding="utf-8").replace(
                'href="/stats/matches/mapstatsid/501/alpha-vs-beta"',
                'href="/events/77/example-cup"',
            )
            self._capture(
                root,
                stream="matches",
                entries=[{"record_id": "match-page", "page_type": "match", "url": url}],
                bodies=[body.encode("utf-8")],
            )

            with self.assertRaises(HltvDiscoveryError):
                extract_mapstats_manifest(root / "state.sqlite3", stream="matches")
            records, report = extract_mapstats_manifest(
                root / "state.sqlite3", stream="matches", require_complete=False
            )
            self.assertEqual([], records)
            self.assertEqual(1, len(report["missing_played_map_stats"]))

    def test_mapstats_manifest_refuses_one_stats_id_from_conflicting_parent_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_url = "https://www.hltv.org/matches/9001/alpha-vs-beta"
            second_url = "https://www.hltv.org/matches/9002/alpha-vs-beta"
            first_body = (FIXTURES / "hltv_match.html").read_bytes()
            second_body = first_body.replace(
                b"/matches/9001/alpha-vs-beta", b"/matches/9002/alpha-vs-beta"
            )
            self._capture(
                root,
                stream="matches",
                entries=[
                    {"record_id": "match-9001", "page_type": "match", "url": first_url},
                    {"record_id": "match-9002", "page_type": "match", "url": second_url},
                ],
                bodies=[first_body, second_body],
            )

            with self.assertRaises(HltvDiscoveryError):
                extract_mapstats_manifest(root / "state.sqlite3", stream="matches")

    def test_results_soft_block_is_not_treated_as_empty_listing(self) -> None:
        with self.assertRaises(HltvDiscoveryError):
            parse_results_html(
                "<html><body>Access denied</body></html>",
                source_url="https://www.hltv.org/results",
            )


if __name__ == "__main__":
    unittest.main()
