from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from predic_v2.bo3_capture import (
    Bo3SourceChangedError,
    audit_bo3_capture,
    bo3_capture_index,
    capture_bo3,
    plan_bo3_capture,
    reprocess_bo3_game_snapshots,
    reprocess_bo3_player_snapshots,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class _Response:
    def __init__(self, url: str, payload: object, status: int = 200) -> None:
        self.url = url
        self.status = status
        self.body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.stream = io.BytesIO(self.body)
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(self.body)),
        }

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url


class _Opener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def open(self, request, timeout: float):
        url = request.full_url
        self.urls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(url)
        if isinstance(item, _Response):
            item.url = url
        return item


def _catalog() -> dict[str, object]:
    return {
        "total": {"count": 1, "pages": 1, "offset": 0, "limit": 100},
        "results": [
            {
                "id": 1,
                "slug": "alpha-vs-beta-15-06-2020",
                "status": "finished",
                "parsed_status": "done",
                "start_date": "2020-06-15T12:00:00.000+00:00",
                "end_date": "2020-06-15T13:00:00.000+00:00",
                "bo_type": 1,
                "game_version": 1,
                "team1_id": 10,
                "team2_id": 20,
                "games": [
                    {
                        "id": 101,
                        "number": 1,
                        "map_name": "de_mirage",
                        "status": "finished",
                        "rounds_count": 2,
                    }
                ],
            }
        ],
    }


def _match() -> dict[str, object]:
    return dict(_catalog()["results"][0])


def _game() -> dict[str, object]:
    return {
        "id": 101,
        "match_id": 1,
        "map_name": "de_mirage",
        "status": "finished",
        "rounds_count": 2,
        "demo_url": "demos/example.dem",
        "game_rounds": [
            {"round_number": 1, "winner_clan_score": 1, "loser_clan_score": 0},
            {"round_number": 2, "winner_clan_score": 1, "loser_clan_score": 1},
        ],
    }


def _players(count: int = 10) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(count):
        team_id = 10 if index < 5 else 20
        result.append(
            {
                "steam_profile_id": 1000 + index,
                "steam_profile": {
                    "nickname": f"p{index}",
                    "steam_id_64": str(76561198000000000 + index),
                    "game_round_steam_profiles": [
                        {"steam_profile_id": 1000 + index, "round_number": 1},
                        {"steam_profile_id": 1000 + index, "round_number": 2},
                    ],
                },
                "team_clan": {"team_id": team_id},
                "kills": 10 + index,
                "death": 8,
                "assists": 3,
                "damage": 900,
                "adr": 75.0,
                "kast": 0.75,
            }
        )
    return result


class Bo3CaptureTest(unittest.TestCase):
    def _policy(self, root: Path, *, live: bool = True) -> Path:
        policy = {
            "live_enabled": live,
            "authorization_ref": "bo3-test-permission",
            "authorization_scope": "api.bo3.gg private ML research test",
            "authorization_confirmed_at": "2026-08-26T00:00:00Z" if live else None,
            "valid_until": None,
            "allowed_schemes": ["https"],
            "allowed_hosts": ["api.bo3.gg"],
            "allowed_path_prefixes": ["/api/v1"],
            "allowed_query_keys": [
                "scope",
                "page[offset]",
                "page[limit]",
                "sort",
                "filter[matches.status][in]",
                "filter[matches.start_date][gt]",
                "filter[matches.start_date][lt]",
                "filter[matches.discipline_id][eq]",
                "with",
            ],
            "user_agent": "PredicTest/1.0 (contact: test@example.invalid)",
            "contact": "test@example.invalid",
            "min_interval_seconds": 1,
            "max_pages_per_run": 100,
            "max_http_requests_per_run": 100,
            "max_response_bytes": 1024 * 1024,
            "max_attempts_per_url": 3,
            "base_backoff_seconds": 5,
            "robots_txt_mode": "written_permission_override",
        }
        path = root / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def _capture(
        self,
        root: Path,
        opener: _Opener,
        clock: _Clock,
        *,
        max_requests: int = 4,
        stream: str = "history",
        profile: str = "core",
        continue_on_quality_error: bool = False,
        continue_on_network_error: bool = False,
        quarantine_incomplete: bool = False,
    ) -> dict[str, object]:
        return capture_bo3(
            root / "state.sqlite3",
            root / "raw",
            stream=stream,
            policy_path=self._policy(root),
            start_date=date(2020, 6, 15),
            end_date=date(2020, 6, 16),
            profile=profile,
            max_requests=max_requests,
            continue_on_quality_error=continue_on_quality_error,
            continue_on_network_error=continue_on_network_error,
            quarantine_incomplete=quarantine_incomplete,
            opener=opener,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )

    def test_plan_is_offline_and_accepts_disabled_example_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = plan_bo3_capture(
                self._policy(root, live=False),
                start_date=date(2020, 6, 15),
                end_date=date(2020, 7, 1),
                profile="core",
            )
            self.assertEqual(3, result["initial_catalog_requests"])
            self.assertTrue(result["written_authorization_required_for_live"])
            self.assertFalse((root / "state.sqlite3").exists())

    def test_core_capture_is_resumable_and_closes_player_map_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            opener = _Opener(
                [
                    lambda url: _Response(url, _catalog()),
                    lambda url: _Response(url, _players()),
                    lambda url: _Response(url, _match()),
                    lambda url: _Response(url, _game()),
                ]
            )

            result = self._capture(root, opener, clock)

            self.assertTrue(result["ok"])
            self.assertEqual(4, result["requests_this_run"])
            self.assertEqual(10, result["totals"]["player_maps"])
            self.assertEqual(0, result["gaps"]["finished_game_player_gap_count"])
            self.assertEqual([1.0, 1.0, 1.0], clock.sleeps)
            self.assertIn(
                "filter%5Bmatches.start_date%5D%5Bgt%5D",
                opener.urls[0],
            )
            self.assertNotIn("%5Bgte%5D", opener.urls[0])
            index = list(bo3_capture_index(root / "state.sqlite3", stream="history"))
            self.assertEqual(4, len(index))
            for record in index:
                object_path = root / "raw" / str(record["object_path"])
                self.assertTrue(object_path.is_file())

            replay = self._capture(root, _Opener([]), clock, max_requests=1)
            self.assertEqual(0, replay["requests_this_run"])
            self.assertTrue(replay["ok"])

    def test_upcoming_placeholder_maps_do_not_create_fake_player_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            upcoming = _catalog()
            match = upcoming["results"][0]
            match["status"] = "upcoming"
            match["parsed_status"] = None
            match["end_date"] = None
            match["games"] = [
                {
                    "id": 101,
                    "number": 1,
                    "map_name": None,
                    "status": "upcoming",
                    "rounds_count": None,
                }
            ]
            detail = dict(match)
            result = capture_bo3(
                root / "state.sqlite3",
                root / "raw",
                stream="upcoming-snapshot",
                policy_path=self._policy(root),
                start_date=date(2020, 6, 15),
                end_date=date(2020, 6, 16),
                statuses=("upcoming",),
                profile="core",
                max_requests=2,
                opener=_Opener(
                    [
                        lambda url: _Response(url, upcoming),
                        lambda url: _Response(url, detail),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(1, result["totals"]["games"])
            self.assertEqual(0, result["totals"]["player_maps"])
            self.assertNotIn("game:pending", result["task_counts"])

    def test_partial_roster_is_retained_once_and_visible_as_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            opener = _Opener(
                [
                    lambda url: _Response(url, _catalog()),
                    lambda url: _Response(url, _players(9)),
                ]
            )

            result = self._capture(root, opener, clock, max_requests=2)

            self.assertFalse(result["ok"])
            self.assertIsNone(result["stopped_reason"])
            self.assertEqual(1, result["gaps"]["finished_game_player_gap_count"])
            self.assertEqual(9, result["totals"]["player_maps"])
            self.assertIn("game_players:complete", result["task_counts"])
            self.assertIn(
                "incomplete lineup",
                result["incomplete_game_samples"][0]["player_quality_error"],
            )
            self.assertEqual(
                "partial_roster",
                result["incomplete_game_samples"][0]["player_quality_class"],
            )

    def test_bulk_mode_keeps_incomplete_player_map_visible_without_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players(9)),
                    ]
                ),
                clock,
                max_requests=2,
                continue_on_quality_error=True,
            )

            self.assertIsNone(result["stopped_reason"])
            self.assertFalse(result["ok"])
            self.assertIn("game_players:complete", result["task_counts"])
            self.assertEqual(1, result["gaps"]["finished_game_player_gap_count"])

    def test_historical_bulk_mode_quarantines_incomplete_payload_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, []),
                    ]
                ),
                clock,
                max_requests=2,
                profile="training",
                continue_on_quality_error=True,
                quarantine_incomplete=True,
            )

            self.assertIsNone(result["stopped_reason"])
            self.assertIn("game_players:quarantined", result["task_counts"])
            self.assertNotIn("game_players:retry", result["task_counts"])
            replay = self._capture(
                root,
                _Opener([]),
                clock,
                max_requests=1,
                profile="training",
                quarantine_incomplete=True,
            )
            self.assertEqual(0, replay["requests_this_run"])

    def test_kast_missing_is_masked_without_rejecting_complete_lineup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            players = _players()
            players[0]["kast"] = None
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, players),
                    ]
                ),
                clock,
                max_requests=2,
                profile="training",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(0, result["gaps"]["finished_game_player_gap_count"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                game = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                player = connection.execute(
                    """
                    SELECT * FROM bo3_player_map_index
                    WHERE game_id = 101 AND steam_profile_id = 1000
                    """
                ).fetchone()
                self.assertEqual("complete_5v5", game["player_quality_class"])
                self.assertEqual(1, game["kast_missing_rows"])
                self.assertEqual(1, game["players_complete"])
                self.assertEqual(0, player["metrics_complete"])
                self.assertEqual(1, player["training_metrics_complete"])
                self.assertEqual('["kast"]', player["missing_metrics_json"])
            finally:
                connection.close()

    def test_substitutions_keep_all_participants_and_round_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            players = _players(12)
            players[0]["steam_profile"]["player"] = {"is_coach": True}
            for player in players[-2:]:
                player["steam_profile"]["game_round_steam_profiles"] = [
                    {
                        "steam_profile_id": player["steam_profile_id"],
                        "round_number": 2,
                    }
                ]
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, players),
                    ]
                ),
                clock,
                max_requests=2,
                profile="training",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(12, result["totals"]["player_maps"])
            self.assertEqual(0, result["gaps"]["finished_game_player_gap_count"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                game = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                substitute = connection.execute(
                    """
                    SELECT * FROM bo3_player_map_index
                    WHERE game_id = 101 AND steam_profile_id = 1011
                    """
                ).fetchone()
                current_coach = connection.execute(
                    """
                    SELECT current_is_coach FROM bo3_player_map_index
                    WHERE game_id = 101 AND steam_profile_id = 1000
                    """
                ).fetchone()
                self.assertEqual("substitution", game["player_quality_class"])
                self.assertEqual(1, game["lineup_complete"])
                self.assertEqual(1, current_coach["current_is_coach"])
                self.assertEqual(1, substitute["rounds_participated"])
                self.assertEqual(2, substitute["first_round"])
                self.assertEqual(0.5, substitute["participation_fraction"])
            finally:
                connection.close()

            replay = reprocess_bo3_player_snapshots(
                root / "state.sqlite3", stream="history"
            )
            self.assertEqual(1, replay["processed"])
            self.assertEqual(1, replay["accepted"])
            self.assertEqual({"substitution": 1}, replay["quality_classes"])
            self.assertEqual(0, replay["raw_objects_modified"])

    def test_partial_game_rounds_are_masked_without_rejecting_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            game = _game()
            game["rounds_count"] = 3
            game["game_rounds"] = [
                {"round_number": 1},
                {"round_number": 3},
            ]
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players()),
                        lambda url: _Response(url, _match()),
                        lambda url: _Response(url, game),
                    ]
                ),
                clock,
            )

            self.assertIsNone(result["stopped_reason"])
            self.assertIn("game:complete", result["task_counts"])
            self.assertEqual(0, result["gaps"]["finished_game_detail_gap_count"])
            self.assertEqual(1, result["gaps"]["finished_game_round_gap_count"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                self.assertEqual("partial_rounds", row["game_quality_class"])
                self.assertEqual(1, row["game_detail_complete"])
                self.assertEqual(0, row["rounds_complete"])
                self.assertAlmostEqual(2 / 3, row["rounds_coverage"])
                self.assertEqual("[2]", row["missing_rounds_json"])
            finally:
                connection.close()

            replay = reprocess_bo3_game_snapshots(
                root / "state.sqlite3", stream="history"
            )
            self.assertEqual(1, replay["processed"])
            self.assertEqual(1, replay["accepted"])
            self.assertEqual({"partial_rounds": 1}, replay["quality_classes"])

    def test_contiguous_rounds_above_declared_count_are_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            game = _game()
            game["game_rounds"].append({"round_number": 3})
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players()),
                        lambda url: _Response(url, _match()),
                        lambda url: _Response(url, game),
                    ]
                ),
                clock,
            )

            self.assertTrue(result["ok"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                self.assertEqual(
                    "declared_round_count_stale", row["game_quality_class"]
                )
                self.assertEqual(1, row["rounds_complete"])
                self.assertEqual("[3]", row["unexpected_rounds_json"])
            finally:
                connection.close()

    def test_duplicate_game_rounds_remain_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            game = _game()
            game["game_rounds"].append({"round_number": 2})
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players()),
                        lambda url: _Response(url, _match()),
                        lambda url: _Response(url, game),
                    ]
                ),
                clock,
                continue_on_quality_error=True,
                quarantine_incomplete=True,
            )

            self.assertIn("game:quarantined", result["task_counts"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                self.assertEqual("duplicate_rounds", row["game_quality_class"])
                self.assertEqual(0, row["game_detail_complete"])
            finally:
                connection.close()

    def test_empty_game_rounds_never_claim_full_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            game = _game()
            game["rounds_count"] = None
            game["game_rounds"] = []
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players()),
                        lambda url: _Response(url, _match()),
                        lambda url: _Response(url, game),
                    ]
                ),
                clock,
            )

            self.assertEqual(1, result["gaps"]["finished_game_round_gap_count"])
            connection = sqlite3.connect(root / "state.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT * FROM bo3_game_index WHERE game_id = 101"
                ).fetchone()
                self.assertEqual("empty_rounds", row["game_quality_class"])
                self.assertEqual(0, row["rounds_complete"])
                self.assertEqual(0.0, row["rounds_coverage"])
                self.assertEqual("[1,2]", row["missing_rounds_json"])
            finally:
                connection.close()

    def test_unattended_mode_leaves_network_failure_for_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            result = self._capture(
                root,
                _Opener([URLError("temporary disconnect")]),
                clock,
                max_requests=1,
                continue_on_network_error=True,
            )

            self.assertIsNone(result["stopped_reason"])
            self.assertIn("catalog:retry", result["task_counts"])

    def test_parallel_workers_share_one_persisted_start_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            empty_catalog = {
                "total": {"count": 0, "pages": 1, "offset": 0, "limit": 100},
                "results": [],
            }
            result = capture_bo3(
                root / "state.sqlite3",
                root / "raw",
                stream="parallel-catalog",
                policy_path=self._policy(root),
                start_date=date(2020, 6, 15),
                end_date=date(2020, 6, 18),
                window_days=1,
                profile="catalog",
                max_requests=3,
                workers=3,
                opener=_Opener(
                    [
                        lambda url: _Response(url, empty_catalog),
                        lambda url: _Response(url, empty_catalog),
                        lambda url: _Response(url, empty_catalog),
                    ]
                ),
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(3, result["requests_this_run"])
            self.assertEqual(3, result["workers"])
            self.assertEqual([1.0, 1.0], clock.sleeps)

    def test_partial_request_budget_resumes_from_child_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            first = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _catalog()),
                        lambda url: _Response(url, _players()),
                    ]
                ),
                clock,
                max_requests=2,
            )

            self.assertFalse(first["ok"])
            self.assertEqual(2, first["requests_this_run"])
            self.assertEqual(10, first["totals"]["player_maps"])
            self.assertIn("match:pending", first["task_counts"])
            self.assertIn("game:pending", first["task_counts"])

            second = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, _match()),
                        lambda url: _Response(url, _game()),
                    ]
                ),
                clock,
                max_requests=2,
            )

            self.assertTrue(second["ok"])
            self.assertEqual(2, second["requests_this_run"])
            self.assertEqual(10, second["totals"]["player_maps"])

    def test_training_profile_closes_players_without_detail_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            catalog = _catalog()
            catalog["results"][0]["status"] = None
            catalog["results"][0]["parsed_status"] = None
            result = self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(url, catalog),
                        lambda url: _Response(url, _players()),
                    ]
                ),
                clock,
                max_requests=2,
                profile="training",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(10, result["totals"]["player_maps"])
            self.assertNotIn("match:pending", result["task_counts"])
            self.assertNotIn("game:pending", result["task_counts"])

    def test_resume_backfills_player_first_tasks_for_old_core_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            self._capture(
                root,
                _Opener([lambda url: _Response(url, _catalog())]),
                clock,
                max_requests=1,
            )
            with sqlite3.connect(root / "state.sqlite3") as connection:
                connection.execute(
                    "DELETE FROM bo3_task WHERE kind IN ('game', 'game_players')"
                )
                connection.execute(
                    """
                    UPDATE bo3_game_index
                    SET stats_expected = 0, status = NULL, rounds_count = NULL
                    """
                )

            result = self._capture(
                root,
                _Opener([lambda url: _Response(url, _players())]),
                clock,
                max_requests=1,
            )

            self.assertEqual(10, result["totals"]["player_maps"])
            self.assertIn("game:pending", result["task_counts"])
            self.assertNotIn("game_players:pending", result["task_counts"])

    def test_stream_configuration_cannot_change_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            self._capture(
                root,
                _Opener(
                    [
                        lambda url: _Response(
                            url,
                            {
                                "total": {
                                    "count": 0,
                                    "pages": 1,
                                    "offset": 0,
                                    "limit": 100,
                                },
                                "results": [],
                            },
                        )
                    ]
                ),
                clock,
                max_requests=1,
            )

            with self.assertRaises(Bo3SourceChangedError):
                capture_bo3(
                    root / "state.sqlite3",
                    root / "raw",
                    stream="history",
                    policy_path=self._policy(root),
                    start_date=date(2020, 6, 15),
                    end_date=date(2020, 6, 17),
                    max_requests=1,
                    opener=_Opener([]),
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )

    def test_429_is_persisted_without_consuming_more_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            error = HTTPError(
                "https://api.bo3.gg/api/v1/matches",
                429,
                "rate limited",
                {"Retry-After": "120"},
                None,
            )
            result = self._capture(root, _Opener([error]), clock, max_requests=5)

            self.assertEqual("http_429", result["stopped_reason"])
            self.assertEqual(1, result["requests_this_run"])
            self.assertIn("catalog:retry", result["task_counts"])
            audit = audit_bo3_capture(root / "state.sqlite3", stream="history")
            self.assertFalse(audit["ok"])

    def test_403_opens_persistent_host_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = _Clock()
            error = HTTPError(
                "https://api.bo3.gg/api/v1/matches",
                403,
                "forbidden",
                {},
                None,
            )
            first = self._capture(root, _Opener([error]), clock, max_requests=5)

            self.assertEqual("http_403", first["stopped_reason"])
            self.assertEqual(1, first["requests_this_run"])

            second = self._capture(root, _Opener([]), clock, max_requests=5)
            self.assertEqual("http_403", second["stopped_reason"])
            self.assertEqual(0, second["requests_this_run"])


if __name__ == "__main__":
    unittest.main()
