from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import predic_v2.tedtay as tedtay
from predic_v2.audit import audit
from predic_v2.db import connect, initialize
from predic_v2.tedtay import (
    DATASET_URL,
    GAME_DATA_FIELDS,
    HISTORIC_FIELDS,
    LICENSE_SHA256,
    MISSING_GAME_DATA_KIND,
    SOURCE,
    UPSTREAM_COMMIT,
    TedTayImportError,
    import_tedtay_dataset,
)


def _link(mapstatsid: int) -> str:
    return f"/stats/matches/mapstatsid/{mapstatsid}/alpha-vs-beta"


def _historic_row(
    mapstatsid: int,
    *,
    when: datetime,
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
    map_name: str = "inf",
    event_name: str = "Fixture Event",
) -> dict[str, str]:
    when = when.astimezone(timezone.utc)
    return {
        "date_ymd": when.date().isoformat(),
        "date_unix_iso": when.strftime("%Y-%m-%d %H:%M:%S"),
        "date_unix": str(int(when.timestamp() * 1_000)),
        "team1": team_a,
        "team2": team_b,
        "team1_rounds": f"({score_a})",
        "team2_rounds": f"({score_b})",
        "map_name_short": map_name,
        "event_name": event_name,
        "game_link": _link(mapstatsid),
    }


def _stats_row(
    mapstatsid: int,
    *,
    first_score: int,
    second_score: int,
    first_names: list[str],
    second_names: list[str],
    index: int,
) -> dict[str, str]:
    assert len(first_names) == len(second_names) == 5
    row = {field: "" for field in GAME_DATA_FIELDS}
    row.update(
        {
            "Unnamed: 0": str(index),
            "team1_half1_t": str(first_score // 2),
            "team2_half1_ct": str(second_score // 2),
            "team1_half2_ct": str(first_score - first_score // 2),
            # This column is awkwardly named upstream but contains the second
            # stats table's second-half score.
            "team1_half2_t": str(second_score - second_score // 2),
            "team1_first_kills": "10",
            "team2_first_kills": "9",
            "team1_clutches_won": "1",
            "team2_clutches_won": "0",
            "game_link": _link(mapstatsid),
            "collected_timestamp": "2024-02-03 04:05:06.123456",
            "team2_half2_t": "",
        }
    )
    for team_slot, names in ((1, first_names), (2, second_names)):
        for player_slot, nickname in enumerate(names, start=1):
            kills = 20 + player_slot
            deaths = 10 + player_slot
            prefix = f"team{team_slot}_p{player_slot}_"
            row.update(
                {
                    f"{prefix}name": nickname,
                    f"{prefix}khs": f"{kills} ({10 + player_slot})",
                    f"{prefix}assists": "5 (2)",
                    f"{prefix}deaths": str(deaths),
                    f"{prefix}kast": "80.0%",
                    f"{prefix}kddiff": str(kills - deaths),
                    f"{prefix}adr": "83.4",
                    f"{prefix}fkdiff": "+1",
                    f"{prefix}game_rating": "1.05",
                }
            )
    return row


class TedTayImportTest(unittest.TestCase):
    def _database(self, root: Path):
        connection = connect(root / "test.sqlite3")
        initialize(connection)
        self.addCleanup(connection.close)
        return connection

    def _write_csv(self, path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def _dataset(self, root: Path) -> tuple[Path, Path]:
        historic = root / "historic_games_list.csv"
        stats = root / "game_data_rh.csv"
        first = _historic_row(
            1001,
            when=datetime(2021, 4, 2, 12, 30, tzinfo=timezone.utc),
            team_a="Alpha",
            team_b="Beta",
            score_a=16,
            score_b=10,
            event_name="Event Alpha",
        )
        second = _historic_row(
            1002,
            when=datetime(2020, 1, 3, 10, 0, tzinfo=timezone.utc),
            team_a="Gamma",
            team_b="Delta",
            score_a=13,
            score_b=16,
            event_name="Event Gamma",
        )
        unmatched = _historic_row(
            1003,
            when=datetime(2019, 1, 3, 10, 0, tzinfo=timezone.utc),
            team_a="Unused A",
            team_b="Unused B",
            score_a=16,
            score_b=8,
        )
        duplicate_first = dict(first)
        # First row per exact game_link wins; this deliberately malformed
        # later duplicate must neither become a map nor abort the source run.
        duplicate_first["team1_rounds"] = "not-a-score"
        self._write_csv(
            historic,
            HISTORIC_FIELDS,
            [first, second, unmatched, duplicate_first],
        )

        # The stats table order for map 1001 is intentionally opposite the
        # historic row.  Score reconciliation must attach Beta's block first.
        first_stats = _stats_row(
            1001,
            first_score=10,
            second_score=16,
            first_names=["shared", "beta-two", "beta-three", "beta-four", "beta-five"],
            second_names=["shared", "alpha-two", "alpha-three", "alpha-four", "alpha-five"],
            index=1,
        )
        second_stats = _stats_row(
            1002,
            first_score=13,
            second_score=16,
            first_names=["gamma-one", "gamma-two", "gamma-three", "gamma-four", "gamma-five"],
            second_names=["delta-one", "delta-two", "delta-three", "delta-four", "delta-five"],
            index=2,
        )
        unmatched_stats = _stats_row(
            1999,
            first_score=16,
            second_score=6,
            first_names=["u1", "u2", "u3", "u4", "u5"],
            second_names=["v1", "v2", "v3", "v4", "v5"],
            index=3,
        )
        duplicate_stats = dict(first_stats)
        duplicate_stats["team1_p1_khs"] = "not-a-pair"
        self._write_csv(
            stats,
            GAME_DATA_FIELDS,
            [first_stats, second_stats, unmatched_stats, duplicate_stats],
        )
        return historic, stats

    def test_imports_audited_low_confidence_map_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)

            result = import_tedtay_dataset(connection, historic, stats, batch_size=1)

            self.assertEqual(SOURCE, result["source"])
            self.assertEqual(3, result["historic_first_rows"])
            self.assertEqual(1, result["historic_duplicate_links_ignored"])
            self.assertEqual(3, result["stats_first_rows"])
            self.assertEqual(1, result["stats_duplicate_links_ignored"])
            self.assertEqual(2, result["joined_rows"])
            self.assertEqual(3, result["eligible_rows"])
            self.assertEqual(2, result["eligible_joined_rows"])
            self.assertEqual(1, result["eligible_unmatched_historic_rows"])
            self.assertEqual(1, result["unmatched_historic_rows"])
            self.assertEqual(1, result["unmatched_stats_rows"])
            self.assertEqual(3, result["batches_committed"])
            self.assertEqual(
                {
                    "teams": 6,
                    "players": 20,
                    "series": 3,
                    "series_participants": 6,
                    "maps": 3,
                    "lineup_members": 20,
                    "player_map_stats": 20,
                    "ambiguous_team_binding_raw_records": 0,
                    "missing_game_data_rh_raw_records": 1,
                    "quarantine_raw_records": 1,
                },
                result["inserted"],
            )

            snapshot = connection.execute(
                "SELECT * FROM source_snapshot WHERE snapshot_id = ?",
                (result["source_snapshot_id"],),
            ).fetchone()
            self.assertEqual(0, snapshot["point_in_time_eligible"])
            self.assertEqual(DATASET_URL, snapshot["source_locator"])
            self.assertIn("Repository-declared MIT", snapshot["license_ref"])
            self.assertIn("HLTV", snapshot["license_ref"])
            metadata = json.loads(snapshot["metadata_json"])
            self.assertEqual(DATASET_URL, metadata["dataset_url"])
            self.assertEqual("research_bootstrap_only", metadata["import_scope"])
            self.assertEqual(UPSTREAM_COMMIT, snapshot["source_revision"])
            self.assertEqual(UPSTREAM_COMMIT, metadata["upstream_commit"])
            self.assertEqual("MIT", metadata["repository_declared_license"])
            self.assertFalse(metadata["dataset_rights_verified"])
            self.assertEqual(LICENSE_SHA256, metadata["license_sha256"])

            game = connection.execute(
                "SELECT * FROM map_game WHERE map_id = 'tedtay:map:1001'"
            ).fetchone()
            self.assertEqual("1001", game["source_map_id"])
            self.assertEqual("tedtay:series:1001", game["series_id"])
            self.assertEqual(16, game["score_a"])
            self.assertEqual(10, game["score_b"])
            self.assertIsNone(game["known_at"])
            self.assertEqual("CSGO", game["game_version"])
            self.assertEqual("UNKNOWN", game["ruleset"])

            self.assertEqual(
                2,
                connection.execute(
                    "SELECT COUNT(*) FROM series_participant WHERE series_id = ?",
                    (game["series_id"],),
                ).fetchone()[0],
            )
            self.assertEqual(
                5,
                connection.execute(
                    """
                    SELECT COUNT(*) FROM lineup_member
                    WHERE map_id = ? AND team_id = ?
                    """,
                    (game["map_id"], game["team_a_id"]),
                ).fetchone()[0],
            )
            self.assertEqual(
                5,
                connection.execute(
                    """
                    SELECT COUNT(*) FROM lineup_member
                    WHERE map_id = ? AND team_id = ?
                    """,
                    (game["map_id"], game["team_b_id"]),
                ).fetchone()[0],
            )

            # `shared` appears once on each opponent.  Its low-confidence IDs
            # are scoped by team and must remain distinct.
            shared = connection.execute(
                "SELECT player_id FROM player WHERE canonical_nickname = 'shared'"
            ).fetchall()
            self.assertEqual(2, len(shared))
            self.assertNotEqual(shared[0]["player_id"], shared[1]["player_id"])
            self.assertEqual(
                2,
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT team_id) FROM player_map_stats
                    WHERE map_id = ? AND player_id IN (?, ?)
                    """,
                    (game["map_id"], shared[0]["player_id"], shared[1]["player_id"]),
                ).fetchone()[0],
            )

            stat = connection.execute(
                """
                SELECT pms.*, p.canonical_nickname, tc.canonical_name
                FROM player_map_stats AS pms
                JOIN player AS p ON p.player_id = pms.player_id
                JOIN team_core AS tc ON tc.team_id = pms.team_id
                WHERE pms.map_id = ? AND p.canonical_nickname = 'alpha-two'
                """,
                (game["map_id"],),
            ).fetchone()
            self.assertEqual("Alpha", stat["canonical_name"])
            self.assertEqual(22, stat["kills"])
            self.assertEqual(12, stat["headshots"])
            self.assertEqual(5, stat["assists"])
            self.assertEqual(2, stat["flash_assists"])
            self.assertEqual(12, stat["deaths"])
            self.assertEqual(80.0, stat["kast"])
            self.assertEqual(83.4, stat["adr"])
            self.assertEqual(1.05, stat["rating"])
            self.assertIsNone(stat["known_at"])
            metrics = json.loads(stat["metrics_json"])
            self.assertEqual("10", metrics["player_raw_cells"]["kddiff"])
            self.assertEqual("+1", metrics["player_raw_cells"]["fkdiff"])
            self.assertEqual(2, metrics["stats_team_block"])
            self.assertEqual(16, metrics["stats_team_block_score"])
            self.assertNotIn("historic_raw_cells", metrics)
            self.assertNotIn("team_raw_cells", metrics)

            historic_only_map = connection.execute(
                "SELECT * FROM map_game WHERE map_id = 'tedtay:map:1003'"
            ).fetchone()
            self.assertEqual(16, historic_only_map["score_a"])
            self.assertEqual(8, historic_only_map["score_b"])
            self.assertEqual(
                0,
                connection.execute(
                    "SELECT COUNT(*) FROM lineup_member WHERE map_id = ?",
                    (historic_only_map["map_id"],),
                ).fetchone()[0],
            )
            self.assertEqual(
                0,
                connection.execute(
                    "SELECT COUNT(*) FROM player_map_stats WHERE map_id = ?",
                    (historic_only_map["map_id"],),
                ).fetchone()[0],
            )
            missing_raw = connection.execute(
                """
                SELECT * FROM raw_ingest_record
                WHERE source = ? AND record_kind = ? AND source_record_id = '1003'
                """,
                (SOURCE, MISSING_GAME_DATA_KIND),
            ).fetchone()
            self.assertIsNotNone(missing_raw)
            missing_payload = json.loads(missing_raw["payload_json"])
            self.assertEqual(MISSING_GAME_DATA_KIND, missing_payload["record_kind"])
            self.assertEqual(
                "(16)",
                missing_payload["historic_games_list_raw_cells"]["team1_rounds"],
            )
            self.assertNotIn("game_data_rh_raw_cells", missing_payload)
            self.assertTrue(audit(connection)["ok"])

    def test_replay_is_idempotent_and_from_date_filters_the_join(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)

            first = import_tedtay_dataset(
                connection, historic, stats, from_date=date(2021, 1, 1)
            )
            replay = import_tedtay_dataset(
                connection, historic, stats, from_date=date(2021, 1, 1)
            )

            self.assertEqual(2, first["joined_rows"])
            self.assertEqual(1, first["eligible_rows"])
            self.assertEqual(1, first["eligible_joined_rows"])
            self.assertEqual(0, first["eligible_unmatched_historic_rows"])
            self.assertEqual(2, first["filtered_by_from_date"])
            self.assertEqual(1, first["inserted"]["maps"])
            self.assertEqual(
                {
                    "teams": 0,
                    "players": 0,
                    "series": 0,
                    "series_participants": 0,
                    "maps": 0,
                    "lineup_members": 0,
                    "player_map_stats": 0,
                    "quarantine_raw_records": 0,
                    "ambiguous_team_binding_raw_records": 0,
                    "missing_game_data_rh_raw_records": 0,
                },
                replay["inserted"],
            )
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM source_snapshot").fetchone()[0])

    def test_invalid_lineup_fails_before_a_map_batch_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            rows = self._read_csv(stats)
            rows[0]["team1_p5_name"] = ""
            self._write_csv(stats, GAME_DATA_FIELDS, rows)
            connection = self._database(root)

            with self.assertRaisesRegex(TedTayImportError, "team1_p5_name"):
                import_tedtay_dataset(connection, historic, stats, batch_size=10)

            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM lineup_member").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0])

    def test_same_mapstatsid_cannot_be_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)
            import_tedtay_dataset(connection, historic, stats)

            changed_historic = root / "changed_historic_games_list.csv"
            changed_stats = root / "changed_game_data_rh.csv"
            historic_rows = self._read_csv(historic)
            historic_rows = [historic_rows[0]]
            historic_rows[0]["team1_rounds"] = "(15)"
            stats_rows = self._read_csv(stats)
            stats_rows = [stats_rows[0]]
            # Map 1001's stats are reverse-ordered, so make the second block
            # total 15 while retaining a coherent 10:15 source result.
            stats_rows[0]["team2_half1_ct"] = "7"
            stats_rows[0]["team1_half2_t"] = "8"
            self._write_csv(changed_historic, HISTORIC_FIELDS, historic_rows)
            self._write_csv(changed_stats, GAME_DATA_FIELDS, stats_rows)

            with self.assertRaisesRegex(TedTayImportError, "stable mapstatsid 1001"):
                import_tedtay_dataset(connection, changed_historic, changed_stats)

            game = connection.execute(
                "SELECT score_a, score_b FROM map_game WHERE map_id = 'tedtay:map:1001'"
            ).fetchone()
            self.assertEqual((16, 10), (game["score_a"], game["score_b"]))

    def test_schema_and_unresolvable_score_binding_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)
            bad_schema = root / "bad_historic.csv"
            self._write_csv(bad_schema, HISTORIC_FIELDS[:-1], [])
            with self.assertRaisesRegex(TedTayImportError, "header differs"):
                import_tedtay_dataset(connection, bad_schema, stats)

            historic_rows = self._read_csv(historic)
            historic_rows = [historic_rows[0]]
            stats_rows = self._read_csv(stats)
            stats_rows = [stats_rows[0]]
            # Changing one final score makes neither orientation valid.  It is
            # retained as auditable raw data rather than assigning player
            # tables to a team by a guess.
            historic_rows[0]["team1_rounds"] = "(14)"
            bad_historic = root / "bad_score_historic_games_list.csv"
            bad_stats = root / "bad_score_game_data_rh.csv"
            self._write_csv(bad_historic, HISTORIC_FIELDS, historic_rows)
            self._write_csv(bad_stats, GAME_DATA_FIELDS, stats_rows)
            result = import_tedtay_dataset(connection, bad_historic, bad_stats)
            self.assertEqual(1, result["ambiguous_team_binding_count"])
            self.assertEqual(
                {"four_half_totals_do_not_resolve_team_binding": 1},
                result["ambiguous_team_binding_by_reason"],
            )
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM lineup_member").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0])

    def test_optional_player_metrics_preserve_unknown_values_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            rows = self._read_csv(stats)
            rows[0]["team1_p2_assists"] = "5"
            rows[0]["team1_p2_kast"] = "-"
            rows[0]["team1_p2_adr"] = "-"
            rows[0]["team1_p2_game_rating"] = "-"
            self._write_csv(stats, GAME_DATA_FIELDS, rows)
            connection = self._database(root)

            import_tedtay_dataset(connection, historic, stats)
            stat = connection.execute(
                """
                SELECT pms.*
                FROM player_map_stats AS pms
                JOIN player AS p ON p.player_id = pms.player_id
                WHERE p.canonical_nickname = 'beta-two'
                """
            ).fetchone()
            self.assertEqual(5, stat["assists"])
            self.assertIsNone(stat["flash_assists"])
            self.assertIsNone(stat["kast"])
            self.assertIsNone(stat["adr"])
            self.assertIsNone(stat["rating"])
            raw = json.loads(stat["metrics_json"])["player_raw_cells"]
            self.assertEqual("5", raw["assists"])
            self.assertEqual("-", raw["kast"])
            self.assertEqual("-", raw["adr"])
            self.assertEqual("-", raw["game_rating"])

    def test_prefiltered_legacy_unfinished_score_does_not_abort_2018_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            historic_rows = self._read_csv(historic)
            stats_rows = self._read_csv(stats)
            pre_2018 = _historic_row(
                1004,
                when=datetime(2017, 12, 30, 12, 0, tzinfo=timezone.utc),
                team_a="Old Alpha",
                team_b="Old Beta",
                score_a=7,
                score_b=8,
            )
            pre_2018["team2_rounds"] = "(-)"
            historic_rows.append(pre_2018)
            stats_rows.append(
                _stats_row(
                    1004,
                    first_score=7,
                    second_score=8,
                    first_names=["o1", "o2", "o3", "o4", "o5"],
                    second_names=["p1", "p2", "p3", "p4", "p5"],
                    index=4,
                )
            )
            self._write_csv(historic, HISTORIC_FIELDS, historic_rows)
            self._write_csv(stats, GAME_DATA_FIELDS, stats_rows)
            connection = self._database(root)

            result = import_tedtay_dataset(
                connection, historic, stats, from_date=date(2018, 1, 1)
            )
            self.assertEqual(3, result["joined_rows"])
            self.assertEqual(3, result["eligible_rows"])
            self.assertEqual(2, result["eligible_joined_rows"])
            self.assertEqual(1, result["eligible_unmatched_historic_rows"])
            self.assertEqual(1, result["filtered_by_from_date"])
            self.assertEqual(0, result["ambiguous_team_binding_count"])
            self.assertEqual(1, result["missing_game_data_rh_count"])
            self.assertEqual(3, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])

    def test_ties_and_overtime_are_map_complete_but_player_binding_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic = root / "historic_games_list.csv"
            stats = root / "game_data_rh.csv"
            overtime = _historic_row(
                3001,
                when=datetime(2022, 6, 1, 12, 0, tzinfo=timezone.utc),
                team_a="Overtime A",
                team_b="Overtime B",
                score_a=25,
                score_b=23,
            )
            tied = _historic_row(
                3002,
                when=datetime(2022, 6, 2, 12, 0, tzinfo=timezone.utc),
                team_a="Tied A",
                team_b="Tied B",
                score_a=15,
                score_b=15,
            )
            overtime_stats = _stats_row(
                3001,
                first_score=15,
                second_score=15,
                first_names=["a1", "a2", "a3", "a4", "a5"],
                second_names=["b1", "b2", "b3", "b4", "b5"],
                index=1,
            )
            tied_stats = _stats_row(
                3002,
                first_score=15,
                second_score=15,
                first_names=["c1", "c2", "c3", "c4", "c5"],
                second_names=["d1", "d2", "d3", "d4", "d5"],
                index=2,
            )
            self._write_csv(historic, HISTORIC_FIELDS, [overtime, tied])
            self._write_csv(stats, GAME_DATA_FIELDS, [overtime_stats, tied_stats])
            connection = self._database(root)

            result = import_tedtay_dataset(connection, historic, stats, batch_size=2)

            self.assertEqual(2, result["eligible_rows"])
            self.assertEqual(2, result["ambiguous_team_binding_count"])
            self.assertEqual(
                {
                    "four_half_totals_do_not_resolve_team_binding": 1,
                    "tied_final_score": 1,
                },
                result["ambiguous_team_binding_by_reason"],
            )
            self.assertEqual(2, len(result["ambiguous_team_binding_samples"]))
            self.assertEqual(2, result["inserted"]["maps"])
            self.assertEqual(0, result["inserted"]["players"])
            self.assertEqual(0, result["inserted"]["lineup_members"])
            self.assertEqual(0, result["inserted"]["player_map_stats"])
            self.assertEqual(2, result["inserted"]["ambiguous_team_binding_raw_records"])
            self.assertEqual(2, connection.execute("SELECT COUNT(*) FROM series").fetchone()[0])
            self.assertEqual(
                4,
                connection.execute("SELECT COUNT(*) FROM series_participant").fetchone()[0],
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM lineup_member").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0])

            overtime_map = connection.execute(
                "SELECT winner_team_id FROM map_game WHERE map_id = 'tedtay:map:3001'"
            ).fetchone()
            tied_map = connection.execute(
                "SELECT winner_team_id FROM map_game WHERE map_id = 'tedtay:map:3002'"
            ).fetchone()
            self.assertIsNotNone(overtime_map["winner_team_id"])
            self.assertIsNone(tied_map["winner_team_id"])
            tied_series = connection.execute(
                "SELECT winner_team_id, known_at FROM series WHERE series_id = 'tedtay:series:3002'"
            ).fetchone()
            self.assertIsNone(tied_series["winner_team_id"])
            self.assertIsNone(tied_series["known_at"])
            tied_audit = audit(connection)
            self.assertEqual(0, tied_audit["checks"]["negative_finished_scores"])
            self.assertEqual(0, tied_audit["checks"]["winner_score_mismatches"])
            self.assertTrue(tied_audit["ok"])

            connection.execute(
                "UPDATE map_game SET winner_team_id = NULL "
                "WHERE map_id = 'tedtay:map:3001'"
            )
            missing_winner_audit = audit(connection)
            self.assertEqual(
                1, missing_winner_audit["checks"]["winner_score_mismatches"]
            )
            self.assertFalse(missing_winner_audit["ok"])
            connection.execute(
                "UPDATE map_game SET winner_team_id = team_a_id "
                "WHERE map_id = 'tedtay:map:3001'"
            )

            raw_rows = connection.execute(
                """
                SELECT * FROM raw_ingest_record
                WHERE source = ? AND record_kind = ?
                ORDER BY source_record_id
                """,
                (SOURCE, "tedtay_ambiguous_team_binding"),
            ).fetchall()
            self.assertEqual(2, len(raw_rows))
            self.assertTrue(
                all(row["stream"] == result["ambiguous_team_binding_stream"] for row in raw_rows)
            )
            self.assertTrue(all(row["known_at"] is None for row in raw_rows))
            tie_payload = json.loads(raw_rows[1]["payload_json"])
            self.assertEqual("tied_final_score", tie_payload["reason"])
            self.assertEqual("(15)", tie_payload["historic_games_list_raw_cells"]["team1_rounds"])
            self.assertEqual("7", tie_payload["game_data_rh_raw_cells"]["team1_half1_t"])

            replay = import_tedtay_dataset(connection, historic, stats, batch_size=2)
            self.assertEqual(0, replay["inserted"]["maps"])
            self.assertEqual(0, replay["inserted"]["ambiguous_team_binding_raw_records"])

            connection.execute(
                "UPDATE raw_ingest_record SET payload_json = '{}' WHERE source_record_id = '3002'"
            )
            with self.assertRaisesRegex(TedTayImportError, "ambiguous-team-binding raw record"):
                import_tedtay_dataset(connection, historic, stats, batch_size=2)

    def test_replay_rejects_every_normalized_scalar_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)
            import_tedtay_dataset(
                connection, historic, stats, from_date=date(2021, 1, 1)
            )

            def rejects(
                change_sql: str,
                change_values: tuple[object, ...],
                restore_sql: str,
                restore_values: tuple[object, ...],
                message: str,
            ) -> None:
                connection.execute(change_sql, change_values)
                connection.commit()
                with self.assertRaisesRegex(TedTayImportError, message):
                    import_tedtay_dataset(
                        connection, historic, stats, from_date=date(2021, 1, 1)
                    )
                connection.execute(restore_sql, restore_values)
                connection.commit()

            map_observed_at = connection.execute(
                "SELECT observed_at FROM map_game WHERE map_id = 'tedtay:map:1001'"
            ).fetchone()[0]
            rejects(
                "UPDATE map_game SET observed_at = 'corrupt' WHERE map_id = 'tedtay:map:1001'",
                (),
                "UPDATE map_game SET observed_at = ? WHERE map_id = 'tedtay:map:1001'",
                (map_observed_at,),
                "different map relation",
            )

            series_observed_at = connection.execute(
                "SELECT observed_at FROM series WHERE series_id = 'tedtay:series:1001'"
            ).fetchone()[0]
            rejects(
                "UPDATE series SET observed_at = 'corrupt' WHERE series_id = 'tedtay:series:1001'",
                (),
                "UPDATE series SET observed_at = ? WHERE series_id = 'tedtay:series:1001'",
                (series_observed_at,),
                "incompatible source metadata",
            )

            lineup = connection.execute(
                """
                SELECT team_id, slot FROM lineup_member
                WHERE map_id = 'tedtay:map:1001' ORDER BY team_id, slot LIMIT 1
                """
            ).fetchone()
            rejects(
                "UPDATE lineup_member SET role = 'substitute' WHERE map_id = ? AND team_id = ? AND slot = ?",
                ("tedtay:map:1001", lineup["team_id"], lineup["slot"]),
                "UPDATE lineup_member SET role = NULL WHERE map_id = ? AND team_id = ? AND slot = ?",
                ("tedtay:map:1001", lineup["team_id"], lineup["slot"]),
                "lineup rows",
            )

            stat = connection.execute(
                """
                SELECT player_id, observed_at, kills FROM player_map_stats
                WHERE map_id = 'tedtay:map:1001' ORDER BY player_id LIMIT 1
                """
            ).fetchone()
            rejects(
                "UPDATE player_map_stats SET observed_at = 'corrupt' WHERE map_id = ? AND player_id = ? AND side = 'BOTH' AND metric_version = ?",
                ("tedtay:map:1001", stat["player_id"], tedtay.METRIC_VERSION),
                "UPDATE player_map_stats SET observed_at = ? WHERE map_id = ? AND player_id = ? AND side = 'BOTH' AND metric_version = ?",
                (
                    stat["observed_at"],
                    "tedtay:map:1001",
                    stat["player_id"],
                    tedtay.METRIC_VERSION,
                ),
                "player stats",
            )
            rejects(
                "UPDATE player_map_stats SET kills = 999 WHERE map_id = ? AND player_id = ? AND side = 'BOTH' AND metric_version = ?",
                ("tedtay:map:1001", stat["player_id"], tedtay.METRIC_VERSION),
                "UPDATE player_map_stats SET kills = ? WHERE map_id = ? AND player_id = ? AND side = 'BOTH' AND metric_version = ?",
                (
                    stat["kills"],
                    "tedtay:map:1001",
                    stat["player_id"],
                    tedtay.METRIC_VERSION,
                ),
                "player stats",
            )

            # A repaired database is an exact replay again.
            replay = import_tedtay_dataset(
                connection, historic, stats, from_date=date(2021, 1, 1)
            )
            self.assertEqual(0, replay["inserted"]["maps"])

    def test_staging_rehash_rejects_in_place_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            connection = self._database(root)
            original_stage_stats = tedtay._stage_stats

            def stage_then_mutate(
                staged_connection: object, staged_path: Path
            ) -> tuple[int, int]:
                result = original_stage_stats(staged_connection, staged_path)
                historic_rows = self._read_csv(historic)
                historic_rows[0]["event_name"] = "changed after staging"
                self._write_csv(historic, HISTORIC_FIELDS, historic_rows)
                stats_rows = self._read_csv(stats)
                stats_rows[0]["collected_timestamp"] = "changed after staging"
                self._write_csv(stats, GAME_DATA_FIELDS, stats_rows)
                return result

            with patch.object(tedtay, "_stage_stats", side_effect=stage_then_mutate):
                with self.assertRaisesRegex(TedTayImportError, "changed while staging"):
                    import_tedtay_dataset(connection, historic, stats)
            self.assertEqual(
                0, connection.execute("SELECT COUNT(*) FROM source_snapshot").fetchone()[0]
            )
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])

    def test_historic_only_quarantine_is_bounded_idempotent_and_cannot_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic = root / "historic_games_list.csv"
            stats = root / "game_data_rh.csv"
            historic_rows = [
                _historic_row(
                    5000 + offset,
                    when=datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc),
                    team_a=f"Historic A {offset}",
                    team_b=f"Historic B {offset}",
                    score_a=16,
                    score_b=8,
                )
                for offset in range(25)
            ]
            self._write_csv(historic, HISTORIC_FIELDS, historic_rows)
            self._write_csv(stats, GAME_DATA_FIELDS, [])
            connection = self._database(root)

            first = import_tedtay_dataset(connection, historic, stats, batch_size=7)
            self.assertEqual(25, first["eligible_rows"])
            self.assertEqual(0, first["eligible_joined_rows"])
            self.assertEqual(25, first["eligible_unmatched_historic_rows"])
            self.assertEqual(25, first["missing_game_data_rh_count"])
            self.assertEqual(25, first["inserted"]["maps"])
            self.assertEqual(25, first["inserted"]["missing_game_data_rh_raw_records"])
            self.assertLessEqual(len(first["quarantine_samples"]), 20)
            self.assertEqual(20, len(first["quarantine_samples"]))
            self.assertNotIn("ambiguous_team_bindings", first)
            self.assertEqual(
                25,
                connection.execute(
                    "SELECT COUNT(*) FROM raw_ingest_record WHERE record_kind = ?",
                    (MISSING_GAME_DATA_KIND,),
                ).fetchone()[0],
            )

            replay = import_tedtay_dataset(connection, historic, stats, batch_size=11)
            self.assertEqual(0, replay["inserted"]["maps"])
            self.assertEqual(0, replay["inserted"]["quarantine_raw_records"])

            raw = connection.execute(
                """
                SELECT raw_record_id, payload_json FROM raw_ingest_record
                WHERE record_kind = ? AND source_record_id = '5001'
                """,
                (MISSING_GAME_DATA_KIND,),
            ).fetchone()
            connection.execute(
                "UPDATE raw_ingest_record SET payload_json = '{}' WHERE raw_record_id = ?",
                (raw["raw_record_id"],),
            )
            connection.commit()
            with self.assertRaisesRegex(TedTayImportError, "missing-game-data-rh raw record"):
                import_tedtay_dataset(connection, historic, stats)
            connection.execute(
                "UPDATE raw_ingest_record SET payload_json = ? WHERE raw_record_id = ?",
                (raw["payload_json"], raw["raw_record_id"]),
            )
            connection.commit()

            self._write_csv(
                stats,
                GAME_DATA_FIELDS,
                [
                    _stats_row(
                        5000,
                        first_score=16,
                        second_score=8,
                        first_names=["a1", "a2", "a3", "a4", "a5"],
                        second_names=["b1", "b2", "b3", "b4", "b5"],
                        index=1,
                    )
                ],
            )
            with self.assertRaisesRegex(TedTayImportError, "stable mapstatsid 5000"):
                import_tedtay_dataset(connection, historic, stats)

    def test_malformed_score_cell_fails_closed_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historic, stats = self._dataset(root)
            rows = self._read_csv(stats)
            rows[0]["team1_half1_t"] = "not-an-integer"
            self._write_csv(stats, GAME_DATA_FIELDS, rows)
            connection = self._database(root)

            with self.assertRaisesRegex(TedTayImportError, "team1_half1_t"):
                import_tedtay_dataset(connection, historic, stats, batch_size=10)
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM map_game").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM raw_ingest_record").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
