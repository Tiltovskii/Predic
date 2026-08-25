from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from predic_v2.db import connect, initialize
from predic_v2.hltv_offline import (
    HltvParseError,
    parse_file,
    parse_html,
    records_to_jsonl,
)
from predic_v2.raw_jsonl import import_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


class HltvOfflineParserTest(unittest.TestCase):
    def test_source_url_and_canonical_numeric_id_must_agree(self) -> None:
        html = (FIXTURES / "hltv_match.html").read_text(encoding="utf-8")

        with self.assertRaises(HltvParseError):
            parse_html(
                html,
                page_type="match",
                source_url="https://www.hltv.org/matches/9999/wrong-entity",
            )

    def test_match_page_emits_typed_deterministic_records(self) -> None:
        path = FIXTURES / "hltv_match.html"
        first = parse_file(path)
        second = parse_file(path)

        self.assertEqual(records_to_jsonl(first), records_to_jsonl(second))
        self.assertEqual(6, len(first))
        by_kind: dict[str, list[dict[str, object]]] = {}
        for record in first:
            by_kind.setdefault(str(record["kind"]), []).append(record)
            self.assertIsNone(record["known_at"])

        series = by_kind["series"][0]
        payload = series["payload"]
        self.assertEqual("9001", payload["match_id"])
        self.assertEqual("finished", payload["status"])
        self.assertEqual(3, payload["best_of"])
        self.assertEqual("online", payload["lan_online"])
        self.assertEqual(2, payload["map_count"])
        self.assertEqual(3, len(payload["veto"]["actions"]))
        self.assertEqual("decider", payload["veto"]["actions"][-1]["action"])
        self.assertEqual("MR12", payload["ruleset"])
        self.assertEqual(1, payload["series_score_a"])
        self.assertEqual(0, payload["series_score_b"])

        self.assertEqual(1, len(by_kind["ranking"]))
        self.assertEqual(5, by_kind["ranking"][0]["payload"]["rank"])
        self.assertEqual(2, len(by_kind["lineup"]))
        alpha_players = by_kind["lineup"][0]["payload"]["players"]
        self.assertEqual(5, len(alpha_players))
        self.assertEqual("standin", alpha_players[-1]["member_type"])

        maps = by_kind["map"]
        self.assertEqual("played", maps[0]["payload"]["status"])
        self.assertEqual("10", maps[0]["payload"]["winner_team_id"])
        self.assertEqual("10", maps[0]["payload"]["picked_by_team_id"])
        self.assertEqual(13, maps[0]["payload"]["score_a"])
        self.assertEqual(8, maps[0]["payload"]["score_b"])
        self.assertEqual(21, maps[0]["payload"]["completed_rounds"])
        self.assertEqual(21, maps[0]["payload"]["regulation_rounds"])
        self.assertEqual(0, maps[0]["payload"]["overtime_rounds"])
        self.assertEqual([[7, 5], [6, 3]], maps[0]["payload"]["half_scores"])
        self.assertEqual("unplayed", maps[1]["payload"]["status"])
        self.assertIn("unplayed_map_not_training_eligible", maps[1]["warnings"])

    def test_map_stats_preserve_rating_version_and_missing_values(self) -> None:
        records = parse_file(FIXTURES / "hltv_map_stats.html")

        self.assertEqual(2, len(records))
        alpha = records[0]["payload"]
        beta = records[1]["payload"]
        self.assertEqual("player_map_stats", records[0]["kind"])
        self.assertEqual("501", alpha["map_stats_id"])
        self.assertEqual("CS2", alpha["game_version"])
        self.assertEqual("hltv-rating-3.0", alpha["metric_version"])
        self.assertEqual(13, alpha["score_a"])
        self.assertEqual(8, alpha["score_b"])
        self.assertEqual("MR12", alpha["ruleset"])
        self.assertEqual(21, alpha["completed_rounds"])
        self.assertEqual(20, alpha["kills"])
        self.assertEqual(3, alpha["flash_assists"])
        self.assertEqual(4, alpha["opening_kills"])
        self.assertEqual(2, alpha["opening_deaths"])
        self.assertEqual(2.35, alpha["swing"])
        self.assertEqual(-1.8, beta["swing"])
        self.assertIsNone(alpha["headshots"])

    def test_old_rating_page_keeps_swing_null(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/stats/matches/mapstatsid/99/a-vs-b">
        <div class="cs-version">Counter-Strike: Global Offensive</div>
        <a href="/stats/teams/1/a">A</a><a href="/stats/teams/2/b">B</a>
        <table class="table totalstats">
          <tr><th>Player</th><th>ADR</th><th>KAST</th><th>Rating 2.0</th></tr>
          <tr>
            <td><a href="/stats/players/3/p">p</a></td>
            <td class="st-adr">75.0</td><td class="st-kast">70%</td>
            <td class="st-rating">1.02</td>
          </tr>
        </table>
        """
        records = parse_html(html)

        self.assertEqual(1, len(records))
        payload = records[0]["payload"]
        self.assertEqual("CSGO", payload["game_version"])
        self.assertEqual("hltv-rating-2.0", payload["metric_version"])
        self.assertIsNone(payload["swing"])
        self.assertNotIn("rating_3_without_swing_value", records[0]["warnings"])

    def test_header_driven_stats_work_without_metric_classes(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/stats/matches/mapstatsid/100/a-vs-b">
        <a href="/stats/teams/1/a">A</a><a href="/stats/teams/2/b">B</a>
        <table class="totalstats">
          <tr>
            <th>Player</th><th>K (hs)</th><th>A (f)</th><th>D (t)</th>
            <th>ADR</th><th>KAST</th><th>Rating 2.0</th>
          </tr>
          <tr>
            <td><a href="/stats/players/3/p">p</a></td>
            <td>18 (9)</td><td>7 (4)</td><td>13 (2)</td>
            <td>88.1</td><td>77.7%</td><td>1.20</td>
          </tr>
        </table>
        """
        record = parse_html(html)[0]
        payload = record["payload"]

        self.assertEqual(18, payload["kills"])
        self.assertEqual(9, payload["headshots"])
        self.assertEqual(7, payload["assists"])
        self.assertEqual(4, payload["flash_assists"])
        self.assertEqual(13, payload["deaths"])
        self.assertEqual(2, payload["traded_deaths"])
        self.assertEqual(88.1, payload["adr"])
        self.assertEqual(77.7, payload["kast"])

    def test_old_kd_column_preserves_deaths(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/stats/matches/mapstatsid/101/a-vs-b">
        <div class="match-info-box">
          <a href="/stats/teams/1/a">A</a><a href="/stats/teams/2/b">B</a>
        </div>
        <table class="totalstats">
          <tr><th>Player</th><th>K-D</th><th>Rating 2.0</th></tr>
          <tr>
            <td><a href="/stats/players/3/p">p</a></td>
            <td class="st-kills">20-15</td><td class="st-rating">1.20</td>
          </tr>
        </table>
        """
        payload = parse_html(html)[0]["payload"]

        self.assertEqual(20, payload["kills"])
        self.assertEqual(15, payload["deaths"])

    def test_walkover_does_not_invent_maps(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/matches/42/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <div class="match-info-note">Best of 3 — win by walkover</div>
        """
        records = parse_html(html)

        self.assertEqual(1, len(records))
        self.assertEqual("series", records[0]["kind"])
        self.assertEqual("walkover", records[0]["payload"]["status"])
        self.assertEqual(0, records[0]["payload"]["map_count"])

    def test_player_cannot_appear_for_both_teams(self) -> None:
        rows = "".join(
            f'<tr><td><a href="/player/{player}/p{player}">p{player}</a></td></tr>'
            for player in range(1, 6)
        )
        html = f"""
        <link rel="canonical" href="https://www.hltv.org/matches/43/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <table>{rows}</table><table>{rows}</table>
        """

        with self.assertRaises(HltvParseError):
            parse_html(html)

    def test_historical_single_table_can_hold_two_lineup_rows(self) -> None:
        first = "".join(
            f'<td><a href="/player/{player}/a{player}">a{player}</a></td>'
            for player in range(1, 6)
        )
        second = "".join(
            f'<td><a href="/player/{player}/b{player}">b{player}</a></td>'
            for player in range(6, 11)
        )
        html = f"""
        <link rel="canonical" href="https://www.hltv.org/matches/44/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <table><tr>{first}</tr><tr>{second}</tr></table>
        """
        lineups = [record for record in parse_html(html) if record["kind"] == "lineup"]

        self.assertEqual(2, len(lineups))
        self.assertEqual(5, len(lineups[0]["payload"]["players"]))
        self.assertEqual(5, len(lineups[1]["payload"]["players"]))

    def test_live_score_never_becomes_finished_label(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/matches/45/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <div class="match-status">LIVE</div>
        <div class="mapholder">
          <div class="mapname">Mirage</div>
          <div class="results-left">
            <div class="results-teamname">A</div>
            <div class="results-team-score">7</div>
          </div>
          <div class="results-right">
            <div class="results-teamname">B</div>
            <div class="results-team-score">5</div>
          </div>
        </div>
        """
        records = parse_html(html)
        series = next(record for record in records if record["kind"] == "series")
        map_record = next(record for record in records if record["kind"] == "map")

        self.assertEqual("live", series["payload"]["status"])
        self.assertEqual("live", map_record["payload"]["status"])
        self.assertIsNone(map_record["payload"]["winner_team_id"])
        self.assertIn("live_map_not_training_eligible", map_record["warnings"])

    def test_csgo_overtime_round_count_is_explicit(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/matches/47/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <div class="cs-version">Counter-Strike: Global Offensive</div>
        <div class="match-status">Finished</div>
        <div class="mapholder">
          <div class="mapname">Train</div>
          <div class="results-left won">
            <div class="results-teamname">A</div>
            <div class="results-team-score">19</div>
          </div>
          <div class="results-right lost">
            <div class="results-teamname">B</div>
            <div class="results-team-score">17</div>
          </div>
        </div>
        """
        map_payload = next(
            record["payload"]
            for record in parse_html(html)
            if record["kind"] == "map"
        )

        self.assertEqual("MR15", map_payload["ruleset"])
        self.assertEqual(36, map_payload["completed_rounds"])
        self.assertEqual(30, map_payload["regulation_rounds"])
        self.assertEqual(6, map_payload["overtime_rounds"])

    def test_sidebar_timestamp_is_not_used_as_match_time(self) -> None:
        html = """
        <link rel="canonical" href="https://www.hltv.org/matches/46/a-vs-b">
        <a href="/team/1/a">A</a><a href="/team/2/b">B</a>
        <div class="sidebar"><span data-unix="1704196800000">comment</span></div>
        """
        series = parse_html(html)[0]

        self.assertIsNone(series["event_at"])
        self.assertIn("event_at_missing", series["warnings"])

    def test_jsonl_is_valid_and_canonical(self) -> None:
        records = parse_file(FIXTURES / "hltv_map_stats.html")
        lines = records_to_jsonl(records).splitlines()

        self.assertEqual(len(records), len(lines))
        self.assertEqual(records, [json.loads(line) for line in lines])

    def test_capture_observed_at_is_propagated_to_every_record(self) -> None:
        captured_at = "2026-08-25T12:34:56+00:00"
        records = parse_file(FIXTURES / "hltv_match.html", observed_at=captured_at)

        self.assertTrue(records)
        self.assertTrue(all(record["observed_at"] == captured_at for record in records))

    def test_parser_output_resumes_through_raw_ingestion(self) -> None:
        records = parse_file(FIXTURES / "hltv_match.html")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "match.jsonl"
            jsonl_path.write_text(records_to_jsonl(records), encoding="utf-8")
            connection = connect(root / "test.sqlite3")
            self.addCleanup(connection.close)
            initialize(connection)

            first = import_jsonl(
                connection,
                jsonl_path,
                source="authorized-hltv-export",
                stream="fixture-match",
                max_records=2,
            )
            second = import_jsonl(
                connection,
                jsonl_path,
                source="authorized-hltv-export",
                stream="fixture-match",
            )
            third = import_jsonl(
                connection,
                jsonl_path,
                source="authorized-hltv-export",
                stream="fixture-match",
            )

            self.assertEqual(2, first["imported"])
            self.assertEqual(len(records) - 2, second["imported"])
            self.assertEqual(0, third["imported"])
            self.assertEqual(
                len(records),
                connection.execute(
                    "SELECT COUNT(*) FROM raw_ingest_record"
                ).fetchone()[0],
            )


if __name__ == "__main__":
    unittest.main()
