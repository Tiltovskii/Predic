from __future__ import annotations

import unittest

from predic_v2.valve_rankings import parse_standings_markdown


class ValveRankingsTest(unittest.TestCase):
    def test_parses_old_regional_format(self) -> None:
        rows = parse_standings_markdown(
            """
### Regional Standings for Europe as of 2022-12-21
|Standing|Points|Team Name|Roster|
|-|-|-|-|
|1|1949|Heroic| cadiaN, k0nfig, TeSeS, sjuush, jabbi|
""",
            source_commit="abc",
            source_path="standings_europe.md",
        )
        self.assertEqual(1, len(rows))
        self.assertEqual("valve_regional", rows[0].ranking_system)
        self.assertEqual("2022-12-21", rows[0].published_at)
        self.assertEqual(
            ("cadiaN", "k0nfig", "TeSeS", "sjuush", "jabbi"), rows[0].roster
        )

    def test_parses_new_global_format(self) -> None:
        rows = parse_standings_markdown(
            """
### Standings as of 2026_08_03<br />
| Standing | Points | Team Name | Roster | |
| :- | -: | :- | :- | :- |
| 1 | 2011 | Spirit | donk, magixx, sh1ro, tN1R, zont1x | [details](x) |
""",
            source_commit="def",
            source_path="live/2026/standings_global_2026_08_03.md",
        )
        self.assertEqual("valve_global", rows[0].ranking_system)
        self.assertEqual(2011.0, rows[0].points)
        self.assertEqual("donk,magixx,sh1ro,tn1r,zont1x", rows[0].roster_signature)


if __name__ == "__main__":
    unittest.main()
