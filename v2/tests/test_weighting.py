from __future__ import annotations

import unittest

from predic_v2.weighting import effective_tier_weight_mass, tier_weight


class TierWeightingTest(unittest.TestCase):
    def test_tier1_profile_keeps_top_tiers_and_downweights_lower_tiers(self) -> None:
        self.assertEqual(1.0, tier_weight("s", "tier1"))
        self.assertEqual(0.8, tier_weight("A", "tier1"))
        self.assertLess(tier_weight("b", "tier1"), tier_weight("b", "balanced"))
        self.assertLess(tier_weight("d", "tier1"), tier_weight("d", "balanced"))

    def test_effective_mass_is_reported_separately_from_row_fraction(self) -> None:
        report = effective_tier_weight_mass(
            ["s", "b", "b"], [1.0, 0.25, 0.25]
        )

        self.assertAlmostEqual(1 / 3, report["s"]["row_fraction"])
        self.assertAlmostEqual(2 / 3, report["s"]["weight_mass_fraction"])


if __name__ == "__main__":
    unittest.main()
