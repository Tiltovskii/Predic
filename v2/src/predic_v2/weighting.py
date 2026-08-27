from __future__ import annotations

TIER_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    # Historical baseline. Keep it stable so old artifacts remain reproducible.
    "balanced": {
        "s": 1.0,
        "a": 0.80,
        "b": 0.55,
        "c": 0.35,
        "d": 0.20,
        "unknown": 0.30,
    },
    # Optimise the final map head for the matches we are most likely to bet on,
    # while retaining lower-tier data as regularising support.
    "tier1": {
        "s": 1.0,
        "a": 0.80,
        "b": 0.25,
        "c": 0.12,
        "d": 0.05,
        "unknown": 0.10,
    },
    "uniform": {
        "s": 1.0,
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "d": 1.0,
        "unknown": 1.0,
    },
}


def tier_weight(tier: object, profile: str = "balanced") -> float:
    try:
        weights = TIER_WEIGHT_PROFILES[profile]
    except KeyError as error:
        choices = ", ".join(sorted(TIER_WEIGHT_PROFILES))
        raise ValueError(f"unknown tier weight profile {profile!r}; choose {choices}") from error
    key = str(tier or "unknown").strip().casefold()
    return weights.get(key, weights["unknown"])


def effective_tier_weight_mass(
    tiers: list[object], weights: list[float]
) -> dict[str, dict[str, float]]:
    """Summarise row and effective training mass without pandas dependency."""
    if len(tiers) != len(weights):
        raise ValueError("tiers and weights must have the same length")
    rows: dict[str, int] = {}
    mass: dict[str, float] = {}
    for raw_tier, raw_weight in zip(tiers, weights):
        tier = str(raw_tier or "unknown").strip().casefold()
        rows[tier] = rows.get(tier, 0) + 1
        mass[tier] = mass.get(tier, 0.0) + float(raw_weight)
    total_rows = sum(rows.values())
    total_mass = sum(mass.values())
    return {
        tier: {
            "rows": float(rows[tier]),
            "row_fraction": rows[tier] / total_rows if total_rows else 0.0,
            "weight_mass": mass[tier],
            "weight_mass_fraction": mass[tier] / total_mass if total_mass else 0.0,
        }
        for tier in sorted(rows)
    }
