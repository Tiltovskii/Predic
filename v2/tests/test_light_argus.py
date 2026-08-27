from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
    raise unittest.SkipTest(
        "Light Argus optional dependency is not installed"
    ) from error

from predic_v2.light_argus import (
    LightArgusConfig,
    LightTargetAwareArgus,
    swap_batch_sides,
)
from predic_v2.light_argus_data import (
    EVENT_NUMERIC_FIELDS,
    SHARED_NUMERIC_FIELDS,
    SIDE_NUMERIC_FIELDS,
    _history_indices,
)


class LightArgusDataTest(unittest.TestCase):
    def test_history_is_right_aligned_and_cut_at_target_start(self) -> None:
        offsets = np.array([0, 0, 3, 5], dtype=np.int64)
        known = np.array([100, 200, 300, 150, 250], dtype=np.int64)
        matches = np.array([10, 11, 12, 20, 21], dtype=np.int64)

        history = _history_indices(
            offsets,
            known,
            matches,
            [1, 2],
            target_start_ts=250,
            target_match_id=99,
            max_history=3,
        )

        np.testing.assert_array_equal(history[0], [-1, 0, 1])
        np.testing.assert_array_equal(history[1], [-1, 3, 4])

    def test_history_rejects_current_match_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaked"):
            _history_indices(
                np.array([0, 0, 1], dtype=np.int64),
                np.array([100], dtype=np.int64),
                np.array([99], dtype=np.int64),
                [1],
                target_start_ts=100,
                target_match_id=99,
                max_history=2,
            )

    def test_candidate_numeric_schema_contains_no_target_outcome(self) -> None:
        fields = {
            *EVENT_NUMERIC_FIELDS,
            *SIDE_NUMERIC_FIELDS,
            *SHARED_NUMERIC_FIELDS,
        }
        forbidden = {
            "team1_win",
            "team1_map_score",
            "team2_map_score",
            "round_share_target",
        }
        self.assertTrue(fields.isdisjoint(forbidden))


class LightArgusModelTest(unittest.TestCase):
    def _batch(self) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(7)
        batch, players, history, event_numeric = 2, 10, 3, 8
        side_numeric, shared_numeric = 6, 4
        return {
            "history_mask": torch.tensor(
                [[[False, True, True]] * players] * batch, dtype=torch.bool
            ),
            "event_numeric": torch.randn(
                batch, players, history, event_numeric, generator=generator
            ),
            "event_team": torch.randint(
                1, 11, (batch, players, history), generator=generator
            ),
            "event_opponent": torch.randint(
                1, 11, (batch, players, history), generator=generator
            ),
            "event_map": torch.randint(
                1, 7, (batch, players, history), generator=generator
            ),
            "event_tier": torch.randint(
                1, 4, (batch, players, history), generator=generator
            ),
            "event_event_type": torch.randint(
                1, 3, (batch, players, history), generator=generator
            ),
            "event_version": torch.randint(
                1, 3, (batch, players, history), generator=generator
            ),
            "target_players": torch.randint(
                1, 20, (batch, players), generator=generator
            ),
            "target_teams": torch.randint(1, 11, (batch, 2), generator=generator),
            "target_side_numeric": torch.randn(
                batch, 2, side_numeric, generator=generator
            ),
            "target_shared_numeric": torch.randn(
                batch, shared_numeric, generator=generator
            ),
            "target_map": torch.randint(1, 7, (batch,), generator=generator),
            "target_tier": torch.randint(1, 4, (batch,), generator=generator),
            "target_event_type": torch.randint(1, 3, (batch,), generator=generator),
            "target_version": torch.randint(1, 3, (batch,), generator=generator),
            "target_bo_type": torch.randint(1, 3, (batch,), generator=generator),
            "target_role": torch.randint(1, 3, (batch,), generator=generator),
            "target_slot": torch.randint(1, 4, (batch,), generator=generator),
            "target_label": torch.tensor([0.0, 1.0]),
            "target_weight": torch.ones(batch),
        }

    def _model(self) -> LightTargetAwareArgus:
        return LightTargetAwareArgus(
            LightArgusConfig(
                player_vocab_size=20,
                team_vocab_size=11,
                map_vocab_size=7,
                tier_vocab_size=4,
                event_type_vocab_size=3,
                version_vocab_size=3,
                bo_vocab_size=3,
                role_vocab_size=3,
                event_numeric_size=8,
                side_numeric_size=6,
                shared_numeric_size=4,
                max_history=3,
                d_model=16,
                layers=2,
                heads=4,
                dropout=0.0,
            )
        ).eval()

    def test_side_swap_negates_logit_exactly(self) -> None:
        model = self._model()
        batch = self._batch()

        with torch.inference_mode():
            direct = model(batch)
            mirrored = model(swap_batch_sides(batch))

        torch.testing.assert_close(direct, -mirrored, atol=1e-6, rtol=1e-6)

    def test_candidate_map_is_early_bound_into_encoder(self) -> None:
        model = self._model()
        batch = self._batch()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["target_map"] = (changed["target_map"] % 6) + 1

        with torch.inference_mode():
            direct = model(batch)
            other_map = model(changed)

        self.assertFalse(torch.allclose(direct, other_map))


if __name__ == "__main__":
    unittest.main()
