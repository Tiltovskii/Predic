from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta

from predic_v2.counters import EnrichedCounterStore

_ROSTER1 = ("11:a", "12:b", "13:c", "14:d", "15:e")
_ROSTER2 = ("21:f", "22:g", "23:h", "24:i", "25:j")


def _match(
    *,
    team1_id: int = 1,
    team2_id: int = 2,
    team1_win: bool = True,
    map_names: tuple[str, ...] = (),
) -> dict[str, str]:
    winner = team1_id if team1_win else team2_id
    loser = team2_id if team1_win else team1_id
    map_results = [
        {
            "map_name": map_name,
            "winner_team_id": winner,
            "loser_team_id": loser,
            "winner_score": 13,
            "loser_score": 8,
        }
        for map_name in map_names
    ]
    maps_played = len(map_names) or 1
    return {
        "team1_id": str(team1_id),
        "team2_id": str(team2_id),
        "team1_win": str(int(team1_win)),
        "team1_map_wins": str(maps_played if team1_win else 0),
        "team2_map_wins": str(0 if team1_win else maps_played),
        "maps_played": str(maps_played),
        "team1_rounds": str(13 * maps_played if team1_win else 8 * maps_played),
        "team2_rounds": str(8 * maps_played if team1_win else 13 * maps_played),
        "rounds_known": "1",
        "bo_type": "3",
        "tournament_tier": "a",
        "event_type": "lan",
        "game_version": "2",
        "tournament_id": "10",
        "bracket_type": "upper",
        "map_results": json.dumps(map_results),
    }


class EnrichedCounterStoreTest(unittest.TestCase):
    def _update(
        self,
        store: EnrichedCounterStore,
        match: dict[str, str],
        at: datetime,
        roster1: tuple[str, ...] = _ROSTER1,
        roster2: tuple[str, ...] = _ROSTER2,
    ) -> None:
        store.update(
            match,
            at,
            roster1,
            roster2,
            team1_elo=1500.0,
            team2_elo=1500.0,
            team1_elo_delta=16.0 if int(match["team1_win"]) else -16.0,
        )

    def test_unknown_and_empty_map_names_do_not_enter_map_pool(self) -> None:
        store = EnrichedCounterStore()
        self._update(
            store,
            _match(map_names=("", " UNKNOWN ")),
            datetime(2025, 1, 1, tzinfo=UTC),
        )

        self.assertFalse(store.maps.get(1))
        self.assertFalse(store.maps.get(2))

    def test_cache_and_de_cache_share_one_canonical_history(self) -> None:
        store = EnrichedCounterStore()
        at = datetime(2025, 1, 1, tzinfo=UTC)
        self._update(store, _match(map_names=("Cache",)), at)
        self._update(store, _match(map_names=(" de_cache ",)), at + timedelta(days=1))

        self.assertEqual({"cache"}, set(store.maps[1]))
        self.assertEqual({"cache"}, set(store.maps[2]))
        self.assertEqual(2, store.maps[1]["cache"].matches)
        self.assertEqual(2, store.maps[2]["cache"].matches)

    def test_map_features_are_symmetric_when_sides_are_swapped(self) -> None:
        store = EnrichedCounterStore()
        at = datetime(2025, 1, 1, tzinfo=UTC)
        self._update(
            store,
            _match(team1_id=1, team2_id=3, map_names=("cache", "inferno")),
            at,
        )
        self._update(
            store,
            _match(
                team1_id=2,
                team2_id=4,
                team1_win=False,
                map_names=("de_cache", "mirage"),
            ),
            at + timedelta(days=1),
        )
        feature_at = at + timedelta(days=2)
        forward = store.features(_match(), feature_at, _ROSTER1, _ROSTER2)
        reverse = store.features(
            _match(team1_id=2, team2_id=1), feature_at, _ROSTER2, _ROSTER1
        )

        for key, value in forward.items():
            if key.startswith("team1_counter_map_pool_"):
                counterpart = key.replace("team1_", "team2_", 1)
                self.assertAlmostEqual(value, reverse[counterpart], msg=key)
            elif key.startswith("team2_counter_map_pool_"):
                counterpart = key.replace("team2_", "team1_", 1)
                self.assertAlmostEqual(value, reverse[counterpart], msg=key)
            elif key.startswith("diff_counter_map_pool_"):
                self.assertAlmostEqual(value, -reverse[key], msg=key)
            elif key.startswith("counter_map_pool_"):
                self.assertAlmostEqual(value, reverse[key], msg=key)

    def test_partial_roster_does_not_replace_last_full_five(self) -> None:
        store = EnrichedCounterStore()
        at = datetime(2025, 1, 1, tzinfo=UTC)
        self._update(store, _match(map_names=("cache",)), at)
        self._update(
            store,
            _match(map_names=("inferno",)),
            at + timedelta(days=1),
            roster1=_ROSTER1[:4],
        )

        expected = frozenset(player.split(":", 1)[0] for player in _ROSTER1)
        self.assertEqual(expected, store.teams[1].last_roster)
        features = store.features(_match(), at + timedelta(days=2), _ROSTER1, _ROSTER2)
        self.assertEqual(5.0, features["team1_counter_roster_overlap_previous"])
        self.assertEqual(0.0, features["team1_counter_roster_changed"])


if __name__ == "__main__":
    unittest.main()
