from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

_WINDOWS = (1, 3, 7, 14, 30, 90, 180)
_LAST_COUNTS = (3, 5, 10, 20)
_HALF_LIVES = (7, 30, 90)
_ROUND_STYLE_WINDOWS = (30, 180)
_ROUND_STYLE_RATES = {
    "t_win_rate": ("t_wins", "t_rounds"),
    "ct_win_rate": ("ct_wins", "ct_rounds"),
    "pistol_win_rate": ("pistol_wins", "pistol_rounds"),
    "eco_win_rate": ("eco_wins", "eco_rounds"),
    "force_win_rate": ("force_wins", "force_rounds"),
    "full_buy_win_rate": ("full_buy_wins", "full_buy_rounds"),
    "full_buy_duel_win_rate": (
        "full_buy_vs_full_buy_wins",
        "full_buy_vs_full_buy_rounds",
    ),
    "low_buy_upset_rate": (
        "low_buy_vs_full_buy_wins",
        "low_buy_vs_full_buy_rounds",
    ),
    "anti_low_buy_conversion": (
        "full_buy_vs_low_buy_wins",
        "full_buy_vs_low_buy_rounds",
    ),
    "opening_conversion": ("opening_wins", "opening_rounds"),
    "opening_recovery": (
        "opening_recovery_wins",
        "opening_conceded_rounds",
    ),
    "close_round_win_rate": ("close_wins", "close_rounds"),
    "trailing_3plus_recovery": (
        "trailing_3plus_wins",
        "trailing_3plus_rounds",
    ),
    "after_win_conversion": ("after_win_wins", "after_win_rounds"),
    "after_loss_recovery": ("after_loss_wins", "after_loss_rounds"),
    "clutch_conversion": ("clutches", "clutch_attempts"),
    "hit_rate": ("hits", "shots"),
}
_ROUND_STYLE_PER_ROUND = (
    "trade_kills",
    "trade_deaths",
    "flash_assists",
    "grenades_damage",
    "utility_value",
    "damage",
    "equipment_value",
    "money_spent",
)
_UNKNOWN_MAP_NAMES = frozenset(
    {"", "-", "?", "n/a", "na", "none", "null", "tbd", "unknown"}
)
_DE_MAP_ALIASES = {
    "de_ancient": "ancient",
    "de_anubis": "anubis",
    "de_cache": "cache",
    "de_cbble": "cobblestone",
    "de_cobblestone": "cobblestone",
    "de_dust2": "dust2",
    "de_inferno": "inferno",
    "de_mirage": "mirage",
    "de_nuke": "nuke",
    "de_overpass": "overpass",
    "de_season": "season",
    "de_train": "train",
    "de_tuscan": "tuscan",
    "de_vertigo": "vertigo",
}
_VETO_PATTERNS = {
    1: (2, 2, 2, 2, 2, 2, 3),
    3: (2, 2, 1, 1, 2, 2, 3),
    5: (2, 2, 1, 1, 1, 1, 3),
}
_VETO_ACTOR_COUNTS = {
    1: (0, 3),
    3: (1, 2),
    5: (2, 1),
}
_VETO_CATEGORY_KEYS = (
    *(f"team{side}_veto_pick_{slot}_map" for side in (1, 2) for slot in (1, 2)),
    *(f"team{side}_veto_ban_{slot}_map" for side in (1, 2) for slot in (1, 2, 3)),
    "veto_decider_map",
    *(f"veto_selected_map_{slot}" for slot in range(1, 6)),
)


def _canonical_map_name(value: object) -> str | None:
    """Return one stable map-pool key without guessing unfamiliar aliases."""
    name = str(value or "").strip().casefold()
    if name in _UNKNOWN_MAP_NAMES:
        return None
    return _DE_MAP_ALIASES.get(name, name)


def canonical_map_name(value: object) -> str | None:
    """Return the public canonical map key used by every modeling layer."""
    return _canonical_map_name(value)


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


@dataclass(frozen=True)
class _VetoAction:
    order: int
    choice_type: int
    team_id: int | None
    map_name: str


def _parse_veto_actions(match: dict[str, object]) -> tuple[_VetoAction, ...] | None:
    raw_actions = match.get("veto_actions") or "[]"
    if isinstance(raw_actions, str):
        try:
            raw_actions = json.loads(raw_actions)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw_actions, list):
        return None

    actions: list[_VetoAction] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            return None
        order = _strict_int(raw.get("order"))
        choice_type = _strict_int(raw.get("choice_type"))
        map_name = _canonical_map_name(raw.get("map_name"))
        if order is None or choice_type is None or map_name is None:
            return None
        team_id = _strict_int(raw.get("team_id"))
        actions.append(_VetoAction(order, choice_type, team_id, map_name))
    return tuple(sorted(actions, key=lambda action: action.order))


def _strict_veto_actions(
    match: dict[str, object],
) -> tuple[_VetoAction, ...] | None:
    actions = _parse_veto_actions(match)
    bo_type = _strict_int(match.get("bo_type"))
    team1_id = _strict_int(match.get("team1_id"))
    team2_id = _strict_int(match.get("team2_id"))
    if (
        actions is None
        or bo_type not in _VETO_PATTERNS
        or team1_id is None
        or team2_id is None
        or team1_id == team2_id
        or len(actions) != 7
    ):
        return None
    if tuple(action.order for action in actions) != tuple(range(1, 8)):
        return None
    if tuple(action.choice_type for action in actions) != _VETO_PATTERNS[bo_type]:
        return None
    if len({action.map_name for action in actions}) != len(actions):
        return None

    participants = (team1_id, team2_id)
    if any(
        action.team_id not in participants
        for action in actions
        if action.choice_type in {1, 2}
    ):
        return None
    if any(
        action.team_id not in {None, 0} for action in actions if action.choice_type == 3
    ):
        return None
    expected_picks, expected_bans = _VETO_ACTOR_COUNTS[bo_type]
    for team_id in participants:
        if (
            sum(
                action.choice_type == 1 and action.team_id == team_id
                for action in actions
            )
            != expected_picks
            or sum(
                action.choice_type == 2 and action.team_id == team_id
                for action in actions
            )
            != expected_bans
        ):
            return None
    return actions


def strict_veto_complete(match: dict[str, object]) -> bool:
    """Return whether a BO1/BO3/BO5 veto is complete and side-consistent."""
    return _strict_veto_actions(match) is not None


def _smoothed_rate(successes: float, total: float, prior: float = 0.5) -> float:
    return (successes + 4.0 * prior) / (total + 4.0)


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _age_days(at: datetime, previous: datetime | None, default: float = 999.0) -> float:
    if previous is None:
        return default
    return min(default, max(0.0, (at - previous).total_seconds() / 86_400))


@dataclass
class _Outcome:
    at: datetime
    win: float
    map_share: float | None
    round_share: float | None
    opponent_elo: float
    expected_win: float
    elo_delta: float
    maps_played: int
    round_margin_per_map: float | None
    bo_type: int


@dataclass
class _AggregateOutcome:
    at: datetime
    win: float
    map_share: float | None
    round_share: float | None


@dataclass
class _AggregateState:
    matches: int = 0
    wins: float = 0.0
    map_share_sum: float = 0.0
    map_known: int = 0
    round_share_sum: float = 0.0
    round_known: int = 0
    first_at: datetime | None = None
    last_at: datetime | None = None
    outcomes: deque[_AggregateOutcome] = field(default_factory=deque)

    def add(
        self,
        at: datetime,
        win: float,
        map_share: float | None,
        round_share: float | None,
    ) -> None:
        self.matches += 1
        self.wins += win
        if map_share is not None:
            self.map_share_sum += map_share
            self.map_known += 1
        if round_share is not None:
            self.round_share_sum += round_share
            self.round_known += 1
        self.first_at = self.first_at or at
        self.last_at = at
        self.outcomes.append(_AggregateOutcome(at, win, map_share, round_share))
        while self.outcomes and _age_days(at, self.outcomes[0].at) >= 365:
            self.outcomes.popleft()


@dataclass
class _TeamState:
    outcomes: deque[_Outcome] = field(default_factory=deque)
    last_roster: frozenset[str] = field(default_factory=frozenset)
    roster_changes: deque[datetime] = field(default_factory=deque)


@dataclass
class _PlayerState:
    matches: int = 0
    wins: float = 0.0
    first_at: datetime | None = None
    last_at: datetime | None = None
    teams: set[int] = field(default_factory=set)
    outcomes: deque[_AggregateOutcome] = field(default_factory=deque)


@dataclass(frozen=True)
class RoundStyleOutcome:
    at: datetime
    values: dict[str, float]


def round_stats_for_team(
    game: dict[str, object], team_id: int
) -> dict[str, float] | None:
    raw = game.get("round_stats") or {}
    if not isinstance(raw, dict):
        return None
    values = raw.get(str(team_id), raw.get(team_id))
    if not isinstance(values, dict):
        return None
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    if result.get("rounds", 0.0) <= 0:
        return None
    return result


def round_style_features(
    outcomes: deque[RoundStyleOutcome] | list[RoundStyleOutcome], at: datetime
) -> dict[str, float]:
    result: dict[str, float] = {}
    values = list(outcomes)
    for window in _ROUND_STYLE_WINDOWS:
        recent = [item.values for item in values if _age_days(at, item.at) < window]
        totals: defaultdict[str, float] = defaultdict(float)
        for item in recent:
            for key, value in item.items():
                totals[key] += value
        rounds = totals["rounds"]
        classified = (
            totals["eco_rounds"]
            + totals["force_rounds"]
            + totals["full_buy_rounds"]
        )
        result[f"round_style_maps_{window}d"] = float(len(recent))
        result[f"round_style_rounds_{window}d"] = rounds
        result[f"round_style_economy_coverage_{window}d"] = (
            classified / rounds if rounds else 0.0
        )
        for name, (numerator, denominator) in _ROUND_STYLE_RATES.items():
            result[f"round_style_{name}_{window}d"] = _smoothed_rate(
                totals[numerator], totals[denominator]
            )
            result[f"round_style_{name}_support_{window}d"] = totals[denominator]
        for metric in _ROUND_STYLE_PER_ROUND:
            result[f"round_style_{metric}_per_round_{window}d"] = (
                totals[metric] / rounds if rounds else 0.0
            )
    return result


def _aggregate_features(state: _AggregateState, at: datetime) -> dict[str, float]:
    recent = [item for item in state.outcomes if _age_days(at, item.at) < 90]
    return {
        "matches": float(state.matches),
        "log_matches": math.log1p(state.matches),
        "win_rate": _smoothed_rate(state.wins, state.matches),
        "map_share": _smoothed_rate(state.map_share_sum, state.map_known),
        "map_known": float(state.map_known),
        "round_share": _smoothed_rate(state.round_share_sum, state.round_known),
        "round_known": float(state.round_known),
        "matches_90d": float(len(recent)),
        "win_rate_90d": _smoothed_rate(sum(item.win for item in recent), len(recent)),
        "round_share_90d": _smoothed_rate(
            sum(item.round_share for item in recent if item.round_share is not None),
            sum(item.round_share is not None for item in recent),
        ),
        "round_known_90d": float(sum(item.round_share is not None for item in recent)),
        "days_since": _age_days(at, state.last_at),
        "tenure_days": 0.0
        if state.first_at is None
        else max(0.0, (at - state.first_at).total_seconds() / 86_400),
    }


class EnrichedCounterStore:
    """Additional causal counters built only from earlier series outcomes."""

    def __init__(self) -> None:
        self.teams: dict[int, _TeamState] = defaultdict(_TeamState)
        self.players: dict[str, _PlayerState] = defaultdict(_PlayerState)
        self.lineups: dict[tuple[int, tuple[str, ...]], _AggregateState] = defaultdict(
            _AggregateState
        )
        self.memberships: dict[tuple[int, str], _AggregateState] = defaultdict(
            _AggregateState
        )
        self.pairs: dict[tuple[int, str, str], _AggregateState] = defaultdict(
            _AggregateState
        )
        self.global_pairs: dict[tuple[str, str], _AggregateState] = defaultdict(
            _AggregateState
        )
        self.contexts: dict[tuple[int, str, str], _AggregateState] = defaultdict(
            _AggregateState
        )
        self.h2h: dict[tuple[int, int], _AggregateState] = defaultdict(_AggregateState)
        self.maps: dict[int, dict[str, _AggregateState]] = defaultdict(dict)
        self.round_styles: dict[int, deque[RoundStyleOutcome]] = defaultdict(deque)

    @staticmethod
    def _ids(roster: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(item.split(":", 1)[0] for item in roster))

    def _team_features(self, team_id: int, at: datetime) -> dict[str, float]:
        state = self.teams[team_id]
        outcomes = list(state.outcomes)
        result: dict[str, float] = {
            "rest_hours": _age_days(at, outcomes[-1].at) * 24 if outcomes else 999 * 24,
            "signed_streak": 0.0,
        }
        if outcomes:
            latest_win = outcomes[-1].win >= 0.5
            streak = 0
            for item in reversed(outcomes):
                if (item.win >= 0.5) != latest_win:
                    break
                streak += 1
            result["signed_streak"] = float(streak if latest_win else -streak)

        for window in _WINDOWS:
            recent = [item for item in outcomes if _age_days(at, item.at) < window]
            count = len(recent)
            wins = [item.win for item in recent]
            maps = [item.map_share for item in recent if item.map_share is not None]
            rounds = [
                item.round_share for item in recent if item.round_share is not None
            ]
            opponents = [item.opponent_elo for item in recent]
            residuals = [item.win - item.expected_win for item in recent]
            round_residuals = [
                item.round_share - item.expected_win
                for item in recent
                if item.round_share is not None
            ]
            map_margins = [
                2.0 * item.map_share - 1.0
                for item in recent
                if item.map_share is not None
            ]
            round_margins = [
                item.round_margin_per_map
                for item in recent
                if item.round_margin_per_map is not None
            ]
            result.update(
                {
                    f"matches_{window}d": float(count),
                    f"maps_played_{window}d": float(
                        sum(item.maps_played for item in recent)
                    ),
                    f"win_rate_{window}d": _smoothed_rate(sum(wins), count),
                    f"map_share_{window}d": _smoothed_rate(sum(maps), len(maps)),
                    f"map_known_{window}d": float(len(maps)),
                    f"round_share_{window}d": _smoothed_rate(sum(rounds), len(rounds)),
                    f"round_known_{window}d": float(len(rounds)),
                    f"round_share_std_{window}d": _std(rounds),
                    f"map_margin_per_map_{window}d": _mean(map_margins),
                    f"map_margin_std_{window}d": _std(map_margins),
                    f"round_margin_per_map_{window}d": _mean(round_margins),
                    f"round_margin_std_{window}d": _std(round_margins),
                    f"opponent_elo_{window}d": _mean(opponents, 1500.0),
                    f"opponent_elo_std_{window}d": _std(opponents),
                    f"elo_residual_{window}d": _mean(residuals),
                    f"round_elo_residual_proxy_{window}d": _mean(round_residuals),
                    f"elo_change_{window}d": sum(item.elo_delta for item in recent),
                }
            )

        for count in _LAST_COUNTS:
            recent = outcomes[-count:]
            size = len(recent)
            result.update(
                {
                    f"last_{count}_available": float(size),
                    f"win_rate_last_{count}": _smoothed_rate(
                        sum(item.win for item in recent), size
                    ),
                    f"map_share_last_{count}": _smoothed_rate(
                        sum(
                            item.map_share
                            for item in recent
                            if item.map_share is not None
                        ),
                        sum(item.map_share is not None for item in recent),
                    ),
                    f"map_known_last_{count}": float(
                        sum(item.map_share is not None for item in recent)
                    ),
                    f"round_share_last_{count}": _smoothed_rate(
                        sum(
                            item.round_share
                            for item in recent
                            if item.round_share is not None
                        ),
                        sum(item.round_share is not None for item in recent),
                    ),
                    f"round_known_last_{count}": float(
                        sum(item.round_share is not None for item in recent)
                    ),
                    f"elo_residual_last_{count}": _mean(
                        [item.win - item.expected_win for item in recent]
                    ),
                    f"round_share_std_last_{count}": _std(
                        [
                            item.round_share
                            for item in recent
                            if item.round_share is not None
                        ]
                    ),
                }
            )

        for half_life in _HALF_LIVES:
            weighted = [
                (0.5 ** (_age_days(at, item.at) / half_life), item)
                for item in outcomes
                if _age_days(at, item.at) < 6 * half_life
            ]
            total_weight = sum(weight for weight, _ in weighted)

            def weighted_mean(
                attribute: str,
                default: float,
                weighted_values: list[tuple[float, _Outcome]] = weighted,
            ) -> float:
                known_values = [
                    (weight, value)
                    for weight, item in weighted_values
                    if (value := getattr(item, attribute)) is not None
                ]
                known_weight = sum(weight for weight, _ in known_values)
                if not known_weight:
                    return default
                return (
                    sum(weight * float(value) for weight, value in known_values)
                    / known_weight
                )

            result.update(
                {
                    f"ewm_effective_matches_{half_life}d": total_weight,
                    f"ewm_effective_maps_{half_life}d": sum(
                        weight
                        for weight, item in weighted
                        if item.map_share is not None
                    ),
                    f"ewm_effective_rounds_{half_life}d": sum(
                        weight
                        for weight, item in weighted
                        if item.round_share is not None
                    ),
                    f"ewm_win_rate_{half_life}d": weighted_mean("win", 0.5),
                    f"ewm_map_share_{half_life}d": weighted_mean("map_share", 0.5),
                    f"ewm_round_share_{half_life}d": weighted_mean("round_share", 0.5),
                    f"ewm_opponent_elo_{half_life}d": weighted_mean(
                        "opponent_elo", 1500.0
                    ),
                    f"ewm_elo_residual_{half_life}d": (
                        sum(
                            weight * (item.win - item.expected_win)
                            for weight, item in weighted
                        )
                        / total_weight
                        if total_weight
                        else 0.0
                    ),
                }
            )

        result["win_trend_7d_vs_90d"] = result["win_rate_7d"] - result["win_rate_90d"]
        result["map_trend_14d_vs_180d"] = (
            result["map_share_14d"] - result["map_share_180d"]
        )
        result["round_trend_14d_vs_180d"] = (
            result["round_share_14d"] - result["round_share_180d"]
        )
        recent_180 = [item for item in outcomes if _age_days(at, item.at) < 180]
        favourites = [item for item in recent_180 if item.expected_win >= 0.60]
        underdogs = [item for item in recent_180 if item.expected_win <= 0.40]
        deciders = [
            item
            for item in recent_180
            if item.bo_type in {3, 5} and item.maps_played == item.bo_type
        ]
        sweeps = [
            item
            for item in recent_180
            if item.bo_type in {3, 5}
            and item.maps_played > 0
            and item.map_share in {0.0, 1.0}
        ]
        result.update(
            {
                "favourite_matches_180d": float(len(favourites)),
                "favourite_conversion_180d": _smoothed_rate(
                    sum(item.win for item in favourites), len(favourites)
                ),
                "underdog_matches_180d": float(len(underdogs)),
                "underdog_upset_rate_180d": _smoothed_rate(
                    sum(item.win for item in underdogs), len(underdogs)
                ),
                "decider_matches_180d": float(len(deciders)),
                "decider_win_rate_180d": _smoothed_rate(
                    sum(item.win for item in deciders), len(deciders)
                ),
                "sweep_matches_180d": float(len(sweeps)),
                "sweep_win_rate_180d": _smoothed_rate(
                    sum(item.win for item in sweeps), len(sweeps)
                ),
            }
        )
        return result

    def _player_features(
        self, team_id: int, roster: tuple[str, ...], at: datetime
    ) -> dict[str, float]:
        ids = self._ids(roster)
        states = [self.players[player_id] for player_id in ids]
        memberships = [self.memberships[(team_id, player_id)] for player_id in ids]
        if not states:
            return {
                "player_matches_min": 0.0,
                "player_matches_max": 0.0,
                "player_matches_std": 0.0,
                "player_win_rate_min": 0.5,
                "player_win_rate_max": 0.5,
                "player_win_rate_std": 0.0,
                "player_inactivity_min": 999.0,
                "player_inactivity_max": 999.0,
                "player_inactivity_std": 0.0,
                "player_recent_matches_30d_mean": 0.0,
                "player_recent_win_rate_30d_mean": 0.5,
                "player_recent_win_rate_30d_min": 0.5,
                "player_teams_seen_mean": 0.0,
                "player_transfers_total": 0.0,
                "membership_matches_mean": 0.0,
                "membership_matches_min": 0.0,
                "membership_win_rate_mean": 0.5,
                "membership_tenure_days_mean": 0.0,
                "membership_tenure_days_min": 0.0,
                "membership_newcomers": 0.0,
            }
        matches = [float(state.matches) for state in states]
        win_rates = [_smoothed_rate(state.wins, state.matches) for state in states]
        inactivity = [_age_days(at, state.last_at) for state in states]
        recent = [
            [item for item in state.outcomes if _age_days(at, item.at) < 30]
            for state in states
        ]
        recent_rates = [
            _smoothed_rate(sum(item.win for item in items), len(items))
            for items in recent
        ]
        membership_matches = [float(state.matches) for state in memberships]
        membership_rates = [
            _smoothed_rate(state.wins, state.matches) for state in memberships
        ]
        tenure = [
            0.0
            if state.first_at is None
            else max(0.0, (at - state.first_at).total_seconds() / 86_400)
            for state in memberships
        ]
        teams_seen = [float(len(state.teams)) for state in states]
        return {
            "player_matches_min": min(matches),
            "player_matches_max": max(matches),
            "player_matches_std": _std(matches),
            "player_win_rate_min": min(win_rates),
            "player_win_rate_max": max(win_rates),
            "player_win_rate_std": _std(win_rates),
            "player_inactivity_min": min(inactivity),
            "player_inactivity_max": max(inactivity),
            "player_inactivity_std": _std(inactivity),
            "player_recent_matches_30d_mean": _mean(
                [float(len(items)) for items in recent]
            ),
            "player_recent_win_rate_30d_mean": _mean(recent_rates, 0.5),
            "player_recent_win_rate_30d_min": min(recent_rates),
            "player_teams_seen_mean": _mean(teams_seen),
            "player_transfers_total": sum(
                max(0.0, value - 1.0) for value in teams_seen
            ),
            "membership_matches_mean": _mean(membership_matches),
            "membership_matches_min": min(membership_matches),
            "membership_win_rate_mean": _mean(membership_rates, 0.5),
            "membership_tenure_days_mean": _mean(tenure),
            "membership_tenure_days_min": min(tenure),
            "membership_newcomers": float(
                sum(state.matches == 0 for state in memberships)
            ),
        }

    def _lineup_features(
        self, team_id: int, roster: tuple[str, ...], at: datetime
    ) -> dict[str, float]:
        ids = self._ids(roster)
        roster_full5 = len(ids) == 5
        lineup = (
            self.lineups.get((team_id, ids), _AggregateState())
            if roster_full5
            else _AggregateState()
        )
        result = {
            f"lineup_{key}": value
            for key, value in _aggregate_features(lineup, at).items()
        }
        states = (
            [
                self.pairs.get((team_id, ids[left], ids[right]), _AggregateState())
                for left in range(len(ids))
                for right in range(left + 1, len(ids))
            ]
            if roster_full5
            else []
        )
        pair_matches = [float(state.matches) for state in states]
        pair_rates = [_smoothed_rate(state.wins, state.matches) for state in states]
        global_states = (
            [
                self.global_pairs.get((ids[left], ids[right]), _AggregateState())
                for left in range(len(ids))
                for right in range(left + 1, len(ids))
            ]
            if roster_full5
            else []
        )
        global_pair_matches = [float(state.matches) for state in global_states]
        result.update(
            {
                "pair_matches_mean": _mean(pair_matches),
                "pair_matches_min": min(pair_matches, default=0.0),
                "pair_matches_max": max(pair_matches, default=0.0),
                "pair_matches_std": _std(pair_matches),
                "pair_win_rate_mean": _mean(pair_rates, 0.5),
                "pair_win_rate_min": min(pair_rates, default=0.5),
                "global_pair_matches_mean": _mean(global_pair_matches),
                "global_pair_matches_min": min(global_pair_matches, default=0.0),
                "global_pair_matches_max": max(global_pair_matches, default=0.0),
            }
        )
        team = self.teams[team_id]
        previous = team.last_roster
        current = frozenset(ids)
        overlap = len(previous & current) if previous and roster_full5 else 0
        union = len(previous | current) if roster_full5 else 0
        result.update(
            {
                "roster_size_inferred": float(len(ids)),
                "roster_full5_inferred": float(roster_full5),
                "roster_overlap_previous": float(overlap),
                "roster_jaccard_previous": overlap / union if union else 0.0,
                "roster_changed": float(
                    bool(roster_full5 and previous and previous != current)
                ),
                "roster_changes_30d": float(
                    sum(_age_days(at, changed) < 30 for changed in team.roster_changes)
                ),
                "roster_changes_90d": float(
                    sum(_age_days(at, changed) < 90 for changed in team.roster_changes)
                ),
            }
        )
        return result

    def _context_features(
        self, team_id: int, match: dict[str, str], at: datetime
    ) -> dict[str, float]:
        contexts = {
            "tier": match["tournament_tier"],
            "venue": match["event_type"],
            "format": str(match["bo_type"]),
            "version": str(match["game_version"]),
            "bracket": str(match.get("bracket_type") or "unknown"),
            "tournament": str(match["tournament_id"]),
        }
        result: dict[str, float] = {}
        for kind, value in contexts.items():
            state = self.contexts[(team_id, kind, value)]
            result.update(
                {
                    f"context_{kind}_{key}": feature
                    for key, feature in _aggregate_features(state, at).items()
                }
            )
        return result

    def _map_features(self, team_id: int, at: datetime) -> dict[str, float]:
        counts: list[float] = []
        rates: list[float] = []
        distinct = {30: 0, 90: 0, 180: 0}
        for state in self.maps[team_id].values():
            recent_180 = [
                item for item in state.outcomes if _age_days(at, item.at) < 180
            ]
            if not recent_180:
                continue
            counts.append(float(len(recent_180)))
            rates.append(
                _smoothed_rate(sum(item.win for item in recent_180), len(recent_180))
            )
            for window in distinct:
                if any(_age_days(at, item.at) < window for item in recent_180):
                    distinct[window] += 1
        total = sum(counts)
        entropy = 0.0
        if total:
            shares = [count / total for count in counts]
            entropy = -sum(share * math.log(share) for share in shares if share)
            if len(shares) > 1:
                entropy /= math.log(len(shares))
        ordered = sorted(rates, reverse=True)
        return {
            "map_pool_maps_30d": float(distinct[30]),
            "map_pool_maps_90d": float(distinct[90]),
            "map_pool_maps_180d": float(distinct[180]),
            "map_pool_games_180d": total,
            "map_pool_entropy_180d": entropy,
            "map_pool_best_rate_180d": max(rates, default=0.5),
            "map_pool_worst_rate_180d": min(rates, default=0.5),
            "map_pool_rate_std_180d": _std(rates),
            "map_pool_top3_rate_180d": _mean(ordered[:3], 0.5),
            "map_pool_maps_3plus_180d": float(sum(count >= 3 for count in counts)),
            "map_pool_most_played_share_180d": max(counts, default=0.0) / total
            if total
            else 0.0,
        }

    def _map_matchup_features(
        self, team1_id: int, team2_id: int, at: datetime
    ) -> dict[str, float]:
        names = set(self.maps[team1_id]) | set(self.maps[team2_id])
        differences: list[float] = []
        weights: list[float] = []
        frequencies1: list[float] = []
        frequencies2: list[float] = []
        overlap = 0
        for name in names:
            recent: list[list[_AggregateOutcome]] = []
            for team_id in (team1_id, team2_id):
                state = self.maps[team_id].get(name)
                recent.append(
                    []
                    if state is None
                    else [
                        item for item in state.outcomes if _age_days(at, item.at) < 180
                    ]
                )
            if not recent[0] and not recent[1]:
                continue
            rates = [
                _smoothed_rate(sum(item.win for item in items), len(items))
                for items in recent
            ]
            differences.append(rates[0] - rates[1])
            weights.append(math.sqrt((len(recent[0]) + 1) * (len(recent[1]) + 1)))
            frequencies1.append(float(len(recent[0])))
            frequencies2.append(float(len(recent[1])))
            overlap += int(bool(recent[0]) and bool(recent[1]))
        total_weight = sum(weights)
        weighted_difference = (
            sum(weight * difference for weight, difference in zip(weights, differences))
            / total_weight
            if total_weight
            else 0.0
        )
        total1, total2 = sum(frequencies1), sum(frequencies2)
        style_distance = (
            sum(
                abs(
                    (left / total1 if total1 else 0.0)
                    - (right / total2 if total2 else 0.0)
                )
                for left, right in zip(frequencies1, frequencies2)
            )
            / 2.0
        )
        return {
            "diff_counter_map_pool_advantage_180d": weighted_difference,
            "counter_map_pool_matchup_range_180d": (
                max(differences) - min(differences) if differences else 0.0
            ),
            "counter_map_pool_matchup_std_180d": _std(differences),
            "counter_map_pool_overlap_180d": float(overlap),
            "counter_map_pool_style_distance_180d": style_distance,
        }

    def _selected_map_side_features(
        self, team_id: int, selected_maps: tuple[str, ...], at: datetime
    ) -> dict[str, float]:
        histories: list[list[_AggregateOutcome]] = []
        for name in selected_maps:
            state = self.maps.get(team_id, {}).get(name)
            histories.append(
                []
                if state is None
                else [item for item in state.outcomes if _age_days(at, item.at) < 180]
            )
        games = sum(len(history) for history in histories)
        wins = sum(item.win for history in histories for item in history)
        known_maps = sum(bool(history) for history in histories)
        rates = [
            _smoothed_rate(sum(item.win for item in history), len(history))
            for history in histories
        ]
        return {
            "veto_selected_map_games_180d": float(games),
            "veto_selected_maps_known_180d": float(known_maps),
            "veto_selected_map_known_fraction_180d": (
                known_maps / len(selected_maps) if selected_maps else 0.0
            ),
            "veto_selected_map_win_rate_180d": _smoothed_rate(wins, games),
            "veto_selected_map_rate_mean_180d": _mean(rates, 0.5),
            "veto_selected_map_rate_std_180d": _std(rates),
        }

    def _selected_map_matchup_features(
        self,
        team1_id: int,
        team2_id: int,
        selected_maps: tuple[str, ...],
        at: datetime,
    ) -> dict[str, float]:
        differences: list[float] = []
        overlap = 0
        union_games = 0
        for name in selected_maps:
            histories: list[list[_AggregateOutcome]] = []
            for team_id in (team1_id, team2_id):
                state = self.maps.get(team_id, {}).get(name)
                histories.append(
                    []
                    if state is None
                    else [
                        item for item in state.outcomes if _age_days(at, item.at) < 180
                    ]
                )
            rates = [
                _smoothed_rate(sum(item.win for item in history), len(history))
                for history in histories
            ]
            differences.append(rates[0] - rates[1])
            overlap += int(bool(histories[0]) and bool(histories[1]))
            union_games += len(histories[0]) + len(histories[1])
        return {
            "diff_counter_veto_selected_map_matchup_mean_180d": _mean(differences),
            "counter_veto_selected_map_matchup_range_180d": (
                max(differences) - min(differences) if differences else 0.0
            ),
            "counter_veto_selected_map_matchup_std_180d": _std(differences),
            "counter_veto_selected_map_overlap_180d": float(overlap),
            "counter_veto_selected_map_union_games_180d": float(union_games),
        }

    def _current_veto_features(
        self, match: dict[str, str], at: datetime
    ) -> dict[str, float | str]:
        result: dict[str, float | str] = {key: "missing" for key in _VETO_CATEGORY_KEYS}
        actions = _strict_veto_actions(match)
        result["counter_veto_complete"] = float(actions is not None)
        selected_maps: tuple[str, ...] = ()
        if actions is not None:
            team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
            side_by_team = {team1_id: "team1", team2_id: "team2"}
            slot_counts: dict[tuple[str, str], int] = defaultdict(int)
            selected_maps = tuple(
                action.map_name for action in actions if action.choice_type in {1, 3}
            )
            for action in actions:
                if action.choice_type == 3:
                    result["veto_decider_map"] = action.map_name
                    continue
                action_name = "pick" if action.choice_type == 1 else "ban"
                side = side_by_team[action.team_id]
                slot_counts[(side, action_name)] += 1
                slot = slot_counts[(side, action_name)]
                result[f"{side}_veto_{action_name}_{slot}_map"] = action.map_name
            for slot, map_name in enumerate(selected_maps, start=1):
                result[f"veto_selected_map_{slot}"] = map_name

        result["counter_veto_selected_map_count"] = float(len(selected_maps))
        team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
        self._add_sides(
            result,
            self._selected_map_side_features(team1_id, selected_maps, at),
            self._selected_map_side_features(team2_id, selected_maps, at),
        )
        result.update(
            self._selected_map_matchup_features(team1_id, team2_id, selected_maps, at)
        )
        return result

    def _h2h_features(
        self, team1_id: int, team2_id: int, at: datetime
    ) -> dict[str, float]:
        key = tuple(sorted((team1_id, team2_id)))
        state = self.h2h[key]
        orientation = 1.0 if key[0] == team1_id else -1.0
        result: dict[str, float] = {}
        for window in (90, 365):
            recent = [
                item for item in state.outcomes if _age_days(at, item.at) < window
            ]
            first_win = _smoothed_rate(sum(item.win for item in recent), len(recent))
            known_maps = [
                item.map_share for item in recent if item.map_share is not None
            ]
            known_rounds = [
                item.round_share for item in recent if item.round_share is not None
            ]
            first_map = _smoothed_rate(sum(known_maps), len(known_maps))
            first_round = _smoothed_rate(sum(known_rounds), len(known_rounds))
            team1_values = (
                (first_win, first_map, first_round)
                if orientation > 0
                else (1 - first_win, 1 - first_map, 1 - first_round)
            )
            result.update(
                {
                    f"counter_h2h_matches_{window}d": float(len(recent)),
                    f"counter_h2h_map_known_{window}d": float(len(known_maps)),
                    f"counter_h2h_round_known_{window}d": float(len(known_rounds)),
                    f"team1_counter_h2h_win_rate_{window}d": team1_values[0],
                    f"team2_counter_h2h_win_rate_{window}d": 1 - team1_values[0],
                    f"diff_counter_h2h_win_rate_{window}d": 2 * team1_values[0] - 1,
                    f"team1_counter_h2h_map_share_{window}d": team1_values[1],
                    f"team2_counter_h2h_map_share_{window}d": 1 - team1_values[1],
                    f"diff_counter_h2h_map_share_{window}d": 2 * team1_values[1] - 1,
                    f"team1_counter_h2h_round_share_{window}d": team1_values[2],
                    f"team2_counter_h2h_round_share_{window}d": 1 - team1_values[2],
                    f"diff_counter_h2h_round_share_{window}d": 2 * team1_values[2] - 1,
                }
            )
        result["counter_h2h_days_since"] = _age_days(at, state.last_at)
        return result

    @staticmethod
    def _add_sides(
        target: dict[str, float | str],
        left: dict[str, float],
        right: dict[str, float],
    ) -> None:
        for key in sorted(set(left) | set(right)):
            left_value = left.get(key, 0.0)
            right_value = right.get(key, 0.0)
            target[f"team1_counter_{key}"] = left_value
            target[f"team2_counter_{key}"] = right_value
            target[f"diff_counter_{key}"] = left_value - right_value

    def features(
        self,
        match: dict[str, str],
        at: datetime,
        roster1: tuple[str, ...],
        roster2: tuple[str, ...],
    ) -> dict[str, float | str]:
        team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
        result: dict[str, float | str] = {}
        for builder in (
            lambda team_id, roster: self._team_features(team_id, at),
            lambda team_id, roster: self._player_features(team_id, roster, at),
            lambda team_id, roster: self._lineup_features(team_id, roster, at),
            lambda team_id, roster: self._context_features(team_id, match, at),
            lambda team_id, roster: self._map_features(team_id, at),
            lambda team_id, roster: round_style_features(
                self.round_styles[team_id], at
            ),
        ):
            self._add_sides(
                result,
                builder(team1_id, roster1),
                builder(team2_id, roster2),
            )
        result.update(self._map_matchup_features(team1_id, team2_id, at))
        result.update(self._h2h_features(team1_id, team2_id, at))
        result.update(self._current_veto_features(match, at))
        return result

    def update(
        self,
        match: dict[str, str],
        at: datetime,
        roster1: tuple[str, ...],
        roster2: tuple[str, ...],
        *,
        team1_elo: float,
        team2_elo: float,
        team1_elo_delta: float,
    ) -> None:
        team1_id, team2_id = int(match["team1_id"]), int(match["team2_id"])
        outcome1 = int(match["team1_win"])
        map_total = int(match["team1_map_wins"]) + int(match["team2_map_wins"])
        map_result_valid = map_total > 0 and map_total == int(match["maps_played"])
        map_share1 = (
            int(match["team1_map_wins"]) / map_total if map_result_valid else None
        )
        round_total = int(match["team1_rounds"] or 0) + int(match["team2_rounds"] or 0)
        round_share1 = (
            int(match["team1_rounds"]) / round_total
            if int(match["rounds_known"]) and round_total
            else None
        )
        round_margin1 = (
            (int(match["team1_rounds"]) - int(match["team2_rounds"]))
            / int(match["maps_played"])
            if int(match["rounds_known"]) and int(match["maps_played"])
            else None
        )
        expected1 = 1.0 / (1.0 + 10.0 ** ((team2_elo - team1_elo) / 400.0))
        for (
            team_id,
            opponent_elo,
            roster,
            win,
            map_share,
            round_share,
            expected,
            delta,
            round_margin,
        ) in (
            (
                team1_id,
                team2_elo,
                roster1,
                float(outcome1),
                map_share1,
                round_share1,
                expected1,
                team1_elo_delta,
                round_margin1,
            ),
            (
                team2_id,
                team1_elo,
                roster2,
                float(1 - outcome1),
                None if map_share1 is None else 1 - map_share1,
                None if round_share1 is None else 1 - round_share1,
                1 - expected1,
                -team1_elo_delta,
                None if round_margin1 is None else -round_margin1,
            ),
        ):
            team = self.teams[team_id]
            team.outcomes.append(
                _Outcome(
                    at,
                    win,
                    map_share,
                    round_share,
                    opponent_elo,
                    expected,
                    delta,
                    map_total if map_result_valid else 0,
                    round_margin,
                    int(match["bo_type"]),
                )
            )
            while team.outcomes and _age_days(at, team.outcomes[0].at) >= 365:
                team.outcomes.popleft()
            ids = self._ids(roster)
            if len(ids) == 5:
                current_roster = frozenset(ids)
                if team.last_roster and team.last_roster != current_roster:
                    team.roster_changes.append(at)
                team.last_roster = current_roster
            while team.roster_changes and _age_days(at, team.roster_changes[0]) >= 365:
                team.roster_changes.popleft()

            aggregate_targets: list[_AggregateState] = []
            if len(ids) == 5:
                aggregate_targets.append(self.lineups[(team_id, ids)])
            aggregate_targets.extend(
                self.memberships[(team_id, player_id)] for player_id in ids
            )
            aggregate_targets.extend(
                self.pairs[(team_id, ids[left], ids[right])]
                for left in range(len(ids))
                for right in range(left + 1, len(ids))
            )
            aggregate_targets.extend(
                self.global_pairs[(ids[left], ids[right])]
                for left in range(len(ids))
                for right in range(left + 1, len(ids))
            )
            for kind, value in (
                ("tier", match["tournament_tier"]),
                ("venue", match["event_type"]),
                ("format", str(match["bo_type"])),
                ("version", str(match["game_version"])),
                ("bracket", str(match.get("bracket_type") or "unknown")),
                ("tournament", str(match["tournament_id"])),
            ):
                aggregate_targets.append(self.contexts[(team_id, kind, value)])
            for state in aggregate_targets:
                state.add(at, win, map_share, round_share)
            for player_id in ids:
                player = self.players[player_id]
                player.matches += 1
                player.wins += win
                player.first_at = player.first_at or at
                player.last_at = at
                player.teams.add(team_id)
                player.outcomes.append(
                    _AggregateOutcome(at, win, map_share, round_share)
                )
                while player.outcomes and _age_days(at, player.outcomes[0].at) >= 365:
                    player.outcomes.popleft()

        pair_key = tuple(sorted((team1_id, team2_id)))
        first_is_team1 = pair_key[0] == team1_id
        self.h2h[pair_key].add(
            at,
            float(outcome1 if first_is_team1 else 1 - outcome1),
            map_share1 if first_is_team1 or map_share1 is None else 1 - map_share1,
            round_share1
            if first_is_team1 or round_share1 is None
            else 1 - round_share1,
        )
        try:
            map_results = json.loads(match.get("map_results") or "[]")
        except (json.JSONDecodeError, TypeError):
            map_results = []
        if not isinstance(map_results, list):
            map_results = []
        for game in map_results:
            if not isinstance(game, dict):
                continue
            name = _canonical_map_name(game.get("map_name"))
            winner = int(game.get("winner_team_id") or 0)
            loser = int(game.get("loser_team_id") or 0)
            raw_winner_score = game.get("winner_score")
            raw_loser_score = game.get("loser_score")
            scores_known = isinstance(raw_winner_score, int) and isinstance(
                raw_loser_score, int
            )
            winner_score = int(raw_winner_score or 0)
            loser_score = int(raw_loser_score or 0)
            total = winner_score + loser_score if scores_known else 0
            if name is None or {winner, loser} != {team1_id, team2_id}:
                continue
            for team_id, win, share in (
                (winner, 1.0, winner_score / total if total else None),
                (loser, 0.0, loser_score / total if total else None),
            ):
                state = self.maps[team_id].setdefault(name, _AggregateState())
                state.add(at, win, win, share)
                round_stats = round_stats_for_team(game, team_id)
                if round_stats is not None:
                    style = self.round_styles[team_id]
                    style.append(RoundStyleOutcome(at, round_stats))
                    while style and _age_days(at, style[0].at) >= 365:
                        style.popleft()
