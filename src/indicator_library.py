"""The exact indicator parameter spaces used by research and production fit."""

from __future__ import annotations

from itertools import combinations, product

from .indicators import IndicatorCandidate, IndicatorRule, create_indicator_rules


def _rules_for_windows(
    family: str,
    feature_template: str,
    windows: tuple[int, ...],
    operator: str,
    thresholds: tuple[float, ...],
) -> list[IndicatorRule]:
    rules: list[IndicatorRule] = []
    for window in windows:
        rules.extend(create_indicator_rules(
            family,
            feature_template.format(window=window),
            operator,
            thresholds,
        ))
    return rules


def build_indicator_spaces() -> dict[str, list[IndicatorCandidate]]:
    """Build the same named architectures and grids as the research notebook."""
    rule_spaces = {
        "level_low": _rules_for_windows("level_low", "percentile_{window}d", (20, 30, 60, 90, 120, 180), "le", (.05, .10, .15, .20)),
        "near_low": _rules_for_windows("near_low", "distance_from_low_{window}d_bps", (20, 30, 60, 90, 120, 180), "le", (25, 50, 100)),
        "momentum_down": _rules_for_windows("momentum_down", "return_{window}d_bps", (1, 3, 5, 10, 20), "le", (-200, -100, -50, 0)),
        "momentum_up": _rules_for_windows("momentum_up", "return_{window}d_bps", (1, 3, 5, 10, 20), "ge", (0, 50, 100, 200)),
        "down_streak": create_indicator_rules("down_streak", "consecutive_down", "ge", (2, 3, 4, 5)),
        "up_streak": create_indicator_rules("up_streak", "consecutive_up", "ge", (2, 3, 4, 5)),
        "trend_down": _rules_for_windows("trend_down", "slope_{window}d_bps_per_day", (3, 5, 10), "le", (-50, 0)),
        "trend_up": _rules_for_windows("trend_up", "slope_{window}d_bps_per_day", (3, 5, 10), "ge", (0, 50)),
        "high_volatility": create_indicator_rules("high_volatility", "volatility_ratio_7d_30d", "ge", (.8, 1.0, 1.2, 1.5)),
    }
    spaces = {
        name: [IndicatorCandidate((rule,)) for rule in rules]
        for name, rules in rule_spaces.items()
    }
    for left_name, right_name in combinations(rule_spaces, 2):
        for logic in ("AND", "OR"):
            spaces[f"{left_name}__{logic}__{right_name}"] = [
                IndicatorCandidate((left, right), logic=logic)
                for left, right in product(rule_spaces[left_name], rule_spaces[right_name])
            ]

    for context_currency in ("usd", "eur", "cny"):
        for direction, operator, thresholds in (
            ("down", "le", (-100, -50, 0)),
            ("up", "ge", (0, 50, 100)),
        ):
            name = f"alexander_context_{context_currency}_{direction}"
            spaces[name] = [IndicatorCandidate((rule,)) for rule in create_indicator_rules(
                name, f"{context_currency}_return_5d_bps", operator, thresholds
            )]
    spaces["alexander_acceleration_down"] = [IndicatorCandidate((rule,)) for rule in create_indicator_rules("alexander_acceleration_down", "acceleration_1d_bps", "le", (-100, -50, 0))]
    spaces["alexander_acceleration_up"] = [IndicatorCandidate((rule,)) for rule in create_indicator_rules("alexander_acceleration_up", "acceleration_1d_bps", "ge", (0, 50, 100))]
    spaces["alexander_low_volatility"] = [IndicatorCandidate((rule,)) for rule in create_indicator_rules("alexander_low_volatility", "rolling_std_20d_bps", "le", (10, 25, 50, 100))]
    for name, feature in {
        "alexander_recipient_preholiday": "recipient_preholiday_7",
        "alexander_recipient_postholiday": "recipient_postholiday_3",
        "alexander_russia_preholiday": "russia_preholiday_7",
    }.items():
        spaces[name] = [IndicatorCandidate((IndicatorRule(name, feature, "ge", 1.0),))]
    spaces["alexander_stability"] = [
        IndicatorCandidate((
            IndicatorRule("alexander_stability_level", f"percentile_{window}d", "le", level),
            IndicatorRule("alexander_stability_move", "absolute_return_1d_bps", "le", move),
            IndicatorRule("alexander_stability_trend", "return_3d_bps", "le", trend),
        ), logic="AND")
        for window, level, move, trend in product((20, 60, 120), (.10, .20), (10, 20, 40), (-25, 0))
    ]
    spaces["alexander_reversal"] = [
        IndicatorCandidate((
            IndicatorRule("alexander_reversal_prior_low", "previous_distance_from_low_20d_bps", "le", near_low),
            IndicatorRule("alexander_reversal_bounce", "bounce_from_prior_low_20d_bps", "ge", bounce),
        ), logic="AND")
        for near_low, bounce in product((0, 25, 50, 100), (20, 40, 75, 100))
    ]
    return spaces
