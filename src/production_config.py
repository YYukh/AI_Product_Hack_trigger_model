"""Frozen architecture/cadence choices for the production signal engines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicator_library import build_indicator_spaces
from .ml_backtest import MLIndicatorConfig


PRODUCTION_CURRENCIES = ("AMD", "KZT", "KGS", "TJS", "UZS")
PRODUCTION_TARGET_FAMILIES = ("G0", "W1")
PRODUCTION_HORIZONS = (1, 3, 5, 10, 20)
PRODUCTION_FIRST_SCORE_DATE = "2025-01-01"
TRAIN_WINDOW_MONTHS = 24
RULE_MIN_SIGNALS_PER_WEEK = 2.0

ML_MODEL_TYPE = "hist_gradient_boosting"
ML_RETRAIN_MONTHS = 12
ML_VALIDATION_MONTHS = 12
ML_MIN_SIGNALS_PER_WEEK = 2.0
ML_CONFIG = MLIndicatorConfig()


@dataclass(frozen=True)
class FixedIndicatorConfig:
    currency: str
    scenario: str
    target_family: str
    target: str
    horizon: int
    indicator: str
    retrain_months: int


def _config(
    currency: str,
    family: str,
    horizon: int,
    indicator: str,
    retrain_months: int,
) -> FixedIndicatorConfig:
    if family == "G0":
        scenario = "GOOD_NOW"
        target = f"target_g0_exact_min_h{horizon}d"
    elif family == "W1":
        scenario = "WINDOW_CLOSING"
        target = f"target_w1_lowpct_0p15_deterioration_75bps_h{horizon}d"
    else:
        raise ValueError(f"Unsupported family: {family}")
    return FixedIndicatorConfig(
        currency, scenario, family, target, horizon, indicator, retrain_months
    )


# Winners are architecture + retraining cadence. Concrete thresholds are fit
# again on matured past data at the start of every production WF period.
FIXED_INDICATORS = (
    _config("AMD", "G0", 1, "level_low__OR__momentum_down", 3),
    _config("AMD", "G0", 3, "near_low__OR__momentum_down", 3),
    _config("AMD", "G0", 5, "momentum_down__OR__trend_down", 6),
    _config("AMD", "G0", 10, "trend_down", 3),
    _config("AMD", "G0", 20, "momentum_down__OR__momentum_up", 3),
    _config("KGS", "G0", 1, "level_low__OR__momentum_down", 3),
    _config("KGS", "G0", 3, "momentum_down", 3),
    _config("KGS", "G0", 5, "momentum_down", 3),
    _config("KGS", "G0", 10, "down_streak__OR__trend_down", 3),
    _config("KGS", "G0", 20, "momentum_down__OR__up_streak", 12),
    _config("KZT", "G0", 1, "level_low__OR__momentum_down", 3),
    _config("KZT", "G0", 3, "level_low__OR__momentum_down", 3),
    _config("KZT", "G0", 5, "momentum_down__AND__trend_down", 6),
    _config("KZT", "G0", 10, "momentum_down__OR__down_streak", 6),
    _config("KZT", "G0", 20, "momentum_down__OR__down_streak", 12),
    _config("TJS", "G0", 1, "level_low__OR__momentum_down", 3),
    _config("TJS", "G0", 3, "momentum_down", 3),
    _config("TJS", "G0", 5, "momentum_down", 3),
    _config("TJS", "G0", 10, "momentum_down", 3),
    _config("TJS", "G0", 20, "momentum_down__OR__momentum_up", 3),
    _config("UZS", "G0", 1, "level_low__OR__momentum_down", 3),
    _config("UZS", "G0", 3, "down_streak__OR__trend_down", 3),
    _config("UZS", "G0", 5, "alexander_context_usd_down", 3),
    _config("UZS", "G0", 10, "alexander_context_usd_down", 3),
    _config("UZS", "G0", 20, "near_low__OR__high_volatility", 3),
    _config("AMD", "W1", 1, "momentum_down", 12),
    _config("AMD", "W1", 3, "momentum_down__OR__momentum_up", 3),
    _config("AMD", "W1", 5, "level_low__OR__high_volatility", 12),
    _config("AMD", "W1", 10, "level_low__OR__high_volatility", 6),
    _config("AMD", "W1", 20, "momentum_down__OR__trend_down", 3),
    _config("KGS", "W1", 1, "alexander_context_usd_down", 3),
    _config("KGS", "W1", 3, "level_low__OR__momentum_up", 3),
    _config("KGS", "W1", 5, "level_low__OR__momentum_up", 3),
    _config("KGS", "W1", 10, "alexander_context_usd_down", 3),
    _config("KGS", "W1", 20, "trend_down", 3),
    _config("KZT", "W1", 1, "momentum_down__OR__trend_down", 3),
    _config("KZT", "W1", 3, "level_low__OR__high_volatility", 3),
    _config("KZT", "W1", 5, "momentum_down__OR__trend_down", 6),
    _config("KZT", "W1", 10, "momentum_down__OR__trend_down", 3),
    _config("KZT", "W1", 20, "momentum_down__OR__trend_down", 6),
    _config("TJS", "W1", 1, "momentum_down", 3),
    _config("TJS", "W1", 3, "momentum_down__OR__high_volatility", 3),
    _config("TJS", "W1", 5, "momentum_down", 12),
    _config("TJS", "W1", 10, "level_low__OR__momentum_up", 12),
    _config("TJS", "W1", 20, "level_low__OR__momentum_up", 12),
    _config("UZS", "W1", 1, "level_low__OR__high_volatility", 6),
    _config("UZS", "W1", 3, "level_low__OR__high_volatility", 12),
    _config("UZS", "W1", 5, "level_low__OR__high_volatility", 12),
    _config("UZS", "W1", 10, "level_low__OR__high_volatility", 3),
    _config("UZS", "W1", 20, "level_low__OR__high_volatility", 12),
)

INDICATOR_SPACES = build_indicator_spaces()

# Continuous components of the selected G0/W1 indicator architectures.
ML_FEATURE_NAMES = tuple(sorted({
    rule.feature
    for config in FIXED_INDICATORS
    for candidate in INDICATOR_SPACES[config.indicator]
    for rule in candidate.rules
}))


def fixed_indicator_registry() -> pd.DataFrame:
    return pd.DataFrame([{
        "currency": config.currency,
        "scenario": config.scenario,
        "target_family": config.target_family,
        "target": config.target,
        "horizon": config.horizon,
        "indicator": config.indicator,
        "parameter_variants": len(INDICATOR_SPACES[config.indicator]),
        "retrain_months": config.retrain_months,
    } for config in FIXED_INDICATORS])
