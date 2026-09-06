"""Compact, interpretable rule families selected by causal discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import pandas as pd


def _predicate(
    frame: pd.DataFrame, feature: str, operator: str, threshold: float
) -> np.ndarray:
    values = pd.to_numeric(frame[feature], errors="coerce")
    if operator == "le":
        return values.le(threshold).fillna(False).to_numpy(dtype=bool)
    if operator == "ge":
        return values.ge(threshold).fillna(False).to_numpy(dtype=bool)
    raise ValueError(f"Unsupported operator: {operator}")


@dataclass(frozen=True)
class RuleVariant:
    """One threshold variant inside an economic rule family."""

    rule_name: str
    feature: str
    operator: str
    threshold: float

    @property
    def variant_id(self) -> str:
        return f"{self.rule_name}:{self.feature}:{self.operator}:{self.threshold:g}"

    @property
    def features(self) -> tuple[str, ...]:
        return (self.feature,)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return _predicate(frame, self.feature, self.operator, self.threshold)


@dataclass(frozen=True)
class CompositeRuleVariant:
    """A predeclared AND archetype whose components remain auditable."""

    rule_name: str
    conditions: tuple[RuleVariant, ...]

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("Composite rule должен содержать минимум два условия")

    @property
    def variant_id(self) -> str:
        specification = "&".join(
            f"{item.feature}:{item.operator}:{item.threshold:g}"
            for item in self.conditions
        )
        return f"{self.rule_name}:{specification}"

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            feature for condition in self.conditions
            for feature in condition.features
        ))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.logical_and.reduce([
            condition.predict(frame) for condition in self.conditions
        ])


class RuleCandidateProtocol(Protocol):
    rule_name: str

    @property
    def variant_id(self) -> str: ...

    @property
    def features(self) -> tuple[str, ...]: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


RuleCandidate: TypeAlias = RuleVariant | CompositeRuleVariant


def _variants(
    rule_name: str,
    feature: str,
    operator: str,
    thresholds: tuple[float, ...],
) -> list[RuleVariant]:
    return [
        RuleVariant(rule_name, feature, operator, float(threshold))
        for threshold in thresholds
    ]


def _and_grid(
    rule_name: str,
    left: list[RuleVariant],
    right: list[RuleVariant],
) -> list[CompositeRuleVariant]:
    return [
        CompositeRuleVariant(rule_name, (left_item, right_item))
        for left_item in left for right_item in right
    ]


def build_rule_library() -> dict[str, tuple[RuleCandidate, ...]]:
    """Build fixed families; only their variants are fitted inside WF.

    Correlated cheapness measures are alternatives inside one engine. The four
    combinations are economic hypotheses, not an unrestricted pairwise search.
    """
    cheapness: list[RuleVariant] = []
    for window in (30, 90, 180):
        cheapness.extend(_variants(
            "relative_cheapness", f"percentile_{window}d", "le",
            (0.10, 0.15, 0.20),
        ))
        cheapness.extend(_variants(
            "relative_cheapness", f"zscore_{window}d", "le",
            (-1.50, -1.00, -0.50),
        ))
        cheapness.extend(_variants(
            "relative_cheapness", f"distance_from_low_{window}d_bps", "le",
            (25.0, 50.0, 100.0),
        ))

    momentum: list[RuleVariant] = []
    for window in (1, 3, 5, 10, 20):
        momentum.extend(_variants(
            "negative_momentum", f"return_{window}d_bps", "le",
            (-100.0, -50.0, 0.0),
        ))
    down_streak = _variants(
        "down_streak", "consecutive_down", "ge", (2.0, 3.0, 4.0)
    )
    negative_surprise = _variants(
        "negative_surprise", "return_surprise_z_60d", "le",
        (-2.0, -1.0, -0.5),
    )
    trend_down: list[RuleVariant] = []
    for window in (3, 5, 10):
        trend_down.extend(_variants(
            "trend_down", f"slope_{window}d_bps_per_day", "le",
            (-50.0, -25.0, 0.0),
        ))
    kalman_downtrend = _variants(
        "kalman_downtrend", "kalman_trend_z", "le", (-2.0, -1.0, -0.5)
    )

    reversal: list[RuleVariant] = []
    for window in (20, 60, 120):
        reversal.extend(_variants(
            "reversal_from_low", f"bounce_from_prior_low_{window}d_bps", "ge",
            (20.0, 50.0, 100.0),
        ))
    reversal.extend(_variants(
        "reversal_from_low", "kalman_reversal_score", "ge", (0.5, 1.0, 2.0)
    ))

    # Composites use only representative 90-day state variants. This prevents
    # the broad discovery Cartesian grid from becoming production fitting.
    composite_cheapness = [
        *_variants(
            "cheapness_component", "percentile_90d", "le", (0.10, 0.15, 0.20)
        ),
        *_variants(
            "cheapness_component", "zscore_90d", "le", (-1.50, -1.00, -0.50)
        ),
    ]
    downward_pressure = [
        *_variants(
            "pressure_component", "return_3d_bps", "le", (-50.0, 0.0)
        ),
        *_variants(
            "pressure_component", "return_5d_bps", "le", (-50.0, 0.0)
        ),
        *_variants(
            "pressure_component", "consecutive_down", "ge", (2.0, 3.0)
        ),
    ]
    composite_surprise = _variants(
        "surprise_component", "return_surprise_z_60d", "le",
        (-2.0, -1.0, -0.5),
    )
    composite_streak = _variants(
        "streak_component", "consecutive_down", "ge", (2.0, 3.0, 4.0)
    )
    composite_kalman = _variants(
        "kalman_component", "kalman_trend_z", "le", (-2.0, -1.0, -0.5)
    )
    composite_reversal: list[RuleVariant] = []
    for window in (20, 60):
        composite_reversal.extend(_variants(
            "reversal_component", f"bounce_from_prior_low_{window}d_bps", "ge",
            (20.0, 50.0),
        ))

    library: dict[str, tuple[RuleCandidate, ...]] = {
        "relative_cheapness": tuple(cheapness),
        "negative_momentum": tuple(momentum),
        "down_streak": tuple(down_streak),
        "negative_surprise": tuple(negative_surprise),
        "trend_down": tuple(trend_down),
        "kalman_downtrend": tuple(kalman_downtrend),
        "reversal_from_low": tuple(reversal),
        "cheapness_and_downward_pressure": tuple(_and_grid(
            "cheapness_and_downward_pressure",
            composite_cheapness, downward_pressure,
        )),
        "cheapness_and_negative_surprise": tuple(_and_grid(
            "cheapness_and_negative_surprise",
            composite_cheapness, composite_surprise,
        )),
        "persistent_kalman_downtrend": tuple(_and_grid(
            "persistent_kalman_downtrend",
            composite_streak, composite_kalman,
        )),
        "cheapness_and_reversal": tuple(_and_grid(
            "cheapness_and_reversal",
            composite_cheapness, composite_reversal,
        )),
    }
    for family, candidates in library.items():
        identifiers = [candidate.variant_id for candidate in candidates]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise AssertionError(f"Некорректная rule family: {family}")
    return library


RULE_LIBRARY = build_rule_library()
RULE_FEATURES = tuple(sorted({
    feature
    for variants in RULE_LIBRARY.values()
    for variant in variants
    for feature in variant.features
}))
