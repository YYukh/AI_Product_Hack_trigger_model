"""Small interpretable rule library; no combinatorial AND/OR search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleVariant:
    rule_name: str
    feature: str
    operator: str
    threshold: float

    @property
    def variant_id(self) -> str:
        return f"{self.rule_name}:{self.feature}:{self.operator}:{self.threshold:g}"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame[self.feature], errors="coerce")
        if self.operator == "le":
            return values.le(self.threshold).fillna(False).to_numpy(dtype=bool)
        if self.operator == "ge":
            return values.ge(self.threshold).fillna(False).to_numpy(dtype=bool)
        raise ValueError(f"Unsupported operator: {self.operator}")


def build_rule_library() -> dict[str, tuple[RuleVariant, ...]]:
    """Six predeclared economic ideas with modest parameter grids."""
    specs: dict[str, list[RuleVariant]] = {
        "level_low": [],
        "near_low": [],
        "momentum_down": [],
        "reversal": [],
        "trend_down": [],
        "high_volatility": [],
    }
    for window in (30, 60, 90, 180):
        for threshold in (0.05, 0.10, 0.15, 0.20):
            specs["level_low"].append(RuleVariant(
                "level_low", f"percentile_{window}d", "le", threshold
            ))
    for window in (30, 60, 90, 180):
        for threshold in (25.0, 50.0, 100.0):
            specs["near_low"].append(RuleVariant(
                "near_low", f"distance_from_low_{window}d_bps", "le", threshold
            ))
    for window in (1, 3, 5, 10, 20):
        for threshold in (-200.0, -100.0, -50.0, 0.0):
            specs["momentum_down"].append(RuleVariant(
                "momentum_down", f"return_{window}d_bps", "le", threshold
            ))
    for window in (20, 60, 120):
        for threshold in (20.0, 40.0, 75.0, 100.0):
            specs["reversal"].append(RuleVariant(
                "reversal", f"bounce_from_prior_low_{window}d_bps", "ge", threshold
            ))
    for window in (3, 5, 10):
        for threshold in (-50.0, -25.0, 0.0):
            specs["trend_down"].append(RuleVariant(
                "trend_down", f"slope_{window}d_bps_per_day", "le", threshold
            ))
    for threshold in (0.8, 1.0, 1.2, 1.5):
        specs["high_volatility"].append(RuleVariant(
            "high_volatility", "volatility_ratio_7d_30d", "ge", threshold
        ))
    return {name: tuple(variants) for name, variants in specs.items()}


RULE_LIBRARY = build_rule_library()
RULE_FEATURES = tuple(sorted({
    variant.feature
    for variants in RULE_LIBRARY.values()
    for variant in variants
}))

