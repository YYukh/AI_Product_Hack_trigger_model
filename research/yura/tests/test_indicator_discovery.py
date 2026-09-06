from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.yura.src.advanced_indicators import (
    ADVANCED_FEATURES, PRODUCTION_ADVANCED_FEATURES,
    add_advanced_indicator_features, add_production_indicator_features,
)
from research.yura.src.discovery_targets import build_discovery_targets
from research.yura.src.indicator_discovery import (
    IndicatorDiscoveryConfig, build_discovery_temporal_plan,
    summarize_discovery_folds,
)


class AdvancedFeatureTests(unittest.TestCase):
    def _frame(self, periods: int) -> pd.DataFrame:
        index = np.arange(periods, dtype=float)
        rate = 100.0 + 0.01 * index + np.sin(index / 9.0)
        return pd.DataFrame({
            "available_at": pd.date_range("2020-01-01", periods=periods, freq="D"),
            "currency": "AMD",
            "rate": rate,
            "return_3d_bps": pd.Series(rate).pct_change(3) * 10_000,
            "rolling_std_7d_bps": pd.Series(rate).pct_change().mul(10_000).rolling(7).std(),
            "rolling_std_30d_bps": pd.Series(rate).pct_change().mul(10_000).rolling(30).std(),
            "slope_3d_bps_per_day": pd.Series(rate).pct_change(3).mul(10_000 / 3),
            "slope_10d_bps_per_day": pd.Series(rate).pct_change(10).mul(1_000),
            "zscore_90d": (
                pd.Series(rate) - pd.Series(rate).rolling(90).mean()
            ) / pd.Series(rate).rolling(90).std(),
        })

    def test_advanced_features_do_not_change_when_future_is_appended(self):
        full = add_advanced_indicator_features(self._frame(220))
        truncated = add_advanced_indicator_features(self._frame(170))
        np.testing.assert_allclose(
            full.loc[:169, list(ADVANCED_FEATURES)].to_numpy(float),
            truncated.loc[:, list(ADVANCED_FEATURES)].to_numpy(float),
            equal_nan=True,
        )

    def test_production_subset_is_causal_and_complete(self):
        full = add_production_indicator_features(self._frame(220))
        truncated = add_production_indicator_features(self._frame(170))
        self.assertTrue(set(PRODUCTION_ADVANCED_FEATURES).issubset(full.columns))
        np.testing.assert_allclose(
            full.loc[:169, list(PRODUCTION_ADVANCED_FEATURES)].to_numpy(float),
            truncated.loc[:, list(PRODUCTION_ADVANCED_FEATURES)].to_numpy(float),
            equal_nan=True,
        )


class DiscoveryTargetTests(unittest.TestCase):
    def test_registry_contains_four_legacy_concepts_and_yura_w1(self):
        dates = pd.date_range("2020-01-01", periods=8, freq="D")
        rates = np.array([100, 99, 98, 99, 101, 102, 101, 103], dtype=float)
        frame = pd.DataFrame({
            "available_at": dates,
            "currency": "AMD",
            "rate": rates,
            "centered_min_rate_1d": pd.Series(rates).rolling(3, center=True).min(),
            "future_best_regret_1d_bps": 0.0,
            "future_return_1d_bps": pd.Series(rates).pct_change().shift(-1) * 10_000,
            "percentile_90d": 0.10,
        })
        labelled, registry = build_discovery_targets(frame, horizons=(1,))
        self.assertEqual(
            set(registry["family"]),
            {"G0", "G1", "W0", "W1"},
        )
        self.assertTrue(set(registry["name"]).issubset(labelled.columns))


class DiscoverySummaryTests(unittest.TestCase):
    def test_temporal_plan_reserves_everything_after_four_oos_years(self):
        frame = pd.DataFrame({
            "available_at": pd.date_range("2018-01-01", "2026-09-01", freq="D")
        })
        plan = build_discovery_temporal_plan(frame, IndicatorDiscoveryConfig())
        self.assertEqual(plan.test_start.min(), pd.Timestamp("2021-01-01"))
        self.assertEqual(plan.discovery_end.iloc[0], pd.Timestamp("2025-01-01"))
        self.assertEqual(len(plan), 8)

    def test_low_support_perfect_precision_gets_conservative_lcb(self):
        folds = pd.DataFrame({
            "fold_id": [1, 2], "currency": "AMD", "target_family": "G0",
            "target": "target", "horizon": 1, "strategy_kind": "single",
            "strategy_name": "rare", "logic": "SINGLE",
            "selected_spec": ["a", "a"],
            "test_observations": [100, 100], "test_positive_count": [10, 10],
            "test_signal_count": [1, 1], "test_true_positive": [1, 1],
            "test_signal_benefit_sum": [10.0, 10.0],
            "test_signal_benefit_count": [1, 1],
            "test_random_benefit_sum": [0.0, 0.0],
            "test_random_benefit_count": [100, 100], "test_weeks": [26.0, 26.0],
        })
        leaderboard, _, _ = summarize_discovery_folds(folds, min_oos_signals=20)
        row = leaderboard.iloc[0]
        self.assertEqual(row.quality_group, "insufficient_support")
        self.assertLess(row.oos_precision_lcb95, row.oos_precision)


if __name__ == "__main__":
    unittest.main()
