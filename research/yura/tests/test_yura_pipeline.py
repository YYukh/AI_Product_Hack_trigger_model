from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.yura.src.arbiter import (
    ArbiterConfig, _remove_horizon_dominance, run_arbiter,
)
from research.yura.src.config import YuraPipelineConfig
from research.yura.src.engines import _target_definitions, _train_slice
from research.yura.src.evidence import (
    _rolling_percentile_rank, add_causal_relative_scores,
)
from research.yura.src.rules import RuleVariant
from research.yura.src.targets import build_yura_targets
from research.yura.src.temporal import TemporalPlan


class RuleTests(unittest.TestCase):
    def test_rule_variant_is_deterministic_and_nan_safe(self):
        frame = pd.DataFrame({"feature": [-1.0, 0.0, 1.0, None]})
        rule = RuleVariant("test", "feature", "le", 0.0)
        self.assertEqual(rule.predict(frame).tolist(), [True, True, False, False])


class TemporalTests(unittest.TestCase):
    def test_train_slice_purges_unmatured_labels(self):
        data = pd.DataFrame({
            "available_at": pd.to_datetime(["2023-12-01", "2023-12-20", "2023-12-31"]),
            "currency": ["AMD", "AMD", "AMD"],
            "target": [1, 0, 1],
        })
        train = _train_slice(
            data, target="target", horizon=20,
            retrain_at=pd.Timestamp("2024-01-01"), train_months=36,
        )
        self.assertEqual(train["available_at"].dt.strftime("%Y-%m-%d").tolist(), ["2023-12-01"])

    def test_target_registry_keeps_only_g0_and_w1(self):
        rows = [
            {"family": family, "horizon": horizon}
            for family in ("G0", "W1", "IRRELEVANT")
            for horizon in (1, 3)
        ]
        config = YuraPipelineConfig(horizons=(1,))
        selected = _target_definitions(pd.DataFrame(rows), config)
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["family"]), {"G0", "W1"})

    def test_temporal_plan_is_derived_from_data(self):
        data = pd.DataFrame({
            "available_at": pd.to_datetime(["2018-01-15", "2026-09-05"]),
        })
        plan = TemporalPlan.from_data(data, YuraPipelineConfig())
        self.assertEqual(plan.base_oos_start, pd.Timestamp("2021-01-01"))
        self.assertEqual(plan.selector_validation_start, pd.Timestamp("2022-01-01"))
        self.assertEqual(plan.holdout_start, pd.Timestamp("2025-01-01"))

    def test_w1_uses_strict_future_window_median(self):
        frame = pd.DataFrame({
            "available_at": pd.date_range("2020-01-01", periods=5, freq="D"),
            "currency": "AMD",
            "rate": [100.0, 102.0, 104.0, 98.0, 110.0],
            "centered_min_rate_2d": [None, None, 98.0, None, None],
        })
        labelled, registry = build_yura_targets(
            frame, horizons=(2,), w1_forward_bps=100.0,
        )
        w1 = registry.loc[registry["family"].eq("W1"), "name"].item()
        # At t0 the future median is median(102, 104)=103, i.e. +300 BPS.
        self.assertEqual(int(labelled.loc[0, w1]), 1)
        # The current 100 is not included in that future median.
        self.assertEqual(labelled.loc[0, "future_median_rate_2d"], 103.0)


class PolicyTests(unittest.TestCase):
    def test_policy_enforces_currency_rolling_cap(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        candidates = pd.DataFrame({
            "available_at": dates,
            "currency": "AMD",
            "scenario": "GOOD_NOW",
            "target_family": "G0",
            "target": "target_g0_exact_min_h1d",
            "horizon": 1,
            "engine_type": "ml",
            "engine_name": "pooled",
            "confidence": 0.8,
            "baseline_probability": 0.2,
            "confidence_lift": 2.0,
            "expected_bps": 20.0,
        })
        config = ArbiterConfig(
            rule_min_lift=1.0, ml_min_lift=1.0, min_expected_bps=0.0,
            min_decision_score=0.0, cooldown_days=0, max_signals_per_7d=2,
        )
        events = run_arbiter(candidates, config)
        emitted = pd.to_datetime(events["available_at"])
        for when in emitted:
            self.assertLessEqual(
                int(emitted.between(when - pd.Timedelta(days=6), when).sum()), 2
            )

    def test_pareto_filter_keeps_only_genuinely_useful_slow_horizon(self):
        base = {
            "available_at": pd.Timestamp("2025-01-01"),
            "currency": "AMD", "scenario": "GOOD_NOW", "target_family": "G0",
        }
        candidates = pd.DataFrame([
            {**base, "horizon": 1, "statistical_evidence": 2.0, "economic_evidence": 3.0},
            {**base, "horizon": 5, "statistical_evidence": 1.0, "economic_evidence": 2.0},
            {**base, "horizon": 20, "statistical_evidence": 3.0, "economic_evidence": 4.0},
        ])
        filtered = _remove_horizon_dominance(candidates)
        self.assertEqual(filtered["horizon"].tolist(), [1, 20])

    def test_relative_score_uses_no_future_row(self):
        frame = pd.DataFrame({
            "available_at": pd.date_range("2024-01-01", periods=3, freq="D"),
            "currency": "AMD", "scenario": "GOOD_NOW", "target_family": "G0",
            "horizon": 1, "confidence": [0.3, 0.4, 0.9],
            "baseline_probability": 0.2, "expected_bps": [10.0, 20.0, 100.0],
        })
        first_two = add_causal_relative_scores(frame.iloc[:2])
        all_rows = add_causal_relative_scores(frame)
        pd.testing.assert_series_equal(
            first_two["decision_score"], all_rows.iloc[:2]["decision_score"],
            check_names=False,
        )

    def test_relative_rank_drops_history_outside_window(self):
        dates = pd.Series(pd.to_datetime([
            "2020-01-01", "2024-01-01", "2024-02-01",
        ]))
        values = pd.Series([100.0, 1.0, 2.0])
        ranks = _rolling_percentile_rank(dates, values, window_months=12)
        self.assertEqual(ranks.tolist(), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
