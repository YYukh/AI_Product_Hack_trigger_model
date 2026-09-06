from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arbiter import (
    ArbiterConfig, _remove_horizon_dominance, run_arbiter,
)
from src import benchmark as benchmark_module
from src.config import YuraPipelineConfig
from src.engines import (
    _group_balanced_weights, _local_training_baseline,
    _target_definitions, _train_slice,
)
from src.evidence import (
    _rolling_percentile_rank, add_causal_relative_scores,
    aggregate_engine_evidence,
)
from src.learned_selector import (
    LearnedOpportunitySelector, _mature_slice, _selector_periods,
)
from src.rules import (
    RULE_LIBRARY, CompositeRuleVariant, RuleVariant,
)
from src.selector import ThresholdSelector, build_opportunity_selector
from src.targets import build_yura_targets
from src.temporal import TemporalPlan


class RuleTests(unittest.TestCase):
    def test_rule_variant_is_deterministic_and_nan_safe(self):
        frame = pd.DataFrame({"feature": [-1.0, 0.0, 1.0, None]})
        rule = RuleVariant("test", "feature", "le", 0.0)
        self.assertEqual(rule.predict(frame).tolist(), [True, True, False, False])

    def test_selected_rule_library_contains_only_predeclared_archetypes(self):
        self.assertEqual(set(RULE_LIBRARY), {
            "relative_cheapness", "negative_momentum", "down_streak",
            "negative_surprise", "trend_down", "kalman_downtrend",
            "reversal_from_low", "cheapness_and_downward_pressure",
            "cheapness_and_negative_surprise",
            "persistent_kalman_downtrend", "cheapness_and_reversal",
        })
        self.assertTrue(any(
            isinstance(candidate, CompositeRuleVariant)
            for candidates in RULE_LIBRARY.values() for candidate in candidates
        ))

    def test_composite_rule_is_strict_and(self):
        frame = pd.DataFrame({"left": [-1.0, -1.0, 1.0], "right": [1.0, -1.0, 1.0]})
        rule = CompositeRuleVariant("test", (
            RuleVariant("left", "left", "le", 0.0),
            RuleVariant("right", "right", "ge", 0.0),
        ))
        self.assertEqual(rule.predict(frame).tolist(), [True, False, False])

class TemporalTests(unittest.TestCase):
    def test_variant_matrix_reuses_base_replay_within_each_ml_scope(self):
        scopes = ("pooled", "hybrid", "per_currency")
        selectors = ("threshold", "logistic_regression", "extra_trees")
        with (
            patch.object(
                benchmark_module, "prepare_yura_pipeline",
                side_effect=lambda *args, config, **kwargs: config,
            ) as prepare,
            patch.object(
                benchmark_module, "run_prepared_yura_pipeline",
                side_effect=lambda prepared, *args, selector, **kwargs: (
                    prepared, selector
                ),
            ) as run_prepared,
            patch.object(
                benchmark_module, "build_opportunity_selector",
                side_effect=lambda name: name,
            ),
            patch.object(
                benchmark_module, "summarize_yura_variant",
                side_effect=lambda result, ml_scope, selector_type: {
                    "ml_scope": ml_scope, "selector_type": selector_type,
                },
            ),
        ):
            benchmark = benchmark_module.run_yura_variant_matrix(
                pd.DataFrame(), target_registry=pd.DataFrame(),
                base_config=YuraPipelineConfig(), ml_scopes=scopes,
                selector_types=selectors, progress=None,
            )
        self.assertEqual(prepare.call_count, len(scopes))
        self.assertEqual(run_prepared.call_count, len(scopes) * len(selectors))
        self.assertEqual(len(benchmark.results), len(scopes) * len(selectors))
        self.assertTrue(benchmark.failures.empty)

    def test_ml_scope_is_explicit_and_validated(self):
        for scope in ("pooled", "hybrid", "per_currency"):
            self.assertEqual(YuraPipelineConfig(ml_scope=scope).ml_scope, scope)
        with self.assertRaises(ValueError):
            YuraPipelineConfig(ml_scope="currency_thresholds")

    def test_group_balancing_gives_equal_mass_to_each_stratum(self):
        frame = pd.DataFrame({
            "currency": ["AMD", "AMD", "AMD", "KGS"],
            "horizon": [1, 1, 1, 1],
        })
        frame["weight"] = _group_balanced_weights(
            frame, ("currency", "horizon")
        )
        mass = frame.groupby(["currency", "horizon"])["weight"].sum()
        self.assertAlmostEqual(float(mass.iloc[0]), float(mass.iloc[1]))

    def test_ml_baseline_is_local_and_shrunk_to_pooled_train_rate(self):
        train = pd.DataFrame({
            "currency": ["AMD"] * 4 + ["KGS"] * 2,
            "target": [1, 1, 1, 0, 0, 0],
        })
        score = pd.DataFrame({"currency": ["AMD", "KGS", "UNKNOWN"]})
        baseline = _local_training_baseline(
            train, score, target="target", prior_strength=2.0,
        )
        pooled = train["target"].mean()
        self.assertGreater(baseline[0], pooled)
        self.assertLess(baseline[1], pooled)
        self.assertEqual(baseline[2], pooled)

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

    def test_learned_selector_uses_three_ordered_pre_holdout_periods(self):
        config = YuraPipelineConfig(
            arbiter_validation_start="2022-01-01",
            holdout_start="2025-01-01",
        )
        periods = _selector_periods(config)
        self.assertEqual(periods.fit_start, pd.Timestamp("2022-01-01"))
        self.assertEqual(periods.fit_end, pd.Timestamp("2023-01-01"))
        self.assertEqual(periods.calibration_start, pd.Timestamp("2023-01-01"))
        self.assertEqual(periods.validation_start, pd.Timestamp("2024-01-01"))
        self.assertEqual(periods.validation_end, pd.Timestamp("2025-01-01"))

    def test_learned_selector_purges_label_crossing_stage_boundary(self):
        rows = pd.DataFrame({
            "available_at": pd.to_datetime(["2022-12-01", "2022-12-20"]),
            "horizon": [20, 20],
        })
        selected = _mature_slice(
            rows, start=pd.Timestamp("2022-01-01"), end=pd.Timestamp("2023-01-01")
        )
        self.assertEqual(
            selected["available_at"].dt.strftime("%Y-%m-%d").tolist(),
            ["2022-12-01"],
        )

    def test_w1_requires_low_zone_and_future_deterioration(self):
        frame = pd.DataFrame({
            "available_at": pd.date_range("2020-01-01", periods=3, freq="D"),
            "currency": "AMD", "rate": [100.0, 100.0, 100.0],
            "percentile_90d": [0.10, 0.30, 0.10],
            "future_return_2d_bps": [100.0, 100.0, 50.0],
            "centered_min_rate_2d": [100.0, 100.0, 100.0],
        })
        labelled, registry = build_yura_targets(
            frame, horizons=(2,), w1_deterioration_bps=75.0,
            w1_low_percentile=0.15,
        )
        w1 = registry.loc[registry["family"].eq("W1"), "name"].item()
        self.assertEqual(int(labelled.loc[0, w1]), 1)
        self.assertEqual(int(labelled.loc[1, w1]), 0)
        self.assertEqual(int(labelled.loc[2, w1]), 0)
        self.assertEqual(
            w1, "target_w1_lowpct_0p15_deterioration_75bps_h2d"
        )


class PolicyTests(unittest.TestCase):
    def test_aggregated_evidence_keeps_winning_sources_own_baseline(self):
        common = {
            "available_at": pd.Timestamp("2025-01-01"),
            "currency": "KGS", "scenario": "GOOD_NOW",
            "target_family": "G0", "target": "target", "horizon": 1,
            "expected_bps": 10.0, "target_value": True,
            "engine_version": "v1",
        }
        candidates = pd.DataFrame([
            {
                **common, "engine_type": "ml", "engine_name": "ml",
                "confidence": 0.40, "baseline_probability": 0.20,
                "confidence_lift": 2.0,
            },
            {
                **common, "engine_type": "rule", "engine_name": "rule",
                "confidence": 0.36, "baseline_probability": 0.10,
                "confidence_lift": 3.6,
            },
        ])
        opportunity = aggregate_engine_evidence(candidates).opportunities.iloc[0]
        self.assertEqual(opportunity["engine_type"], "rule")
        self.assertAlmostEqual(opportunity["baseline_probability"], 0.10)
        self.assertAlmostEqual(opportunity["confidence_lift"], 3.6)

    def test_selector_factory_keeps_default_and_adds_two_ml_choices(self):
        self.assertIsInstance(build_opportunity_selector(), ThresholdSelector)
        for model_type in ("logistic_regression", "extra_trees"):
            selector = build_opportunity_selector(model_type)
            self.assertIsInstance(selector, LearnedOpportunitySelector)
            self.assertEqual(selector.model_type, model_type)

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
