import unittest

import numpy as np
import pandas as pd

from src.features import build_features
from src.indicator_backtest import backtest_selected_indicators
from src.indicators import IndicatorCandidate, IndicatorRule
from src.market_data import build_daily_market_panel
from src.ml_backtest import run_ml_indicator_backtest
from src.outcomes import add_future_outcomes
from src.targets import build_targets


class IndicatorPipelineTest(unittest.TestCase):
    def test_features_outcomes_and_targets_form_one_pipeline(self):
        dates = pd.date_range("2020-01-01", periods=240, freq="2D", name="available_at")
        x = np.arange(len(dates), dtype="float64")
        rates = pd.DataFrame(
            {
                "TJS": 7.0 + 0.10 * np.sin(x / 9) + x / 10_000,
                "KZT": 0.2 + 0.01 * np.cos(x / 11) + x / 100_000,
            },
            index=dates,
        )

        panel = build_daily_market_panel(rates)
        features = build_features(panel)
        outcomes = add_future_outcomes(features, horizons=(1, 3))
        dataset, registry = build_targets(outcomes, horizons=(1, 3))

        self.assertTrue({"G0", "G1", "W0", "W1"}.issubset(set(registry["family"])))
        self.assertTrue(set(registry["name"]).issubset(dataset.columns))
        self.assertFalse(dataset["source_available_at"].gt(dataset["available_at"]).any())
        self.assertTrue(
            {
                "percentile_60d",
                "acceleration_1d_bps",
                "previous_distance_from_low_20d_bps",
                "bounce_from_prior_low_20d_bps",
                "recipient_preholiday_7",
            }.issubset(dataset.columns)
        )

    def test_fixed_backtest_reports_actual_signal_frequency_and_lift(self):
        dates = pd.date_range("2023-01-01", "2025-03-01", freq="D")
        signal_feature = (np.arange(len(dates)) % 3 == 0).astype(float)
        data = pd.DataFrame(
            {
                "available_at": dates,
                "currency": "TJS",
                "signal_feature": signal_feature,
                "target_test": signal_feature.astype("int8"),
            }
        )
        candidate = IndicatorCandidate(
            (IndicatorRule("fixed", "signal_feature", "ge", 1.0),)
        )
        selected = pd.DataFrame(
            [
                {
                    "currency": "TJS",
                    "scenario": "GOOD_NOW",
                    "target_family": "G0",
                    "target": "target_test",
                    "horizon": 1,
                    "indicator": "fixed",
                    "test_months": 1,
                    "fitted_candidate": candidate.name,
                    "fitted_logic": candidate.logic,
                }
            ]
        )

        summary, folds, signals = backtest_selected_indicators(
            data,
            selected_indicators=selected,
            indicator_spaces={"fixed": [candidate]},
            backtest_start="2025-01-01",
            train_months=24,
            min_signals_per_week=2.0,
            target_families=("G0",),
        )

        result = summary.iloc[0]
        expected_signals = int(data.loc[
            data["available_at"].ge("2025-01-01"), "signal_feature"
        ].sum())
        self.assertEqual(result["test_signal_count"], expected_signals)
        self.assertEqual(result["test_signal_count"], len(signals))
        self.assertEqual(len(folds), 3)
        self.assertAlmostEqual(result["test_lift"], 3.0)
        self.assertTrue(folds["train_start"].ge("2023-01-01").all())
        self.assertEqual(result["distinct_candidates"], 1)

    def test_ml_backtest_selects_cadence_without_using_rule_signals(self):
        dates = pd.date_range("2020-01-01", "2025-12-31", freq="2D")
        position = np.arange(len(dates), dtype="float64")
        feature = np.sin(position / 7)
        target = (feature > 0).astype("int8")
        data = pd.DataFrame(
            {
                "available_at": dates,
                "currency": "TJS",
                "continuous_feature": feature,
                "target_test": target,
            }
        )
        registry = pd.DataFrame(
            [
                {
                    "name": "target_test",
                    "scenario": "GOOD_NOW",
                    "family": "G0",
                    "horizon": 1,
                }
            ]
        )

        cadence, summary, folds, signals = run_ml_indicator_backtest(
            data,
            target_registry=registry,
            feature_names=("continuous_feature",),
            first_test_date="2022-01-01",
            winner_backtest_start="2025-01-01",
            test_months_options=(6, 12),
            train_months=24,
            target_families=("G0",),
            min_signals_per_week=0.5,
            validation_months=6,
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(cadence["selected"].sum(), 1)
        self.assertEqual(summary.loc[0, "indicator"], "ml_logistic_regression")
        self.assertEqual(summary.loc[0, "test_signal_count"], len(signals))
        self.assertGreater(len(folds), 0)
        self.assertTrue({
            "confidence", "confidence_method", "confidence_support"
        }.issubset(signals.columns))


if __name__ == "__main__":
    unittest.main()
