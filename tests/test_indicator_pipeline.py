import unittest

import numpy as np
import pandas as pd

from src.features import build_features
from src.indicator_backtest import backtest_fixed_indicators
from src.indicators import IndicatorCandidate, IndicatorRule
from src.market_data import build_daily_market_panel
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

    def test_fixed_backtest_reports_actual_signal_frequency_and_lift(self):
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
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

        summary, folds, signals = backtest_fixed_indicators(
            data,
            selected_indicators=selected,
            fitted_indicators={("TJS", "target_test", 1): candidate},
            backtest_start="2025-01-01",
            target_families=("G0",),
        )

        result = summary.iloc[0]
        self.assertEqual(result["test_signal_count"], 20)
        self.assertEqual(result["test_signal_count"], len(signals))
        self.assertEqual(len(folds), 3)
        self.assertAlmostEqual(result["test_lift"], 3.0)
        self.assertAlmostEqual(result["test_signals_per_week"], 20 / (60 / 7))


if __name__ == "__main__":
    unittest.main()
