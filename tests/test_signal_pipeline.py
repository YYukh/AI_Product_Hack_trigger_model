import unittest
import json

import pandas as pd

from src.meta_model import (
    ConfidenceFilterConfig,
    build_meta_candidates,
    confidence_filter_meta_model,
    fit_logistic_meta_model,
    logistic_meta_model,
    market_event_records,
)
from src.production_config import FIXED_INDICATORS, PRODUCTION_HORIZONS
from src.production_config import FixedIndicatorConfig
from src.indicators import IndicatorCandidate, IndicatorRule
from src.ml_backtest import MLIndicatorConfig
from src.production_pipeline import (
    get_signal,
    initialize_engine_states,
    run_signal_day,
)
from src.signal_backtest import backtest_signal_stream
from src.signal_contract import (
    combine_evidence_streams,
    standardize_ml_signals,
    standardize_rule_signals,
)


class SignalPipelineTest(unittest.TestCase):
    def setUp(self):
        self.rule_signals = pd.DataFrame([{
            "available_at": "2025-01-02",
            "currency": "TJS",
            "target_value": 1,
            "scenario": "GOOD_NOW",
            "target_family": "G0",
            "target": "target_g0_h1",
            "horizon": 1,
            "indicator": "low_level",
            "fixed_candidate": "percentile_90d_le_0.1",
            "fixed_logic": "SINGLE",
            "rebalance_months": 6,
            "fold_id": 1,
        }])
        self.selected = pd.DataFrame([{
            "currency": "TJS",
            "target": "target_g0_h1",
            "horizon": 1,
            "oos_precision": 0.8,
            "test_predicted_positive_count": 40,
        }])
        self.ml_signals = pd.DataFrame([{
            "available_at": "2025-01-02",
            "currency": "TJS",
            "target_value": 1,
            "scenario": "GOOD_NOW",
            "target_family": "G0",
            "target": "target_g0_h1",
            "horizon": 1,
            "indicator": "ml_hist_gradient_boosting",
            "rebalance_months": 12,
            "fold_id": 1,
            "probability": 0.91,
            "probability_threshold": 0.7,
            "confidence": 0.75,
            "confidence_method": "validation_precision_at_probability_threshold",
            "confidence_support": 30,
        }])

    def test_streams_share_contract_and_meta_model_deduplicates(self):
        rule = standardize_rule_signals(
            self.rule_signals, selected_indicators=self.selected
        )
        ml = standardize_ml_signals(self.ml_signals)
        evidence = combine_evidence_streams(rule, ml)
        events = confidence_filter_meta_model(
            evidence,
            ConfidenceFilterConfig(min_confidence=0.7, min_support=10),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc[0, "evidence_count"], 2)
        self.assertEqual(events.loc[0, "confidence"], 0.8)
        self.assertEqual(events.loc[0, "as_of"].hour, 9)
        self.assertEqual(str(events.loc[0, "as_of"].tz), "Europe/Moscow")

    def test_production_rule_registry_is_complete_and_unique(self):
        keys = [
            (item.currency, item.target_family, item.horizon)
            for item in FIXED_INDICATORS
        ]
        self.assertEqual(len(FIXED_INDICATORS), 50)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {item.horizon for item in FIXED_INDICATORS},
            set(PRODUCTION_HORIZONS),
        )

    def test_backtest_uses_pooled_counts(self):
        rule = standardize_rule_signals(
            self.rule_signals, selected_indicators=self.selected
        )
        events = confidence_filter_meta_model(
            rule,
            ConfidenceFilterConfig(min_confidence=0.7, min_support=10),
        )
        universe = pd.DataFrame([
            {
                "available_at": pd.Timestamp("2025-01-02"),
                "currency": "TJS",
                "scenario": "GOOD_NOW",
                "target_family": "G0",
                "target": "target_g0_h1",
                "horizon": 1,
                "target_value": True,
            },
            {
                "available_at": pd.Timestamp("2025-01-03"),
                "currency": "TJS",
                "scenario": "GOOD_NOW",
                "target_family": "G0",
                "target": "target_g0_h1",
                "horizon": 1,
                "target_value": False,
            },
        ])
        report, scored = backtest_signal_stream(
            events, evaluation_universe=universe
        )
        currency_horizon = report.loc[
            report["scope"].eq("currency+horizon+target_family")
        ].iloc[0]
        self.assertEqual(currency_horizon["signal_count"], 1)
        self.assertEqual(currency_horizon["true_positive"], 1)
        self.assertEqual(currency_horizon["precision"], 1.0)
        self.assertEqual(currency_horizon["lift"], 2.0)
        self.assertEqual(len(report), 3)
        self.assertEqual(scored["signal"].sum(), 1)

    def test_backtest_accepts_empty_meta_model_output(self):
        empty_events = pd.DataFrame(columns=[
            "available_at", "currency", "scenario", "target_family",
            "target", "horizon", "event_id", "confidence",
            "evidence_count",
        ])
        universe = pd.DataFrame([{
            "available_at": pd.Timestamp("2025-01-02"),
            "currency": "TJS",
            "scenario": "GOOD_NOW",
            "target_family": "G0",
            "target": "target_g0_h1",
            "horizon": 1,
            "target_value": True,
        }])
        report, scored = backtest_signal_stream(
            empty_events, evaluation_universe=universe
        )
        self.assertEqual(len(report), 3)
        self.assertTrue(
            (report["scope"] == "currency+horizon+target_family").any()
        )
        detailed = report.loc[
            report["scope"].eq("currency+horizon+target_family")
        ].iloc[0]
        self.assertEqual(detailed["signal_count"], 0)
        self.assertEqual(detailed["precision"], 0.0)
        self.assertFalse(scored["signal"].any())

    def test_logistic_meta_model_is_fit_on_engine_evidence(self):
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        rows = []
        for index, available_at in enumerate(dates):
            target = bool(index % 2)
            for engine_type, probability in (
                ("rule", float(target)),
                ("ml", 0.9 if target else 0.1),
            ):
                rows.append({
                    "schema_version": "1.0",
                    "signal_id": f"{index}-{engine_type}",
                    "available_at": available_at,
                    "as_of": available_at + pd.Timedelta(hours=9),
                    "currency": "TJS",
                    "corridor": "RUB_TJS",
                    "scenario": "GOOD_NOW",
                    "target_family": "G0",
                    "target": "target_test",
                    "horizon": 1,
                    "engine_type": engine_type,
                    "engine_name": engine_type,
                    "engine_version": "v1",
                    "signal": True,
                    "raw_score": probability,
                    "decision_threshold": 0.5,
                    "confidence": 0.8,
                    "confidence_support": 100,
                    "confidence_lift": 2.0,
                    "expires_at": available_at + pd.Timedelta(hours=33),
                })
        evidence = pd.DataFrame(rows)
        candidates = build_meta_candidates(evidence)
        candidates["target_value"] = [bool(i % 2) for i in range(len(candidates))]
        fitted = fit_logistic_meta_model(
            candidates,
            train_end="2024-01-01",
            validation_months=12,
            min_signals_per_week=0.1,
        )
        events = logistic_meta_model(evidence, fitted)
        self.assertGreater(len(events), 0)
        self.assertTrue(market_event_records(events))

    def test_stateful_rule_requires_due_update_and_returns_json(self):
        dates = pd.date_range("2024-01-01", "2025-02-01", freq="D")
        feature = pd.Series(range(len(dates)), dtype=float).mod(2).to_numpy()
        data = pd.DataFrame({
            "available_at": dates,
            "currency": "TJS",
            "feature_x": feature,
            "target_test": feature == 0,
        })
        rule = IndicatorRule("test", "feature_x", "le", 0.5)
        candidate = IndicatorCandidate((rule,))
        configuration = FixedIndicatorConfig(
            currency="TJS",
            scenario="GOOD_NOW",
            target_family="G0",
            target="target_test",
            horizon=1,
            indicator="test_architecture",
            retrain_months=1,
        )
        empty_registry = pd.DataFrame(
            columns=["family", "horizon", "scenario", "name"]
        )
        states = initialize_engine_states(
            rule_configurations=(configuration,),
            target_registry=empty_registry,
            currencies=(),
            target_families=(),
            first_score_date="2025-01-01",
            train_months=24,
            ml_feature_names=(),
            ml_model_type="hist_gradient_boosting",
            ml_retrain_months=12,
        )
        daily = run_signal_day(
            as_of="2025-01-01",
            data=data,
            states=states,
            indicator_spaces={"test_architecture": [candidate]},
            rule_min_signals_per_week=2.0,
            ml_validation_months=12,
            ml_min_signals_per_week=2.0,
            ml_model_type="hist_gradient_boosting",
            ml_model_config=MLIndicatorConfig(),
            meta_config=ConfidenceFilterConfig(
                min_confidence=0.0, min_support=0
            ),
        )
        self.assertTrue(daily.training_audit[0]["fitted"])
        self.assertEqual(states["rule:TJS:G0:h1"].next_retrain_at, pd.Timestamp("2025-02-01"))

        signals = daily.raw_signals
        self.assertEqual(len(signals), 1)
        json.dumps(signals)

        with self.assertRaises(RuntimeError):
            get_signal(
                as_of="2025-02-01", feature_snapshot=data, states=states
            )


if __name__ == "__main__":
    unittest.main()
