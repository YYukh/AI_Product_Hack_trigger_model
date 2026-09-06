import unittest
import json

import numpy as np
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
    update_models_if_due,
)
from src.signal_backtest import backtest_signal_stream
from src.signal_contract import (
    calibrate_rule_confidence_from_oos,
    combine_evidence_streams,
    standardize_ml_signals,
    standardize_rule_signals,
)
from src.signal_policy import SignalPolicyConfig, apply_signal_policy


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
        self.assertEqual(events.loc[0, "confidence"], 0.91)
        self.assertEqual(events.loc[0, "as_of"].hour, 9)
        self.assertEqual(str(events.loc[0, "as_of"].tz), "Europe/Moscow")

    def test_policy_enforces_cooldown_and_rolling_seven_day_cap(self):
        events = pd.DataFrame([
            {
                "available_at": pd.Timestamp(f"2025-01-{day:02d}"),
                "currency": "TJS", "confidence": confidence,
                "evidence_count": 1, "horizon": 1,
                "target": f"target_{day}",
            }
            for day, confidence in ((1, 0.7), (2, 0.9), (4, 0.8), (7, 0.95), (8, 0.6))
        ])
        selected = apply_signal_policy(
            events,
            SignalPolicyConfig(cooldown_days=3, max_signals_per_7d=2),
        )
        self.assertEqual(
            selected["available_at"].dt.day.tolist(), [1, 4, 8]
        )
        daily = events.loc[events["available_at"].eq(pd.Timestamp("2025-01-07"))]
        blocked = apply_signal_policy(
            daily,
            SignalPolicyConfig(cooldown_days=3, max_signals_per_7d=2),
            history=selected.loc[selected["available_at"].lt("2025-01-07")],
        )
        self.assertTrue(blocked.empty)

    def test_rule_confidence_uses_only_past_mature_oos_and_freezes(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-04", "2024-01-05", "2024-01-06",
        ])
        signals = pd.DataFrame({
            "available_at": dates,
            "currency": "TJS",
            "scenario": "GOOD_NOW",
            "target_family": "G0",
            "target": "target_test",
            "horizon": 1,
            "engine_id": "rule:TJS:G0:h1",
            "engine_type": "rule",
            "signal": [True, False, True, False, False, False],
            "confidence": 0.99,
            "confidence_method": "train_precision_of_refitted_rule",
            "confidence_support": 100,
            "baseline_probability": 0.5,
            "confidence_lift": 1.98,
        })
        universe = signals.loc[:, [
            "available_at", "currency", "scenario", "target_family",
            "target", "horizon",
        ]].copy()
        universe["target_value"] = [True, False, False, True, True, True]

        calibrated = calibrate_rule_confidence_from_oos(
            signals,
            evaluation_universe=universe,
            freeze_at="2024-01-05",
        )

        # The 01-Jan signal matures at 02-Jan and is usable only strictly
        # after that date.  The 03-Jan false positive becomes usable on
        # 05-Jan, exactly where confidence is frozen for later rows.
        self.assertEqual(calibrated.loc[1, "confidence_support"], 0)
        self.assertEqual(calibrated.loc[2, "confidence_support"], 1)
        self.assertEqual(calibrated.loc[2, "confidence"], 1.0)
        self.assertEqual(calibrated.loc[4, "confidence_support"], 2)
        self.assertEqual(calibrated.loc[4, "confidence"], 0.5)
        self.assertEqual(calibrated.loc[5, "confidence_support"], 2)
        self.assertEqual(calibrated.loc[5, "confidence"], 0.5)

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

    def test_aggregate_lift_matches_realised_target_mix(self):
        base = {
            "currency": "TJS", "scenario": "GOOD_NOW", "horizon": 1,
        }
        universe = pd.DataFrame([
            {**base, "available_at": pd.Timestamp("2025-01-01"),
             "target_family": "G0", "target": "g0", "target_value": True},
            {**base, "available_at": pd.Timestamp("2025-01-02"),
             "target_family": "G0", "target": "g0", "target_value": False},
            {**base, "available_at": pd.Timestamp("2025-01-01"),
             "target_family": "W1", "target": "w1", "target_value": True},
            {**base, "available_at": pd.Timestamp("2025-01-02"),
             "target_family": "W1", "target": "w1", "target_value": False},
            {**base, "available_at": pd.Timestamp("2025-01-03"),
             "target_family": "W1", "target": "w1", "target_value": False},
            {**base, "available_at": pd.Timestamp("2025-01-04"),
             "target_family": "W1", "target": "w1", "target_value": False},
        ])
        universe["benefit_bps"] = [10.0, 30.0, 0.0, 10.0, 20.0, 30.0]
        events = pd.DataFrame([
            {**base, "available_at": pd.Timestamp("2025-01-01"),
             "target_family": "G0", "target": "g0", "event_id": "a",
             "confidence": 0.8, "evidence_count": 1},
            {**base, "available_at": pd.Timestamp("2025-01-02"),
             "target_family": "G0", "target": "g0", "event_id": "b",
             "confidence": 0.8, "evidence_count": 1},
            {**base, "available_at": pd.Timestamp("2025-01-02"),
             "target_family": "W1", "target": "w1", "event_id": "c",
             "confidence": 0.8, "evidence_count": 1},
        ])

        report, _ = backtest_signal_stream(events, evaluation_universe=universe)
        currency = report.loc[report["scope"].eq("currency")].iloc[0]
        # Same emitted mix: 2 × P(G0)=0.5 and 1 × P(W1)=0.25.
        self.assertAlmostEqual(currency["random_precision"], 1.25 / 3)
        self.assertAlmostEqual(currency["lift"], (1 / 3) / (1.25 / 3))
        self.assertAlmostEqual(currency["mean_benefit_bps"], 50.0 / 3)
        self.assertAlmostEqual(currency["random_mean_benefit_bps"], 55.0 / 3)
        self.assertAlmostEqual(currency["benefit_uplift_bps"], -5.0 / 3)

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

    def test_pooled_ml_shares_artifact_but_emits_per_currency_signals(self):
        dates = pd.date_range("2021-01-01", "2024-01-01", freq="3D")
        frames = []
        for offset, currency in enumerate(("AMD", "KZT")):
            position = np.arange(len(dates), dtype=float)
            feature = np.sin(position / 11 + offset * 0.3)
            frames.append(pd.DataFrame({
                "available_at": dates,
                "currency": currency,
                "feature_x": feature,
                "target_test": (feature > 0).astype("int8"),
            }))
        data = pd.concat(frames, ignore_index=True)
        registry = pd.DataFrame([{
            "name": "target_test", "scenario": "GOOD_NOW",
            "family": "G0", "horizon": 1,
        }])
        states = initialize_engine_states(
            rule_configurations=(), target_registry=registry,
            currencies=("AMD", "KZT"), target_families=("G0",),
            first_score_date="2024-01-01", train_months=24,
            ml_feature_names=("feature_x",),
            ml_model_type="logistic_regression", ml_retrain_months=12,
            ml_pooling_mode="pooled_currencies",
        )
        audit = update_models_if_due(
            as_of="2024-01-01", data=data, states=states,
            indicator_spaces={}, rule_min_signals_per_week=0.1,
            ml_validation_months=6, ml_min_signals_per_week=0.1,
            ml_model_type="logistic_regression",
            ml_model_config=MLIndicatorConfig(),
            ml_pooling_mode="pooled_currencies",
        )

        self.assertTrue(all(row["fitted"] for row in audit))
        self.assertIs(
            states["ml:AMD:G0:h1"].payload,
            states["ml:KZT:G0:h1"].payload,
        )
        signals = get_signal(
            as_of="2024-01-01", feature_snapshot=data, states=states
        )
        self.assertEqual(len(signals), 2)
        self.assertEqual({row["currency"] for row in signals}, {"AMD", "KZT"})
        self.assertTrue(all(row["confidence_method"] == "model_predict_proba" for row in signals))


if __name__ == "__main__":
    unittest.main()
