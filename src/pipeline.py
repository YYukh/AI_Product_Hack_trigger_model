"""One public entry point for the complete alternative pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .signal_backtest import backtest_signal_stream, build_evaluation_universe

from .config import YuraPipelineConfig
from .engine_registry import EngineRegistry, default_engine_registry
from .engines import BaseReplayResult, _target_definitions, replay_base_engines
from .evidence import EvidenceResult, aggregate_engine_evidence
from .policy import apply_signal_policy
from .reporting import build_action_summary
from .selector import OpportunitySelector, ThresholdSelector
from .temporal import TemporalPlan


@dataclass
class YuraPipelineResult:
    config: YuraPipelineConfig
    temporal_plan: TemporalPlan
    base_replay: BaseReplayResult
    evidence: EvidenceResult
    engine_registry: EngineRegistry
    fitted_arbiter: object
    selector_name: str
    arbiter_leaderboard: pd.DataFrame
    final_signals: pd.DataFrame
    backtest_summary: pd.DataFrame
    backtest_rows: pd.DataFrame
    action_summary: pd.DataFrame
    rule_baseline_summary: pd.DataFrame
    ml_score_diagnostics: pd.DataFrame
    action_candidates: pd.DataFrame
    holdout_coverage: pd.DataFrame
    holdout_quarterly_stability: pd.DataFrame

    def final_event_records(self) -> list[dict]:
        """JSON-safe representation of the frozen holdout event stream."""
        records = self.final_signals.copy()
        for column in ("available_at", "as_of"):
            records[column] = pd.to_datetime(records[column]).map(pd.Timestamp.isoformat)
        for column in ("engine_types", "engine_names"):
            records[column] = records[column].map(list)
        return records.to_dict(orient="records")


@dataclass
class PreparedYuraPipeline:
    """Selector-independent base replay for one ML scope."""

    config: YuraPipelineConfig
    temporal_plan: TemporalPlan
    base_replay: BaseReplayResult
    evidence: EvidenceResult
    engine_registry: EngineRegistry
    selected_target_registry: pd.DataFrame
    evaluation_universe: pd.DataFrame


def _rule_baselines(candidates: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    rows = candidates.loc[
        candidates["engine_type"].eq("rule")
        & pd.to_datetime(candidates["available_at"]).ge(start)
    ].copy()
    output = []
    group_columns = ["currency", "target_family", "horizon", "engine_name"]
    for keys, sample in rows.groupby(group_columns, sort=True):
        count = len(sample)
        positives = int(sample["target_value"].sum())
        precision = positives / count if count else np.nan
        baseline = float(sample["baseline_probability"].mean())
        benefit = pd.to_numeric(sample["benefit_bps"], errors="coerce")
        output.append({
            **dict(zip(group_columns, keys)),
            "signal_count": count,
            "true_positive": positives,
            "precision": precision,
            "mean_causal_confidence": float(sample["confidence"].mean()),
            "mean_random_precision": baseline,
            "lift": precision / baseline if baseline else np.nan,
            "mean_benefit_bps": float(benefit.mean()),
            "positive_benefit_rate": float(benefit.gt(0).mean()),
        })
    return pd.DataFrame(output)


def _ml_diagnostics(candidates: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    rows = candidates.loc[
        candidates["engine_type"].eq("ml")
        & pd.to_datetime(candidates["available_at"]).ge(start)
    ].copy()
    output = []
    group_columns = ["currency", "target_family", "horizon", "engine_name"]
    for keys, sample in rows.groupby(group_columns, sort=True):
        target = sample["target_value"].astype(int).to_numpy()
        probability = sample["confidence"].astype(float).to_numpy()
        output.append({
            **dict(zip(group_columns, keys)),
            "observations": len(sample),
            "positive_rate": float(target.mean()),
            "mean_probability": float(probability.mean()),
            "brier_score": float(brier_score_loss(target, probability)),
            "roc_auc": (
                float(roc_auc_score(target, probability))
                if np.unique(target).size == 2 else np.nan
            ),
        })
    return pd.DataFrame(output)


def _coverage_report(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "currency", "scenario", "target_family", "horizon", "signal_count"
        ])
    return (
        events.groupby(
            ["currency", "scenario", "target_family", "horizon"], sort=True
        )
        .size().rename("signal_count").reset_index()
    )


def _quarterly_report(rows: pd.DataFrame) -> pd.DataFrame:
    sample = rows.copy()
    sample["quarter"] = pd.to_datetime(sample["available_at"]).dt.to_period("Q").astype(str)
    output = []
    for (currency, quarter), block in sample.groupby(["currency", "quarter"], sort=True):
        signals = block.loc[block["signal"]]
        count = len(signals)
        precision = float(signals["target_value"].mean()) if count else 0.0
        baseline = float(signals["_stratum_random_precision"].mean()) if count else 0.0
        benefit = pd.to_numeric(signals["benefit_bps"], errors="coerce").dropna()
        random_benefit = pd.to_numeric(
            signals["_stratum_random_benefit_bps"], errors="coerce"
        ).dropna()
        output.append({
            "currency": currency,
            "quarter": quarter,
            "signal_count": count,
            "precision": precision,
            "random_precision": baseline,
            "lift": precision / baseline if baseline else 0.0,
            "benefit_uplift_bps": (
                float(benefit.mean() - random_benefit.mean())
                if len(benefit) and len(random_benefit) else np.nan
            ),
        })
    return pd.DataFrame(output)


def _assert_temporal_integrity(result: BaseReplayResult) -> None:
    audit = result.audit.loc[result.audit.get("fitted", False).astype(bool)].copy()
    required = {"retrain_at", "trained_through", "horizon"}
    eligible = audit.dropna(subset=list(required.intersection(audit.columns)))
    if eligible.empty:
        return
    retrain_at = pd.to_datetime(eligible["retrain_at"])
    trained_through = pd.to_datetime(eligible["trained_through"])
    maturity = trained_through + pd.to_timedelta(eligible["horizon"], unit="D")
    if not maturity.lt(retrain_at).all():
        raise AssertionError("Temporal leakage: engine trained on an immature label")


def _assert_policy_cap(events: pd.DataFrame, maximum: int) -> None:
    for _, sample in events.groupby("currency"):
        dates = pd.to_datetime(sample["available_at"]).sort_values().reset_index(drop=True)
        for when in dates:
            count = int(dates.between(when - pd.Timedelta(days=6), when).sum())
            if count > maximum:
                raise AssertionError("Policy violated max_signals_per_7d")


def prepare_yura_pipeline(
    scoring_data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    config: YuraPipelineConfig = YuraPipelineConfig(),
    engine_registry: EngineRegistry | None = None,
) -> PreparedYuraPipeline:
    """Run the expensive selector-independent walk-forward stage once."""
    temporal_plan = TemporalPlan.from_data(scoring_data, config)
    runtime_config = temporal_plan.resolve(config)
    engines = engine_registry or default_engine_registry()
    selected_registry = _target_definitions(target_registry, runtime_config)
    base_replay = replay_base_engines(
        scoring_data, target_registry=target_registry, config=runtime_config,
        engine_registry=engines,
    )
    _assert_temporal_integrity(base_replay)
    evidence = aggregate_engine_evidence(base_replay.candidates, config=runtime_config)
    action_candidates = evidence.opportunities
    all_universe = build_evaluation_universe(
        scoring_data,
        target_registry=selected_registry,
        target_families=runtime_config.target_families,
        currencies=runtime_config.currencies,
        start_date=runtime_config.base_oos_start,
    )
    return PreparedYuraPipeline(
        config=runtime_config,
        temporal_plan=temporal_plan,
        base_replay=base_replay,
        evidence=evidence,
        engine_registry=engines,
        selected_target_registry=selected_registry,
        evaluation_universe=all_universe,
    )


def run_prepared_yura_pipeline(
    prepared: PreparedYuraPipeline,
    scoring_data: pd.DataFrame,
    *,
    selector: OpportunitySelector | None = None,
) -> YuraPipelineResult:
    """Fit and evaluate one selector on a shared prepared base replay."""
    runtime_config = prepared.config
    temporal_plan = prepared.temporal_plan
    base_replay = prepared.base_replay
    evidence = prepared.evidence
    engines = prepared.engine_registry
    selected_registry = prepared.selected_target_registry
    action_candidates = evidence.opportunities
    all_universe = prepared.evaluation_universe
    selector_impl = selector or ThresholdSelector()
    selector_name = getattr(selector_impl, "selector_name", type(selector_impl).__name__)
    fitted_arbiter, leaderboard = selector_impl.fit(
        action_candidates,
        evaluation_universe=all_universe,
        config=runtime_config,
    )
    holdout_start = temporal_plan.holdout_start
    holdout_candidates = action_candidates.loc[
        pd.to_datetime(action_candidates["available_at"]).ge(holdout_start)
    ].copy()
    selected_holdout = selector_impl.select(holdout_candidates, fitted_arbiter)
    selected_holdout["selector_name"] = selector_name
    selected_holdout["selector_fitted_through"] = getattr(
        fitted_arbiter, "trained_through", holdout_start
    )
    policy_config = selector_impl.policy_config(fitted_arbiter)
    selected_holdout["policy_cooldown_days"] = policy_config.cooldown_days
    selected_holdout["policy_max_signals_per_7d"] = policy_config.max_signals_per_7d
    final_signals = apply_signal_policy(
        selected_holdout, policy_config
    )
    _assert_policy_cap(final_signals, runtime_config.max_signals_per_7d)

    holdout_universe = build_evaluation_universe(
        scoring_data,
        target_registry=selected_registry,
        target_families=runtime_config.target_families,
        currencies=runtime_config.currencies,
        start_date=holdout_start,
    )
    summary, rows = backtest_signal_stream(
        final_signals, evaluation_universe=holdout_universe
    )
    expected_rows = (
        len(runtime_config.currencies)
        + len(runtime_config.currencies) * len(runtime_config.horizons)
        + len(runtime_config.currencies) * len(runtime_config.horizons)
        * len(runtime_config.target_families)
    )
    if len(summary) != expected_rows:
        raise AssertionError(f"Ожидалось {expected_rows} summary rows, получено {len(summary)}")
    return YuraPipelineResult(
        config=runtime_config,
        temporal_plan=temporal_plan,
        base_replay=base_replay,
        evidence=evidence,
        engine_registry=engines,
        fitted_arbiter=fitted_arbiter,
        selector_name=selector_name,
        arbiter_leaderboard=leaderboard,
        final_signals=final_signals,
        backtest_summary=summary,
        backtest_rows=rows,
        action_summary=build_action_summary(rows),
        rule_baseline_summary=_rule_baselines(base_replay.candidates, holdout_start),
        ml_score_diagnostics=_ml_diagnostics(base_replay.candidates, holdout_start),
        action_candidates=action_candidates,
        holdout_coverage=_coverage_report(final_signals),
        holdout_quarterly_stability=_quarterly_report(rows),
    )


def run_yura_pipeline(
    scoring_data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    config: YuraPipelineConfig = YuraPipelineConfig(),
    engine_registry: EngineRegistry | None = None,
    selector: OpportunitySelector | None = None,
) -> YuraPipelineResult:
    """Run base WF, validation-only selection and one untouched holdout."""
    prepared = prepare_yura_pipeline(
        scoring_data,
        target_registry=target_registry,
        config=config,
        engine_registry=engine_registry,
    )
    return run_prepared_yura_pipeline(
        prepared,
        scoring_data,
        selector=selector,
    )
