"""Validation-selected, source-aware arbiter and deterministic hard policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib

import numpy as np
import pandas as pd

from src.signal_backtest import backtest_signal_stream

from .config import YuraPipelineConfig
from .engines import KEYS
from .policy import SignalPolicyConfig, apply_signal_policy


@dataclass(frozen=True)
class ArbiterConfig:
    rule_min_lift: float
    ml_min_lift: float
    min_expected_bps: float
    min_decision_score: float
    cooldown_days: int
    max_signals_per_7d: int


@dataclass
class FittedArbiter:
    config: ArbiterConfig
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_geometric_lift: float
    validation_macro_lift: float
    validation_min_currency_lift: float
    validation_macro_benefit_uplift_bps: float
    validation_mean_signals_per_week: float
    validation_quarterly_geometric_lift: float
    validation_min_currency_quarterly_geometric_lift: float
    validation_quarterly_lift_p10: float
    validation_stability_ok: bool
    validation_scenario_coverage: float
    validation_horizon_coverage: float


def _eligible_sources(candidates: pd.DataFrame, config: ArbiterConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    confidence_lift = pd.to_numeric(candidates["confidence_lift"], errors="coerce")
    expected_bps = pd.to_numeric(candidates["expected_bps"], errors="coerce")
    source_threshold = np.where(
        candidates["engine_type"].eq("rule"),
        config.rule_min_lift,
        config.ml_min_lift,
    )
    decision_score = pd.to_numeric(
        candidates["decision_score"], errors="coerce"
    ) if "decision_score" in candidates else pd.Series(1.0, index=candidates.index)
    return candidates.loc[
        confidence_lift.ge(source_threshold)
        & expected_bps.ge(config.min_expected_bps)
        & decision_score.ge(config.min_decision_score)
    ].copy()


def _remove_horizon_dominance(candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop a slow opportunity only when a faster one Pareto-dominates it.

    This is weight-free: a shorter horizon must be at least as strong on both
    target evidence and volatility-scaled economic evidence, with one strict
    improvement. Long signals therefore survive when waiting adds information
    or economic value rather than because raw long-window BPS is larger.
    """
    required = {"statistical_evidence", "economic_evidence"}
    if candidates.empty or not required.issubset(candidates.columns):
        return candidates.copy()
    keep = pd.Series(True, index=candidates.index)
    groups = ["available_at", "currency", "scenario", "target_family"]
    for _, group in candidates.groupby(groups, sort=False):
        ordered = group.sort_values("horizon")
        horizons = pd.to_numeric(ordered["horizon"], errors="raise").to_numpy()
        statistical = pd.to_numeric(
            ordered["statistical_evidence"], errors="coerce"
        ).fillna(-np.inf).to_numpy()
        economic = pd.to_numeric(
            ordered["economic_evidence"], errors="coerce"
        ).fillna(-np.inf).to_numpy()
        indices = ordered.index.to_numpy()
        for position in range(1, len(ordered)):
            faster = horizons[:position] < horizons[position]
            no_worse = (
                (statistical[:position] >= statistical[position])
                & (economic[:position] >= economic[position])
            )
            strictly_better = (
                (statistical[:position] > statistical[position])
                | (economic[:position] > economic[position])
            )
            if np.any(faster & no_worse & strictly_better):
                keep.loc[indices[position]] = False
    return candidates.loc[keep].copy()


def _collapse_evidence(candidates: pd.DataFrame) -> pd.DataFrame:
    """One event candidate per exact target configuration and date."""
    if candidates.empty:
        return pd.DataFrame(columns=[
            *KEYS, "event_id", "confidence", "confidence_lift", "expected_bps",
            "evidence_count", "engine_types", "engine_names",
            "winning_engine_type", "winning_engine_name", "rule_count",
            "winning_engine_version", "aggregation_version",
            "source_profile", "decision_score",
        ])
    ordered = candidates.copy()
    if "decision_score" in ordered:
        ordered["_decision_score"] = pd.to_numeric(
            ordered["decision_score"], errors="coerce"
        ).fillna(0.0)
    else:
        # Backward-compatible path for direct calls and unit tests. Production
        # Production replay supplies the causal cross-horizon evidence score.
        confidence = pd.to_numeric(
            ordered["confidence"], errors="coerce"
        ).fillna(0.0).clip(1e-6, 1.0 - 1e-6)
        baseline = pd.to_numeric(
            ordered["baseline_probability"], errors="coerce"
        ).fillna(0.0).clip(1e-6, 1.0 - 1e-6)
        log_odds_gain = (
            np.log(confidence / (1.0 - confidence))
            - np.log(baseline / (1.0 - baseline))
        ).clip(lower=0.0)
        horizon_scale = np.sqrt(
            pd.to_numeric(ordered["horizon"], errors="raise")
        )
        ordered["_decision_score"] = (
            log_odds_gain
            * pd.to_numeric(ordered["expected_bps"], errors="coerce")
            .clip(lower=0.0)
            .div(horizon_scale)
        )
    ordered = ordered.sort_values(
        [*KEYS, "_decision_score", "confidence", "expected_bps",
         "engine_type", "engine_name"],
        ascending=[True] * len(KEYS) + [False, False, False, True, True],
    )
    rows: list[dict] = []
    for key, group in ordered.groupby(list(KEYS), sort=False):
        winner = group.iloc[0]
        identity = "|".join(map(str, key))
        event_date = pd.Timestamp(key[0]).normalize()
        if event_date.tzinfo is None:
            event_date = event_date.tz_localize("Europe/Moscow")
        else:
            event_date = event_date.tz_convert("Europe/Moscow")
        rows.append({
            **dict(zip(KEYS, key)),
            "event_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
            "confidence": float(winner["confidence"]),
            "confidence_lift": float(winner["confidence_lift"]),
            "expected_bps": float(winner["expected_bps"]),
            "decision_score": float(winner["_decision_score"]),
            "evidence_count": int(winner.get("evidence_count", len(group))),
            "engine_types": tuple(sorted(group["engine_type"].unique())),
            "engine_names": tuple(sorted(group["engine_name"].unique())),
            "rule_count": int(winner.get("rule_count", 0)),
            "source_profile": (
                "rule+ml" if int(winner.get("rule_count", 0)) > 0 else "ml_only"
            ),
            "winning_engine_type": (
                "rule+ml" if int(winner.get("rule_count", 0)) > 0 else "ml"
            ),
            "winning_engine_name": str(winner["engine_name"]),
            "winning_engine_version": str(winner.get("engine_version", "unknown")),
            "aggregation_version": str(
                winner.get("aggregation_version", "unknown")
            ),
            "as_of": event_date + pd.Timedelta(hours=9),
            "corridor": f"RUB_{key[1]}",
            "status": "READY",
        })
    return pd.DataFrame(rows)


def _apply_policy(events: pd.DataFrame, config: ArbiterConfig) -> pd.DataFrame:
    return apply_signal_policy(
        events,
        SignalPolicyConfig(
            cooldown_days=config.cooldown_days,
            max_signals_per_7d=config.max_signals_per_7d,
        ),
    )


def run_arbiter(candidates: pd.DataFrame, fitted: FittedArbiter | ArbiterConfig) -> pd.DataFrame:
    config = fitted.config if isinstance(fitted, FittedArbiter) else fitted
    events = select_opportunities(candidates, config)
    return _apply_policy(events, config)


def select_opportunities(
    candidates: pd.DataFrame,
    config: ArbiterConfig,
) -> pd.DataFrame:
    """Quality selection only; product frequency policy is deliberately absent."""
    eligible = _eligible_sources(candidates, config)
    eligible = _remove_horizon_dominance(eligible)
    return _collapse_evidence(eligible)


def _validation_universe(
    universe: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(universe["available_at"])
    maturity = dates + pd.to_timedelta(universe["horizon"], unit="D")
    return universe.loc[
        dates.ge(start) & dates.lt(end) & maturity.lt(end)
    ].copy()


def _validation_metrics(summary: pd.DataFrame) -> dict:
    currency = summary.loc[summary["scope"].eq("currency")].copy()
    lifts = pd.to_numeric(currency["lift"], errors="coerce").fillna(0.0).to_numpy()
    positive_lifts = np.maximum(lifts, 1e-9)
    bps = pd.to_numeric(currency["benefit_uplift_bps"], errors="coerce")
    return {
        "geometric_lift": float(np.exp(np.log(positive_lifts).mean())),
        "macro_lift": float(np.mean(lifts)),
        "min_currency_lift": float(np.min(lifts)),
        "macro_benefit_uplift_bps": float(bps.mean()) if bps.notna().any() else np.nan,
        "mean_signals_per_week": float(currency["signals_per_week"].mean()),
        "min_signals_per_week": float(currency["signals_per_week"].min()),
        "max_signals_per_week": float(currency["signals_per_week"].max()),
    }


def _quarterly_stability(rows: pd.DataFrame) -> dict:
    """Jeffreys-smoothed lift stability over currency × calendar quarter."""
    sample = rows.copy()
    sample["quarter"] = pd.to_datetime(sample["available_at"]).dt.to_period("Q")
    block_lifts: list[float] = []
    currency_geometric: list[float] = []
    for _, currency_rows in sample.groupby("currency", sort=True):
        current: list[float] = []
        for _, block in currency_rows.groupby("quarter", sort=True):
            signals = block.loc[block["signal"]]
            count = len(signals)
            if count:
                true_positive = int(signals["target_value"].sum())
                baseline = float(signals["_stratum_random_precision"].mean())
                smoothed_precision = (true_positive + 0.5) / (count + 1.0)
                lift = smoothed_precision / baseline if baseline > 0 else 0.0
            else:
                # No opportunity delivered for a complete quarter is unstable.
                lift = 1e-9
            current.append(max(lift, 1e-9))
            block_lifts.append(max(lift, 1e-9))
        currency_geometric.append(float(np.exp(np.mean(np.log(current)))))
    return {
        "quarterly_geometric_lift": float(np.exp(np.mean(np.log(block_lifts)))),
        "min_currency_quarterly_geometric_lift": float(min(currency_geometric)),
        "quarterly_lift_p10": float(np.quantile(block_lifts, 0.10)),
        "quarterly_lift_std": float(np.std(block_lifts)),
    }


def _coverage_metrics(
    events: pd.DataFrame,
    currencies: tuple[str, ...],
    horizons: tuple[int, ...],
) -> dict:
    """Structural diagnostics only; no target/horizon quota is imposed."""
    if events.empty:
        return {
            "scenario_coverage": 0.0,
            "horizon_coverage": 0.0,
            "coverage_ok": False,
        }
    expected_scenarios = {"GOOD_NOW", "WINDOW_CLOSING"}
    expected_horizons = set(horizons)
    scenario_ratios = []
    horizon_ratios = []
    for currency in currencies:
        sample = events.loc[events["currency"].eq(currency)]
        scenario_ratios.append(
            len(set(sample["scenario"]) & expected_scenarios) / len(expected_scenarios)
        )
        horizon_ratios.append(
            len(set(pd.to_numeric(sample["horizon"], errors="coerce").dropna().astype(int))
                & expected_horizons) / len(expected_horizons)
        )
    scenario_coverage = float(np.mean(scenario_ratios))
    horizon_coverage = float(np.mean(horizon_ratios))
    return {
        "scenario_coverage": scenario_coverage,
        "horizon_coverage": horizon_coverage,
        "coverage_ok": bool(min(scenario_ratios) == 1.0),
    }


def fit_arbiter(
    candidates: pd.DataFrame,
    *,
    evaluation_universe: pd.DataFrame,
    config: YuraPipelineConfig = YuraPipelineConfig(),
) -> tuple[FittedArbiter, pd.DataFrame]:
    """Validate/select a small transparent threshold policy before holdout."""
    start = pd.Timestamp(config.arbiter_validation_start)
    end = pd.Timestamp(config.holdout_start)
    dates = pd.to_datetime(candidates["available_at"])
    validation_candidates = candidates.loc[dates.ge(start) & dates.lt(end)].copy()
    universe = _validation_universe(evaluation_universe, start=start, end=end)
    if validation_candidates.empty or universe.empty:
        raise ValueError("Недостаточно данных для arbiter validation")

    rows: list[dict] = []
    fitted_configs: list[ArbiterConfig] = []
    for rule_lift, ml_lift, min_bps, decision_threshold, cooldown in product(
        config.rule_lift_thresholds,
        config.ml_lift_thresholds,
        config.min_expected_bps_options,
        config.decision_score_thresholds,
        config.cooldown_options,
    ):
        candidate_config = ArbiterConfig(
            rule_min_lift=float(rule_lift), ml_min_lift=float(ml_lift),
            min_expected_bps=float(min_bps), cooldown_days=int(cooldown),
            min_decision_score=float(decision_threshold),
            max_signals_per_7d=config.max_signals_per_7d,
        )
        events = run_arbiter(validation_candidates, candidate_config)
        summary, scored_rows = backtest_signal_stream(events, evaluation_universe=universe)
        metrics = _validation_metrics(summary)
        stability = _quarterly_stability(scored_rows)
        coverage = _coverage_metrics(
            events, tuple(config.currencies), tuple(config.horizons)
        )
        # Read values from user configuration instead of hard-coding them.
        currency_frequency = summary.loc[summary["scope"].eq("currency")].copy()
        # Counts are integers while calendar weeks are fractional. Permit at
        # most one-event rounding error over the entire validation interval.
        lower_rounding = 1.0 / currency_frequency["calendar_weeks"]
        frequency_ok = bool(
            currency_frequency["signals_per_week"].add(lower_rounding).ge(
                config.min_average_signals_per_week
            ).all()
            and currency_frequency["signals_per_week"].le(
                config.max_average_signals_per_week
            ).all()
        )
        rows.append({
            "rule_min_lift": rule_lift, "ml_min_lift": ml_lift,
            "min_expected_bps": min_bps,
            "min_decision_score": decision_threshold,
            "cooldown_days": cooldown,
            **metrics, **stability, **coverage,
            "stability_ok": bool(
                stability["quarterly_lift_p10"]
                >= config.validation_quarterly_lift_floor
            ),
            "frequency_ok": frequency_ok,
        })
        fitted_configs.append(candidate_config)

    leaderboard = pd.DataFrame(rows)
    structurally_eligible = (
        leaderboard["frequency_ok"] & leaderboard["coverage_ok"]
    )
    stable = structurally_eligible & leaderboard["stability_ok"]
    # A stability floor is an explicit product objective. If the small fixed
    # grid cannot reach it, do not fabricate a result: choose the best
    # structurally valid policy and expose stability_ok=False in the audit.
    selection_mask = stable if stable.any() else structurally_eligible
    eligible = leaderboard.index[selection_mask].tolist()
    if not eligible:
        diagnostic = leaderboard.sort_values(
            ["min_signals_per_week", "geometric_lift"], ascending=False
        ).head(5)
        raise ValueError(
            "На validation нет policy с 1–2 сигналами в неделю для каждой валюты "
            "и покрытием обоих бизнес-сценариев. "
            f"Лучшие диагностические строки:\n{diagnostic.to_string(index=False)}"
        )
    winner_index = max(
        eligible,
        key=lambda index: (
            leaderboard.at[index, "min_currency_quarterly_geometric_lift"],
            leaderboard.at[index, "quarterly_geometric_lift"],
            leaderboard.at[index, "quarterly_lift_p10"],
            leaderboard.at[index, "geometric_lift"],
            leaderboard.at[index, "min_currency_lift"],
            leaderboard.at[index, "macro_lift"],
            leaderboard.at[index, "macro_benefit_uplift_bps"],
            -abs(leaderboard.at[index, "mean_signals_per_week"] - 1.5),
        ),
    )
    leaderboard["selected"] = False
    leaderboard.at[winner_index, "selected"] = True
    winner = leaderboard.loc[winner_index]
    fitted = FittedArbiter(
        config=fitted_configs[winner_index],
        validation_start=start, validation_end=end,
        validation_geometric_lift=float(winner["geometric_lift"]),
        validation_macro_lift=float(winner["macro_lift"]),
        validation_min_currency_lift=float(winner["min_currency_lift"]),
        validation_macro_benefit_uplift_bps=float(winner["macro_benefit_uplift_bps"]),
        validation_mean_signals_per_week=float(winner["mean_signals_per_week"]),
        validation_quarterly_geometric_lift=float(winner["quarterly_geometric_lift"]),
        validation_min_currency_quarterly_geometric_lift=float(
            winner["min_currency_quarterly_geometric_lift"]
        ),
        validation_quarterly_lift_p10=float(winner["quarterly_lift_p10"]),
        validation_stability_ok=bool(winner["stability_ok"]),
        validation_scenario_coverage=float(winner["scenario_coverage"]),
        validation_horizon_coverage=float(winner["horizon_coverage"]),
    )
    return fitted, leaderboard.sort_values(
        ["selected", "min_currency_quarterly_geometric_lift", "geometric_lift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
