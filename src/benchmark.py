"""Comparable runs of the fixed Yura architecture across public switches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import YuraPipelineConfig
from .engine_registry import EngineRegistry
from .pipeline import (
    YuraPipelineResult, prepare_yura_pipeline, run_prepared_yura_pipeline,
)
from .selector import build_opportunity_selector


DEFAULT_ML_SCOPES = ("pooled", "hybrid", "per_currency")
DEFAULT_SELECTOR_TYPES = ("threshold", "logistic_regression", "extra_trees")


@dataclass
class YuraVariantBenchmark:
    results: dict[tuple[str, str], YuraPipelineResult]
    summary: pd.DataFrame
    failures: pd.DataFrame


def _weighted_mean(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows[column], errors="coerce")
    weights = pd.to_numeric(rows["signal_count"], errors="coerce").fillna(0.0)
    valid = values.notna() & weights.gt(0.0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def summarize_yura_variant(
    result: YuraPipelineResult,
    *,
    ml_scope: str,
    selector_type: str,
) -> dict:
    """Build one comparable row without inventing an aggregate target baseline."""
    currency = result.backtest_summary.loc[
        result.backtest_summary["scope"].eq("currency")
    ].copy()
    signal_count = int(currency["signal_count"].sum())
    true_positive = int(currency["true_positive"].sum())
    precision = true_positive / signal_count if signal_count else 0.0
    if signal_count:
        random_precision = float(np.average(
            pd.to_numeric(currency["random_precision"], errors="coerce"),
            weights=pd.to_numeric(currency["signal_count"], errors="coerce"),
        ))
    else:
        random_precision = np.nan
    lifts = pd.to_numeric(currency["lift"], errors="coerce").fillna(0.0)
    quarterly = result.holdout_quarterly_stability.copy()
    quarterly_lift = pd.to_numeric(quarterly["lift"], errors="coerce").dropna()
    fitted = result.fitted_arbiter
    policy = getattr(fitted, "policy", None)
    if policy is None:
        policy = getattr(fitted, "config", None)
    return {
        "ml_scope": ml_scope,
        "selector_type": selector_type,
        "selected_threshold": float(getattr(fitted, "threshold", np.nan)),
        "cooldown_days": int(getattr(policy, "cooldown_days", 0)),
        "validation_geometric_lift": float(
            getattr(fitted, "validation_geometric_lift", np.nan)
        ),
        "validation_min_currency_lift": float(
            getattr(fitted, "validation_min_currency_lift", np.nan)
        ),
        "validation_benefit_uplift_bps": float(
            getattr(fitted, "validation_macro_benefit_uplift_bps", np.nan)
        ),
        "validation_signals_per_week": float(
            getattr(fitted, "validation_mean_signals_per_week", np.nan)
        ),
        "holdout_signal_count": signal_count,
        "holdout_signals_per_week": float(currency["signals_per_week"].mean()),
        "holdout_precision": precision,
        "holdout_random_precision": random_precision,
        "holdout_lift": (
            precision / random_precision
            if pd.notna(random_precision) and random_precision > 0 else np.nan
        ),
        "holdout_macro_lift": float(lifts.mean()),
        "holdout_min_currency_lift": float(lifts.min()),
        "holdout_mean_benefit_bps": _weighted_mean(currency, "mean_benefit_bps"),
        "holdout_benefit_uplift_bps": _weighted_mean(
            currency, "benefit_uplift_bps"
        ),
        "holdout_positive_benefit_rate": _weighted_mean(
            currency, "positive_benefit_rate"
        ),
        "holdout_quarterly_geometric_lift": (
            float(np.exp(np.log(quarterly_lift.clip(lower=1e-9)).mean()))
            if not quarterly_lift.empty else np.nan
        ),
        "holdout_quarterly_lift_p10": (
            float(quarterly_lift.quantile(0.10))
            if not quarterly_lift.empty else np.nan
        ),
        "holdout_min_quarterly_lift": (
            float(quarterly_lift.min()) if not quarterly_lift.empty else np.nan
        ),
        "holdout_quarters_below_one": int(quarterly_lift.lt(1.0).sum()),
        "holdout_nonzero_configurations": int(len(result.holdout_coverage)),
    }


def run_yura_variant_matrix(
    scoring_data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    base_config: YuraPipelineConfig,
    ml_scopes: tuple[str, ...] = DEFAULT_ML_SCOPES,
    selector_types: tuple[str, ...] = DEFAULT_SELECTOR_TYPES,
    engine_registry: EngineRegistry | None = None,
    progress: Callable[[str], None] | None = print,
) -> YuraVariantBenchmark:
    """Run all scope-selector pairs, sharing expensive base WF within a scope.

    Holdout metrics are reported for diagnosis, not used to choose a winner.
    Selection between variants must be made from pre-holdout validation metrics.
    """
    results: dict[tuple[str, str], YuraPipelineResult] = {}
    rows: list[dict] = []
    failures: list[dict] = []
    for ml_scope in ml_scopes:
        config = replace(base_config, ml_scope=ml_scope)
        if progress is not None:
            progress(f"[{ml_scope}] base walk-forward")
        try:
            prepared = prepare_yura_pipeline(
                scoring_data,
                target_registry=target_registry,
                config=config,
                engine_registry=engine_registry,
            )
        except Exception as error:
            for selector_type in selector_types:
                failures.append({
                    "ml_scope": ml_scope,
                    "selector_type": selector_type,
                    "stage": "base_walk_forward",
                    "error": f"{type(error).__name__}: {error}",
                })
            continue

        for selector_type in selector_types:
            if progress is not None:
                progress(f"[{ml_scope} × {selector_type}] selector")
            try:
                result = run_prepared_yura_pipeline(
                    prepared,
                    scoring_data,
                    selector=build_opportunity_selector(selector_type),
                )
                results[(ml_scope, selector_type)] = result
                rows.append(summarize_yura_variant(
                    result,
                    ml_scope=ml_scope,
                    selector_type=selector_type,
                ))
            except Exception as error:
                failures.append({
                    "ml_scope": ml_scope,
                    "selector_type": selector_type,
                    "stage": "selector",
                    "error": f"{type(error).__name__}: {error}",
                })
    return YuraVariantBenchmark(
        results=results,
        summary=pd.DataFrame(rows),
        failures=pd.DataFrame(failures),
    )
