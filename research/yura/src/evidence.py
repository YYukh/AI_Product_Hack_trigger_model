"""Deterministic aggregation of extensible engine outputs."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import YuraPipelineConfig
from .engines import KEYS


@dataclass
class EvidenceResult:
    opportunities: pd.DataFrame
    diagnostics: pd.DataFrame


def build_evidence_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build one causal evidence row per exact opportunity configuration."""
    required = {
        *KEYS, "engine_type", "confidence", "confidence_lift", "expected_bps",
        "baseline_probability", "target_value",
    }
    if missing := required.difference(candidates.columns):
        raise KeyError(f"В base candidates нет evidence-полей: {sorted(missing)}")
    grouped = candidates.groupby(list(KEYS), sort=False)
    if grouped["target_value"].nunique(dropna=False).gt(1).any():
        raise ValueError("Источники расходятся в target_value одного ключа")
    base = grouped.agg(
        target_value=("target_value", "first"),
        expected_bps=("expected_bps", "max"),
        baseline_probability=("baseline_probability", "max"),
        evidence_count=("engine_type", "size"),
    ).reset_index()
    ml = (
        candidates.loc[candidates["engine_type"].eq("ml")]
        .groupby(list(KEYS), sort=False)
        .agg(ml_confidence=("confidence", "max"), ml_lift=("confidence_lift", "max"))
        .reset_index()
    )
    rule = (
        candidates.loc[candidates["engine_type"].eq("rule")]
        .groupby(list(KEYS), sort=False)
        .agg(
            rule_confidence=("confidence", "max"),
            rule_lift=("confidence_lift", "max"),
            rule_count=("engine_name", "nunique"),
        )
        .reset_index()
    )
    result = base.merge(ml, on=list(KEYS), how="left", validate="one_to_one")
    result = result.merge(rule, on=list(KEYS), how="left", validate="one_to_one")
    result["ml_confidence"] = result["ml_confidence"].fillna(
        result["baseline_probability"]
    )
    result["ml_lift"] = result["ml_lift"].fillna(1.0)
    result["rule_confidence"] = result["rule_confidence"].fillna(
        result["baseline_probability"]
    )
    result["rule_lift"] = result["rule_lift"].fillna(1.0)
    result["rule_count"] = result["rule_count"].fillna(0).astype(int)
    return result.sort_values(
        ["available_at", "currency", "target_family", "horizon"]
    )


def _rolling_percentile_rank(
    dates: pd.Series,
    values: pd.Series,
    *,
    window_months: int,
) -> pd.Series:
    """Rank every current value against a causal trailing calendar window."""
    ordered = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="raise"),
        "value": pd.to_numeric(values, errors="coerce"),
    }, index=values.index).sort_values("date")
    active_values: list[float] = []
    active_rows: list[tuple[pd.Timestamp, float]] = []
    left = 0
    output = pd.Series(np.nan, index=ordered.index, dtype=float)

    for index, row in ordered.iterrows():
        current_date = pd.Timestamp(row["date"])
        lower = current_date - pd.DateOffset(months=window_months)
        while left < len(active_rows) and active_rows[left][0] < lower:
            expired = active_rows[left][1]
            position = bisect_left(active_values, expired)
            if position < len(active_values) and active_values[position] == expired:
                active_values.pop(position)
            left += 1
        current = float(row["value"])
        if np.isfinite(current):
            insort(active_values, current)
            active_rows.append((current_date, current))
            lo = bisect_left(active_values, current)
            hi = bisect_right(active_values, current)
            # Average rank for ties, expressed as a percentile in (0, 1].
            output.loc[index] = (lo + hi + 1.0) / (2.0 * len(active_values))
        else:
            active_rows.append((current_date, current))
    return output.reindex(values.index)


def add_causal_relative_scores(
    candidates: pd.DataFrame,
    *,
    window_months: int = 36,
) -> pd.DataFrame:
    """Normalize scales using only current and trailing-window evidence."""
    if window_months <= 0:
        raise ValueError("window_months должен быть положительным")
    result = candidates.copy().sort_values(
        ["available_at", "currency", "target_family", "horizon"]
    )
    confidence = pd.to_numeric(result["confidence"], errors="coerce").clip(
        1e-6, 1.0 - 1e-6
    )
    baseline = pd.to_numeric(
        result["baseline_probability"], errors="coerce"
    ).clip(1e-6, 1.0 - 1e-6)
    result["statistical_evidence"] = (
        np.log(confidence / (1.0 - confidence))
        - np.log(baseline / (1.0 - baseline))
    )
    horizon_scale = np.sqrt(pd.to_numeric(result["horizon"], errors="raise"))
    result["economic_evidence"] = (
        pd.to_numeric(result["expected_bps"], errors="coerce") / horizon_scale
    )
    groups = ["currency", "target_family", "horizon"]
    result["statistical_rank"] = np.nan
    result["economic_rank"] = np.nan
    for _, group in result.groupby(groups, sort=False):
        result.loc[group.index, "statistical_rank"] = _rolling_percentile_rank(
            group["available_at"], group["statistical_evidence"],
            window_months=window_months,
        )
        result.loc[group.index, "economic_rank"] = _rolling_percentile_rank(
            group["available_at"], group["economic_evidence"],
            window_months=window_months,
        )
    result["decision_score"] = np.sqrt(
        result["statistical_rank"].clip(0.0, 1.0)
        * result["economic_rank"].clip(0.0, 1.0)
    )
    return result.reset_index(drop=True)


def aggregate_engine_evidence(
    candidates: pd.DataFrame,
    *,
    config: YuraPipelineConfig = YuraPipelineConfig(),
) -> EvidenceResult:
    """Create one opportunity per date × currency × target × horizon.

    No model is fitted here. ML and simple rules retain equal access to the
    pipeline: whichever source has the larger causal lift supplies confidence,
    while all source counts remain attached as transparent evidence.
    """
    matrix = build_evidence_matrix(candidates)
    ml_names = (
        candidates.loc[candidates["engine_type"].eq("ml")]
        .sort_values([*KEYS, "confidence"], ascending=[True] * len(KEYS) + [False])
        .drop_duplicates(list(KEYS))
        .loc[:, [*KEYS, "engine_name", "engine_version"]]
        .rename(columns={
            "engine_name": "best_ml_engine",
            "engine_version": "best_ml_version",
        })
    )
    rule_names = (
        candidates.loc[candidates["engine_type"].eq("rule")]
        .sort_values([*KEYS, "confidence"], ascending=[True] * len(KEYS) + [False])
        .drop_duplicates(list(KEYS))
        .loc[:, [*KEYS, "engine_name", "engine_version"]]
        .rename(columns={
            "engine_name": "best_rule_engine",
            "engine_version": "best_rule_version",
        })
    )
    matrix = matrix.merge(ml_names, on=list(KEYS), how="left", validate="one_to_one")
    matrix = matrix.merge(rule_names, on=list(KEYS), how="left", validate="one_to_one")
    rule_wins = matrix["rule_count"].gt(0) & matrix["rule_lift"].gt(
        matrix["ml_lift"]
    )
    matrix["confidence"] = np.where(
        rule_wins, matrix["rule_confidence"], matrix["ml_confidence"]
    )
    matrix["confidence_lift"] = np.where(
        rule_wins, matrix["rule_lift"], matrix["ml_lift"]
    )
    matrix["engine_type"] = np.where(rule_wins, "rule", "ml")
    matrix["engine_name"] = np.where(
        rule_wins, matrix["best_rule_engine"], matrix["best_ml_engine"]
    )
    matrix["engine_name"] = matrix["engine_name"].fillna("unavailable_engine")
    matrix["engine_version"] = np.where(
        rule_wins, matrix["best_rule_version"], matrix["best_ml_version"]
    )
    matrix["engine_version"] = matrix["engine_version"].fillna("unavailable_version")
    matrix["aggregation_version"] = "deterministic_evidence_v1"
    matrix["horizon"] = pd.to_numeric(matrix["horizon"], errors="raise").astype(int)
    opportunities = add_causal_relative_scores(
        matrix, window_months=config.relative_rank_months
    )
    diagnostics = (
        opportunities.groupby(
            ["currency", "target_family", "horizon", "engine_type"], sort=True
        )
        .agg(
            observations=("available_at", "size"),
            mean_confidence=("confidence", "mean"),
            mean_confidence_lift=("confidence_lift", "mean"),
            mean_expected_bps=("expected_bps", "mean"),
            mean_rule_count=("rule_count", "mean"),
        )
        .reset_index()
    )
    return EvidenceResult(
        opportunities=opportunities.loc[:, [
            *KEYS, "engine_type", "engine_name", "engine_version",
            "confidence", "baseline_probability", "confidence_lift",
            "expected_bps", "target_value", "evidence_count", "rule_count",
            "aggregation_version",
            "statistical_evidence", "economic_evidence",
            "statistical_rank", "economic_rank", "decision_score",
        ]],
        diagnostics=diagnostics,
    )
