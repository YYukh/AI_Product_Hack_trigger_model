"""Единый контракт evidence-сигналов rule-based и ML движков."""

from __future__ import annotations

import hashlib

import pandas as pd


SIGNAL_SCHEMA_VERSION = "1.0"
SIGNAL_COLUMNS = (
    "schema_version",
    "signal_id",
    "available_at",
    "as_of",
    "currency",
    "corridor",
    "scenario",
    "target_family",
    "target",
    "horizon",
    "engine_id",
    "engine_type",
    "engine_name",
    "engine_version",
    "model_version",
    "status",
    "signal",
    "raw_score",
    "decision_threshold",
    "confidence",
    "confidence_method",
    "confidence_support",
    "baseline_probability",
    "confidence_lift",
    "trained_at",
    "trained_through",
    "next_retrain_at",
    "last_retrain_error",
    "expires_at",
)


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"В {name} нет полей: {sorted(missing)}")


def _moscow_scoring_time(values: pd.Series, hour: int = 9) -> pd.Series:
    dates = pd.to_datetime(values).dt.normalize()
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("Europe/Moscow")
    else:
        dates = dates.dt.tz_convert("Europe/Moscow")
    return dates + pd.Timedelta(hours=hour)


def _signal_ids(data: pd.DataFrame) -> pd.Series:
    identity = data[
        ["as_of", "currency", "target", "horizon", "engine_type", "engine_version"]
    ].astype(str).agg("|".join, axis=1)
    return identity.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:24])


def _finish_contract(data: pd.DataFrame, *, validity_hours: int) -> pd.DataFrame:
    result = data.copy()
    result["schema_version"] = SIGNAL_SCHEMA_VERSION
    result["available_at"] = pd.to_datetime(result["available_at"])
    result["as_of"] = _moscow_scoring_time(result["available_at"])
    result["corridor"] = "RUB_" + result["currency"].astype(str)
    result["signal"] = True
    result["expires_at"] = result["as_of"] + pd.Timedelta(hours=validity_hours)
    result["signal_id"] = _signal_ids(result)
    result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")
    result["confidence_support"] = pd.to_numeric(
        result["confidence_support"], errors="coerce"
    ).fillna(0).astype(int)
    for column in ("baseline_probability", "confidence_lift"):
        if column not in result:
            result[column] = float("nan")
    if "engine_id" not in result:
        result["engine_id"] = (
            result["engine_type"].astype(str) + ":"
            + result["currency"].astype(str) + ":"
            + result["target_family"].astype(str) + ":h"
            + result["horizon"].astype(str)
        )
    if "model_version" not in result:
        result["model_version"] = result["engine_version"]
    if "status" not in result:
        result["status"] = "READY"
    for column in (
        "trained_at", "trained_through", "next_retrain_at",
        "last_retrain_error",
    ):
        if column not in result:
            result[column] = None
    return result.loc[:, SIGNAL_COLUMNS].sort_values(
        ["as_of", "currency", "target", "horizon", "engine_type"]
    ).reset_index(drop=True)


def standardize_rule_signals(
    signals: pd.DataFrame,
    *,
    selected_indicators: pd.DataFrame,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Convert rule backtest emissions to the shared evidence contract.

    Confidence is discovery pooled OOS precision, never final holdout precision.
    """
    signal_required = {
        "available_at", "currency", "target_value", "scenario",
        "target_family", "target", "horizon", "indicator",
        "fixed_candidate", "rebalance_months", "fold_id",
    }
    selection_required = {
        "currency", "target", "horizon", "oos_precision",
        "test_predicted_positive_count",
    }
    _require_columns(signals, signal_required, "rule signals")
    _require_columns(selected_indicators, selection_required, "selected_indicators")

    confidence = selected_indicators.loc[:, [
        "currency", "target", "horizon", "oos_precision",
        "test_predicted_positive_count",
    ]].drop_duplicates(["currency", "target", "horizon"])
    result = signals.merge(
        confidence,
        on=["currency", "target", "horizon"],
        how="left",
        validate="many_to_one",
    )
    result["engine_type"] = "rule"
    result["engine_name"] = result["indicator"]
    result["engine_version"] = (
        result["fixed_candidate"].astype(str)
        + ":rebalance_"
        + result["rebalance_months"].astype(str)
        + "m"
    )
    result["raw_score"] = 1.0
    result["decision_threshold"] = 1.0
    result["confidence"] = result["oos_precision"]
    result["confidence_method"] = "pooled_discovery_oos_precision"
    result["confidence_support"] = result["test_predicted_positive_count"]
    result["baseline_probability"] = float("nan")
    result["confidence_lift"] = float("nan")
    return _finish_contract(result, validity_hours=validity_hours)


def standardize_ml_signals(
    signals: pd.DataFrame,
    *,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Convert ML backtest emissions to the shared evidence contract."""
    required = {
        "available_at", "currency", "target_value", "scenario",
        "target_family", "target", "horizon", "indicator",
        "rebalance_months", "fold_id", "probability",
        "probability_threshold", "confidence", "confidence_method",
        "confidence_support",
    }
    _require_columns(signals, required, "ML signals")
    result = signals.copy()
    result["engine_type"] = "ml"
    result["engine_name"] = result["indicator"]
    result["engine_version"] = (
        result["indicator"].astype(str)
        + ":rebalance_"
        + result["rebalance_months"].astype(str)
        + "m"
    )
    result["raw_score"] = result["probability"]
    result["decision_threshold"] = result["probability_threshold"]
    return _finish_contract(result, validity_hours=validity_hours)


def finalize_signal_evidence(
    signals: pd.DataFrame,
    *,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Finalize engine output that already contains contract attributes."""
    required = {
        "available_at", "currency", "scenario", "target_family", "target",
        "horizon", "engine_type", "engine_name", "engine_version",
        "raw_score", "decision_threshold", "confidence",
        "confidence_method", "confidence_support",
    }
    _require_columns(signals, required, "engine signals")
    return _finish_contract(signals, validity_hours=validity_hours)


def combine_evidence_streams(*streams: pd.DataFrame) -> pd.DataFrame:
    """Concatenate already standardized streams and reject duplicate IDs."""
    non_empty = [stream for stream in streams if not stream.empty]
    if not non_empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    for stream in non_empty:
        _require_columns(stream, set(SIGNAL_COLUMNS), "evidence stream")
    result = pd.concat(non_empty, ignore_index=True).loc[:, SIGNAL_COLUMNS]
    if result["signal_id"].duplicated().any():
        duplicates = result.loc[result["signal_id"].duplicated(), "signal_id"].tolist()
        raise ValueError(f"Повторяющиеся signal_id: {duplicates[:5]}")
    return result.sort_values(
        ["as_of", "currency", "target", "horizon", "engine_type"]
    ).reset_index(drop=True)
