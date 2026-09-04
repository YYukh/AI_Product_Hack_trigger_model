"""Backtest финального потока MarketOpportunityEvent."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEY_COLUMNS = (
    "available_at", "currency", "scenario", "target_family", "target", "horizon"
)


def build_evaluation_universe(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    target_families: tuple[str, ...] = ("G0", "W1"),
    currencies: tuple[str, ...] | list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create one evaluable row per date/currency/target/horizon."""
    definitions = target_registry.loc[
        target_registry["family"].isin(target_families)
    ]
    selected_currencies = (
        tuple(currencies) if currencies is not None
        else tuple(sorted(data["currency"].dropna().unique()))
    )
    frames = []
    dates = pd.to_datetime(data["available_at"])
    for definition in definitions.itertuples(index=False):
        if definition.name not in data.columns:
            raise KeyError(f"В data нет target {definition.name!r}")
        mask = data["currency"].isin(selected_currencies) & data[definition.name].notna()
        if start_date is not None:
            mask &= dates.ge(pd.Timestamp(start_date))
        frame = data.loc[mask, ["available_at", "currency", definition.name]].rename(
            columns={definition.name: "target_value"}
        )
        frame = frame.assign(
            scenario=definition.scenario,
            target_family=definition.family,
            target=definition.name,
            horizon=int(definition.horizon),
        )
        benefit_column = f"local_advantage_{int(definition.horizon)}d_bps"
        frame["benefit_bps"] = (
            pd.to_numeric(data.loc[frame.index, benefit_column], errors="coerce")
            if benefit_column in data.columns else np.nan
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "target_value"])
    result = pd.concat(frames, ignore_index=True)
    result["available_at"] = pd.to_datetime(result["available_at"])
    result["target_value"] = result["target_value"].astype(bool)
    return result.loc[:, [*KEY_COLUMNS, "target_value", "benefit_bps"]]


def _metrics(sample: pd.DataFrame) -> dict:
    observations = len(sample)
    positive_count = int(sample["target_value"].sum())
    signal_count = int(sample["signal"].sum())
    true_positive = int((sample["signal"] & sample["target_value"]).sum())
    false_positive = signal_count - true_positive
    precision = true_positive / signal_count if signal_count else 0.0
    random_precision = positive_count / observations if observations else 0.0
    date_span_days = (
        sample["available_at"].max() - sample["available_at"].min()
    ).days + 1 if observations else 0
    calendar_weeks = date_span_days / 7
    benefit = (
        pd.to_numeric(sample.loc[sample["signal"], "benefit_bps"], errors="coerce")
        if "benefit_bps" in sample.columns else pd.Series(dtype=float)
    ).dropna()
    return {
        "observations": observations,
        "positive_count": positive_count,
        "signal_count": signal_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "random_precision": random_precision,
        "lift": precision / random_precision if random_precision else np.nan,
        "calendar_weeks": calendar_weeks,
        "signals_per_week": signal_count / calendar_weeks if calendar_weeks else 0.0,
        "mean_benefit_bps": float(benefit.mean()) if len(benefit) else np.nan,
        "positive_benefit_rate": float((benefit > 0).mean()) if len(benefit) else 0.0,
    }


def _summary_row(sample: pd.DataFrame, *, scope: str, values: dict) -> dict:
    return {
        "scope": scope,
        "currency": values.get("currency", "ALL"),
        "horizon": values.get("horizon", "ALL"),
        "target_family": values.get("target_family", "ALL"),
        **_metrics(sample),
    }


def backtest_signal_stream(
    events: pd.DataFrame,
    *,
    evaluation_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate final events at currency, currency×horizon and
    currency×horizon×target-family.

    Counts and precision are pooled within each displayed group; no overall or
    target-family-only rows are emitted.
    """
    events = events.copy()
    evaluation_universe = evaluation_universe.copy()
    event_required = {*KEY_COLUMNS, "event_id"}
    universe_required = {*KEY_COLUMNS, "target_value"}
    if missing := event_required.difference(events.columns):
        raise KeyError(f"В events нет полей: {sorted(missing)}")
    if missing := universe_required.difference(evaluation_universe.columns):
        raise KeyError(f"В evaluation_universe нет полей: {sorted(missing)}")

    # Empty meta-model output keeps object dtype; normalize both sides at the
    # public boundary so an empty signal stream is a valid backtest outcome.
    events["available_at"] = pd.to_datetime(
        events["available_at"], errors="raise"
    ).astype("datetime64[ns]")
    evaluation_universe["available_at"] = pd.to_datetime(
        evaluation_universe["available_at"], errors="raise"
    ).astype("datetime64[ns]")
    for column in ("currency", "scenario", "target_family", "target"):
        events[column] = events[column].astype(str)
        evaluation_universe[column] = evaluation_universe[column].astype(str)
    events["horizon"] = pd.to_numeric(events["horizon"], errors="raise").astype(int)
    evaluation_universe["horizon"] = pd.to_numeric(
        evaluation_universe["horizon"], errors="raise"
    ).astype(int)

    if events.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Meta-model должен отдавать не более одного события на ключ")

    event_index = events.loc[:, [*KEY_COLUMNS, "event_id", "confidence", "evidence_count"]]
    scored = evaluation_universe.merge(
        event_index,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    scored["signal"] = scored["event_id"].notna()

    rows: list[dict] = []
    dimensions = (
        ("currency",),
        ("currency", "horizon"),
        ("currency", "horizon", "target_family"),
    )
    for group_columns in dimensions:
        scope = "+".join(group_columns)
        for keys, sample in scored.groupby(list(group_columns), sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append(_summary_row(
                sample,
                scope=scope,
                values=dict(zip(group_columns, keys)),
            ))
    report = pd.DataFrame(rows)
    return report, scored
