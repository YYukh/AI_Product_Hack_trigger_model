"""Диагностика таргетов до обучения индикаторов и ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_target_family_frequency(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    min_signals_per_week: float = 2.0,
    max_signals_per_week: float = 3.0,
) -> pd.DataFrame:
    """Оценить теоретическую частоту уникальных сигналов семейства.

    Для currency × target family дата считается одним сигналом, если в этот
    день положителен хотя бы один вариант target этого семейства. Пересечения
    порогов и горизонтов не увеличивают число уникальных сигналов.

    В расчёт входят только даты, размеченные для всех вариантов семейства:
    так разные горизонты сравниваются на едином доступном периоде.
    """
    if min_signals_per_week < 0:
        raise ValueError("min_signals_per_week не может быть отрицательным")
    if max_signals_per_week < min_signals_per_week:
        raise ValueError("max_signals_per_week должен быть не меньше min")

    registry_required = {"name", "scenario", "family", "horizon"}
    registry_missing = registry_required.difference(target_registry.columns)
    if registry_missing:
        raise KeyError(
            f"В target_registry нет полей: {sorted(registry_missing)}"
        )
    required = {"available_at", "currency", *target_registry["name"]}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(
            f"Не хватает полей для анализа частоты: {sorted(missing)}"
        )
    if data.duplicated(["currency", "available_at"]).any():
        raise ValueError("Для currency × available_at ожидается одна строка")

    rows: list[dict] = []
    for family, definitions in target_registry.groupby("family", sort=True):
        target_names = definitions["name"].tolist()
        scenarios = definitions["scenario"].dropna().unique()
        if len(scenarios) != 1:
            raise ValueError(f"У family={family} должен быть один scenario")

        for currency, currency_data in data.groupby("currency", sort=True):
            labelled = currency_data[target_names].notna().all(axis=1)
            sample = currency_data.loc[
                labelled,
                ["available_at", *target_names],
            ].sort_values("available_at")
            if sample.empty:
                continue

            positives = sample[target_names].astype(bool)
            raw_positive_count = int(positives.to_numpy().sum())
            unique_signal_mask = positives.any(axis=1)
            unique_signal_count = int(unique_signal_mask.sum())
            period_start = pd.Timestamp(sample["available_at"].min())
            period_end = pd.Timestamp(sample["available_at"].max())
            calendar_days = (period_end - period_start).days + 1
            calendar_weeks = calendar_days / 7
            observations = len(sample)
            signals_per_week = unique_signal_count / calendar_weeks
            target_share = unique_signal_count / observations
            min_required_share = (
                min_signals_per_week * calendar_weeks / observations
            )
            max_required_share = (
                max_signals_per_week * calendar_weeks / observations
            )

            rows.append(
                {
                    "currency": currency,
                    "scenario": scenarios[0],
                    "target_family": family,
                    "configuration_count": len(target_names),
                    "observations": observations,
                    "period_start": period_start,
                    "period_end": period_end,
                    "calendar_weeks": calendar_weeks,
                    "raw_positive_count_with_duplicates": raw_positive_count,
                    "unique_signal_count": unique_signal_count,
                    "overlap_removed_count": (
                        raw_positive_count - unique_signal_count
                    ),
                    "unique_target_share": target_share,
                    "min_required_target_share": min_required_share,
                    "max_required_target_share": max_required_share,
                    "unique_signals_per_week": signals_per_week,
                    "target_frequency_status": (
                        "BELOW"
                        if signals_per_week < min_signals_per_week
                        else "ABOVE"
                        if signals_per_week > max_signals_per_week
                        else "OK"
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["scenario", "target_family", "currency"]
    ).reset_index(drop=True)


def evaluate_perfect_target_lift(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Посчитать теоретический lift идеального предсказания таргета.

    Идеальный сигнал возникает ровно при Y=1, поэтому его precision равна
    единице. Precision случайного входа равна prevalence таргета на том же
    наборе доступных наблюдений. Следовательно, ideal_lift = 1 / prevalence.
    """
    registry_required = {"name", "scenario", "family", "horizon"}
    registry_missing = registry_required.difference(target_registry.columns)
    if registry_missing:
        raise KeyError(
            f"В target_registry нет полей: {sorted(registry_missing)}"
        )

    targets = target_registry["name"].tolist()
    required = {"currency", *targets}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Не хватает полей для target study: {sorted(missing)}")

    result = (
        data[["currency", *targets]]
        .melt(
            id_vars="currency",
            var_name="target",
            value_name="target_value",
        )
        .dropna(subset=["target_value"])
        .groupby(["currency", "target"], as_index=False)
        .agg(
            observations=("target_value", "size"),
            positive_count=("target_value", "sum"),
            random_precision=("target_value", "mean"),
        )
    )
    has_positive = result["positive_count"].gt(0)
    result["perfect_precision"] = np.where(has_positive, 1.0, np.nan)
    result["ideal_lift_vs_random"] = np.divide(
        result["perfect_precision"],
        result["random_precision"],
        out=np.full(len(result), np.nan),
        where=result["random_precision"].gt(0),
    )

    registry = target_registry.rename(
        columns={"name": "target", "family": "target_family"}
    )
    return (
        result.merge(
            registry,
            on="target",
            how="left",
            validate="many_to_one",
        )
        .sort_values(
            ["scenario", "target_family", "target", "horizon", "currency"]
        )
        .reset_index(drop=True)
    )

