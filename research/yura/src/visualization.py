"""Presentation-ready diagnostics for the alternative pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RED = "#d32f2f"
DARK_RED = "#8b0000"
BLACK = "#161616"
GRAY = "#777777"
LIGHT_GRAY = "#e5e5e5"

VARIANT_COLORS = {
    "pooled": "#d32f2f",
    "hybrid": "#161616",
    "per_currency": "#888888",
}
VARIANT_LINESTYLES = {
    "threshold": "-",
    "logistic_regression": "--",
    "extra_trees": ":",
}
VARIANT_MARKERS = {
    "threshold": "o",
    "logistic_regression": "s",
    "extra_trees": "^",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": BLACK, "text.color": BLACK,
        "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
        "font.size": 11,
    })


def plot_validation_frontier(leaderboard: pd.DataFrame):
    """Lift/BPS/frequency trade-off of the frozen validation choice."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))
    eligible = leaderboard["frequency_ok"].astype(bool)
    scatter = ax.scatter(
        leaderboard["mean_signals_per_week"], leaderboard["geometric_lift"],
        c=leaderboard["macro_benefit_uplift_bps"], cmap="Reds",
        s=np.where(eligible, 55, 18), alpha=np.where(eligible, 0.9, 0.25),
        edgecolors=np.where(eligible, BLACK, "none"), linewidths=0.5,
    )
    selected = leaderboard.loc[leaderboard["selected"]]
    ax.scatter(
        selected["mean_signals_per_week"], selected["geometric_lift"],
        marker="*", s=320, color=DARK_RED, edgecolor=BLACK, linewidth=1.0,
        label="Замороженная конфигурация", zorder=5,
    )
    ax.axvspan(1, 2, color=LIGHT_GRAY, alpha=0.45, label="Допустимая частота")
    ax.axhline(1, color=GRAY, linestyle="--", linewidth=1)
    ax.set(
        title="Validation: устойчивый lift, BPS и частота сигналов",
        xlabel="Среднее число сигналов в неделю по валюте",
        ylabel="Геометрическое среднее lift по валютам",
    )
    fig.colorbar(scatter, ax=ax, label="Средний benefit uplift, BPS")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_holdout_horizon_mix(signals: pd.DataFrame):
    _style()
    if signals.empty:
        raise ValueError("Нет holdout-сигналов")
    table = signals.pivot_table(
        index="currency", columns="horizon", values="event_id",
        aggfunc="count", fill_value=0,
    )
    table = table.reindex(columns=sorted(table.columns))
    colors = ["#202020", "#555555", "#8b0000", "#d32f2f", "#ef9a9a"][:len(table.columns)]
    ax = table.plot(kind="bar", stacked=True, figsize=(10, 6), color=colors)
    ax.set(
        title="Holdout: состав конечного потока по горизонтам",
        xlabel="Валюта", ylabel="Число сигналов",
    )
    ax.legend(title="Горизонт", frameon=False, ncol=len(table.columns))
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    plt.xticks(rotation=0)
    ax.figure.tight_layout()
    return ax.figure


def build_sequential_metrics(
    backtest_rows: pd.DataFrame,
    *,
    rolling_signals: int = 25,
) -> pd.DataFrame:
    """Causal cumulative and trailing-window metrics after every signal."""
    if rolling_signals <= 0:
        raise ValueError("rolling_signals должен быть положительным")
    required = {
        "available_at", "currency", "signal", "target_value",
        "_stratum_random_precision", "benefit_bps",
        "_stratum_random_benefit_bps",
    }
    if missing := required.difference(backtest_rows.columns):
        raise KeyError(f"В backtest_rows нет полей: {sorted(missing)}")

    signals = backtest_rows.loc[backtest_rows["signal"]].copy()
    if signals.empty:
        raise ValueError("Нет holdout-сигналов")
    signals["available_at"] = pd.to_datetime(signals["available_at"])
    signals = signals.sort_values(["currency", "available_at"]).reset_index(drop=True)
    output: list[pd.DataFrame] = []

    for _, sample in signals.groupby("currency", sort=True):
        sample = sample.sort_values("available_at").copy()
        hit = sample["target_value"].astype(float)
        random_probability = pd.to_numeric(
            sample["_stratum_random_precision"], errors="coerce"
        )
        benefit_delta = (
            pd.to_numeric(sample["benefit_bps"], errors="coerce")
            - pd.to_numeric(
                sample["_stratum_random_benefit_bps"], errors="coerce"
            )
        )
        count = pd.Series(np.arange(1, len(sample) + 1), index=sample.index)
        sample["signal_number"] = count.to_numpy()
        sample["cumulative_precision"] = hit.cumsum() / count
        sample["cumulative_random_precision"] = random_probability.cumsum() / count
        sample["cumulative_lift"] = (
            sample["cumulative_precision"]
            / sample["cumulative_random_precision"].replace(0.0, np.nan)
        )
        sample["cumulative_benefit_uplift_bps"] = (
            benefit_delta.expanding(min_periods=1).mean()
        )

        rolling_hits = hit.rolling(rolling_signals, min_periods=rolling_signals).sum()
        rolling_expected = random_probability.rolling(
            rolling_signals, min_periods=rolling_signals
        ).sum()
        sample["rolling_lift"] = rolling_hits / rolling_expected.replace(0.0, np.nan)
        sample["rolling_benefit_uplift_bps"] = benefit_delta.rolling(
            rolling_signals, min_periods=rolling_signals
        ).mean()
        sample["rolling_signals"] = int(rolling_signals)
        output.append(sample)
    return pd.concat(output, ignore_index=True)


def plot_holdout_sequential_metrics(
    backtest_rows: pd.DataFrame,
    *,
    rolling_signals: int = 25,
    warmup_signals: int = 20,
    title: str = "Holdout: накопительная и текущая устойчивость после каждого сигнала",
):
    """Plot cumulative history and recent quality after every signal."""
    _style()
    metrics = build_sequential_metrics(
        backtest_rows, rolling_signals=rolling_signals
    )
    currencies = sorted(metrics["currency"].unique())
    fig, axes = plt.subplots(
        len(currencies), 2, figsize=(16, 3.6 * len(currencies)), squeeze=False
    )
    for row_index, currency in enumerate(currencies):
        sample = metrics.loc[metrics["currency"].eq(currency)]
        dates = sample["available_at"]

        lift_ax = axes[row_index, 0]
        lift_ax.plot(
            dates, sample["cumulative_lift"], color=RED, linewidth=2.2,
            label="Накопительный lift",
        )
        lift_ax.plot(
            dates, sample["rolling_lift"], color=BLACK, linewidth=1.5,
            alpha=0.8, label=f"Последние {rolling_signals} сигналов",
        )
        lift_ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1)
        lift_ax.set_title(f"RUB/{currency}: lift", fontweight="bold")
        lift_ax.set_ylabel("Lift")

        bps_ax = axes[row_index, 1]
        bps_ax.plot(
            dates, sample["cumulative_benefit_uplift_bps"],
            color=RED, linewidth=2.2, label="Накопительный BPS uplift",
        )
        bps_ax.plot(
            dates, sample["rolling_benefit_uplift_bps"],
            color=BLACK, linewidth=1.5, alpha=0.8,
            label=f"Последние {rolling_signals} сигналов",
        )
        bps_ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
        bps_ax.set_title(f"RUB/{currency}: экономический эффект", fontweight="bold")
        bps_ax.set_ylabel("Benefit uplift, BPS")

        if warmup_signals > 0 and len(sample) >= 2:
            warmup_end = sample.iloc[min(warmup_signals, len(sample)) - 1]["available_at"]
            for ax in (lift_ax, bps_ax):
                ax.axvspan(
                    sample["available_at"].iloc[0], warmup_end,
                    color=LIGHT_GRAY, alpha=0.45, linewidth=0,
                )
        for ax in (lift_ax, bps_ax):
            ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
            ax.tick_params(axis="x", rotation=25)
            ax.legend(frameon=False, loc="best")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_holdout_quarterly_lift(
    quarterly_report: pd.DataFrame,
    *,
    currencies: tuple[str, ...] | list[str] | None = None,
    title: str = "Устойчивость lift по кварталам и валютам",
):
    """Plot raw quarterly lift and the actual signal count behind each point."""
    _style()
    required = {"currency", "quarter", "signal_count", "lift"}
    if missing := required.difference(quarterly_report.columns):
        raise KeyError(f"В quarterly_report нет полей: {sorted(missing)}")
    rows = quarterly_report.copy()
    if rows.empty:
        raise ValueError("Квартальный holdout-отчёт пуст")
    rows["quarter"] = rows["quarter"].astype(str)
    rows["lift"] = pd.to_numeric(rows["lift"], errors="coerce")
    rows = rows.sort_values(["currency", "quarter"])
    available = set(rows["currency"])
    selected = (
        [currency for currency in currencies if currency in available]
        if currencies is not None else sorted(available)
    )
    columns = 2
    plot_rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(
        plot_rows, columns, figsize=(15, 4.2 * plot_rows),
        constrained_layout=True, squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, currency in zip(flat_axes, selected):
        sample = rows.loc[rows["currency"].eq(currency)].reset_index(drop=True)
        positions = np.arange(len(sample))
        ax.plot(
            positions, sample["lift"], color=RED, marker="o",
            markersize=6, linewidth=2.2,
        )
        ax.axhline(1.0, color=BLACK, linestyle="--", linewidth=1.1)
        ax.set_xticks(
            positions, sample["quarter"], rotation=35, ha="right"
        )
        ax.set_title(f"RUB/{currency}", loc="left", fontweight="bold")
        ax.set_ylabel("Квартальный lift")
        ax.set_facecolor("#f7f7f7")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
        finite = sample["lift"].dropna()
        upper = max(1.25, float(finite.max()) * 1.18) if len(finite) else 1.25
        ax.set_ylim(-0.08, upper)
        for position, row in sample.iterrows():
            if pd.notna(row["lift"]):
                ax.annotate(
                    f"n={int(row['signal_count'])}",
                    (position, row["lift"]), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=BLACK,
                )
    for ax in flat_axes[len(selected):]:
        ax.remove()
    fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig


def _ordered_variant_keys(results: Mapping) -> list[tuple[str, str]]:
    preferred = [
        (scope, selector)
        for scope in ("pooled", "hybrid", "per_currency")
        for selector in ("threshold", "logistic_regression", "extra_trees")
    ]
    return [key for key in preferred if key in results]


def plot_variant_cumulative_lift(results: Mapping):
    """Compare cumulative holdout lift of all scope-selector variants."""
    _style()
    keys = _ordered_variant_keys(results)
    if not keys:
        raise ValueError("Нет успешно рассчитанных вариантов")
    sequential = {
        key: build_sequential_metrics(results[key].backtest_rows)
        for key in keys
    }
    currencies = list(results[keys[0]].config.currencies)
    fig, axes = plt.subplots(
        math.ceil(len(currencies) / 2), 2,
        figsize=(17, 4.2 * math.ceil(len(currencies) / 2)),
        constrained_layout=True, squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, currency in zip(flat_axes, currencies):
        for scope, selector in keys:
            sample = sequential[(scope, selector)]
            sample = sample.loc[sample["currency"].eq(currency)]
            ax.plot(
                sample["available_at"], sample["cumulative_lift"],
                color=VARIANT_COLORS[scope],
                linestyle=VARIANT_LINESTYLES[selector], linewidth=1.8,
                alpha=0.9, label=f"{scope} | {selector}",
            )
        ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
        ax.set_title(f"RUB/{currency}", loc="left", fontweight="bold")
        ax.set_ylabel("Накопительный lift")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)
    for ax in flat_axes[len(currencies):]:
        ax.remove()
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Holdout: накопительный lift — все варианты",
        fontsize=16, fontweight="bold", y=1.045,
    )
    return fig


def plot_variant_quarterly_lift(results: Mapping):
    """Compare raw quarterly holdout lift of all scope-selector variants."""
    _style()
    keys = _ordered_variant_keys(results)
    if not keys:
        raise ValueError("Нет успешно рассчитанных вариантов")
    currencies = list(results[keys[0]].config.currencies)
    fig, axes = plt.subplots(
        math.ceil(len(currencies) / 2), 2,
        figsize=(17, 4.2 * math.ceil(len(currencies) / 2)),
        constrained_layout=True, squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, currency in zip(flat_axes, currencies):
        reference_quarters: list[str] = []
        for scope, selector in keys:
            sample = results[(scope, selector)].holdout_quarterly_stability.copy()
            sample = sample.loc[sample["currency"].eq(currency)].sort_values("quarter")
            quarters = sample["quarter"].astype(str).tolist()
            if not reference_quarters:
                reference_quarters = quarters
            positions = np.arange(len(sample))
            ax.plot(
                positions, pd.to_numeric(sample["lift"], errors="coerce"),
                color=VARIANT_COLORS[scope],
                linestyle=VARIANT_LINESTYLES[selector],
                marker=VARIANT_MARKERS[selector], markersize=4,
                linewidth=1.7, alpha=0.9,
                label=f"{scope} | {selector}",
            )
        ax.set_xticks(
            np.arange(len(reference_quarters)), reference_quarters,
            rotation=35, ha="right",
        )
        ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
        ax.set_title(f"RUB/{currency}", loc="left", fontweight="bold")
        ax.set_ylabel("Квартальный lift")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    for ax in flat_axes[len(currencies):]:
        ax.remove()
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Holdout: квартальный lift — все варианты",
        fontsize=16, fontweight="bold", y=1.045,
    )
    return fig


def plot_source_contribution(signals: pd.DataFrame):
    _style()
    if signals.empty:
        raise ValueError("Нет holdout-сигналов")
    table = signals.groupby(
        ["currency", "winning_engine_type"], sort=True
    ).size().unstack(fill_value=0)
    colors = {"rule+ml": RED, "ml": BLACK}
    ax = table.plot(
        kind="bar", figsize=(10, 5),
        color=[colors.get(column, GRAY) for column in table.columns],
    )
    ax.set(
        title="Состав evidence в конечных сигналах",
        xlabel="Валюта", ylabel="Число сигналов",
    )
    ax.legend(title="Evidence", frameon=False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    plt.xticks(rotation=0)
    ax.figure.tight_layout()
    return ax.figure
