"""Presentation-ready diagnostics for the alternative pipeline."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RED = "#d32f2f"
DARK_RED = "#8b0000"
BLACK = "#161616"
GRAY = "#777777"
LIGHT_GRAY = "#e5e5e5"


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

    fig.suptitle(
        "Holdout: накопительная и текущая устойчивость после каждого сигнала",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
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
