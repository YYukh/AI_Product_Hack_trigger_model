"""Presentation-oriented plots for the final signal outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


PALETTE = {
    "ink": "#171717",
    "dark_gray": "#4A4A4A",
    "gray": "#8A8A8A",
    "light_gray": "#D9D9D9",
    "red": "#C62828",
    "light_red": "#EF9A9A",
}


def plot_rate_panel(
    market_data: pd.DataFrame,
    *,
    currencies: tuple[str, ...] = ("AMD", "KGS", "KZT", "TJS", "UZS"),
    start_date: str | pd.Timestamp = "2025-01-01",
) -> plt.Figure:
    """Plot post-holdout rate dynamics for all target currencies."""
    frame = market_data.loc[
        pd.to_datetime(market_data["available_at"]).ge(pd.Timestamp(start_date))
        & market_data["currency"].isin(currencies)
    ].copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    figure, axes = plt.subplots(len(currencies), 1, figsize=(14, 2.5 * len(currencies)), sharex=True)
    axes = [axes] if len(currencies) == 1 else axes
    for ax, currency in zip(axes, currencies):
        subset = frame.loc[frame["currency"].eq(currency)]
        ax.plot(subset["available_at"], subset["rate"], color=PALETTE["ink"], linewidth=1.1)
        _style(ax, currency)
        ax.set_ylabel("RUB")
    axes[-1].set_xlabel("Дата")
    figure.tight_layout()
    return figure


def _style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", color=PALETTE["ink"], weight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(PALETTE["gray"])
    ax.spines["bottom"].set_color(PALETTE["gray"])
    ax.tick_params(colors=PALETTE["dark_gray"])


def plot_signal_series(
    market_data: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    currency: str,
    target_family: str | None = None,
    title: str | None = None,
    start_date: str | pd.Timestamp = "2025-01-01",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot rate with G0/W1 signal markers for one currency."""
    required = {"available_at", "currency", "rate"}
    missing = required.difference(market_data.columns)
    if missing:
        raise KeyError(f"В market_data нет полей: {sorted(missing)}")
    frame = market_data.loc[
        market_data["currency"].eq(currency)
        & pd.to_datetime(market_data["available_at"]).ge(pd.Timestamp(start_date))
    ].copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    ax = ax or plt.subplots(figsize=(14, 5))[1]
    ax.plot(frame["available_at"], frame["rate"], color=PALETTE["ink"], linewidth=1.2)
    subset = signals.loc[signals["currency"].eq(currency)].copy()
    subset = subset.loc[
        pd.to_datetime(subset["available_at"]).ge(pd.Timestamp(start_date))
    ]
    if target_family is not None and "target_family" in subset:
        subset = subset.loc[subset["target_family"].eq(target_family)]
    subset["available_at"] = pd.to_datetime(subset["available_at"])
    for family, color, marker in (
        ("G0", PALETTE["red"], "o"),
        ("W1", PALETTE["dark_gray"], "^"),
    ):
        points = subset.loc[subset.get("target_family", pd.Series(index=subset.index)).eq(family)]
        if not points.empty:
            points = points.merge(frame[["available_at", "rate"]], on="available_at", how="left")
            ax.scatter(points["available_at"], points["rate"], color=color, marker=marker,
                       s=42, label=family, zorder=3, edgecolors="white", linewidths=0.5)
    _style(ax, title or f"{currency}: курс и финальные сигналы")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Курс, RUB")
    ax.legend(frameon=False, ncol=2)
    return ax


def plot_oos_uplift(
    summary: pd.DataFrame,
    *,
    uplift_column: str = "test_lift",
    title: str = "OOS uplift по конфигурациям",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot configuration-level uplift without comparing currencies in one scale."""
    required = {"currency", "target_family", "horizon", uplift_column}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"В summary нет полей: {sorted(missing)}")
    data = summary.sort_values(["currency", "target_family", "horizon"]).copy()
    data["label"] = data.apply(
        lambda row: f"{row.currency} · {row.target_family} · h{int(row.horizon)}",
        axis=1,
    )
    ax = ax or plt.subplots(figsize=(15, 7))[1]
    colors = [PALETTE["red"] if value >= 1.3 else PALETTE["gray"] for value in data[uplift_column]]
    ax.bar(range(len(data)), data[uplift_column], color=colors)
    ax.axhline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=1, label="Случайный уровень")
    ax.axhline(1.3, color=PALETTE["light_red"], linestyle=":", linewidth=1, label="Целевой uplift 1.3")
    ax.set_xticks(range(len(data)), data["label"], rotation=70, ha="right")
    ax.set_ylabel("Uplift")
    _style(ax, title)
    ax.legend(frameon=False)
    return ax


def plot_benefit_distribution(
    signals: pd.DataFrame,
    *,
    title: str = "Распределение BPS-эффекта сигналов",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the distribution of realised signal benefit in basis points."""
    if "benefit_bps" not in signals.columns:
        raise KeyError("В signals нет поля benefit_bps")
    data = pd.to_numeric(signals["benefit_bps"], errors="coerce").dropna()
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    if len(data):
        ax.hist(data, bins=30, color=PALETTE["gray"], edgecolor=PALETTE["ink"], alpha=0.9)
        ax.axvline(data.mean(), color=PALETTE["red"], linewidth=2, label=f"Среднее: {data.mean():.1f} bps")
        ax.axvline(0, color=PALETTE["ink"], linestyle="--", linewidth=1)
    ax.set_xlabel("Эффект, bps")
    ax.set_ylabel("Количество сигналов")
    _style(ax, title)
    if len(data):
        ax.legend(frameon=False)
    return ax


def plot_fast_slow_summary(summary: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot fast/slow precision and BPS effect for one compact comparison."""
    required = {"currency", "target_family", "engine_type", "fast_precision", "slow_precision"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"В fast_slow_summary нет полей: {sorted(missing)}")
    data = summary.copy()
    data["label"] = data["currency"] + " · " + data["target_family"] + " · " + data["engine_type"]
    ax = ax or plt.subplots(figsize=(14, 7))[1]
    positions = range(len(data))
    ax.bar([p - 0.18 for p in positions], data["fast_precision"], width=0.36,
           color=PALETTE["light_red"], label="Fast (h=1/3)")
    ax.bar([p + 0.18 for p in positions], data["slow_precision"], width=0.36,
           color=PALETTE["dark_gray"], label="Slow (h=10/20)")
    ax.set_xticks(list(positions), data["label"], rotation=65, ha="right")
    ax.set_ylabel("Precision")
    _style(ax, "Fast и slow: точность OOS-сигналов")
    ax.legend(frameon=False)
    return ax
