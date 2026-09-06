"""Presentation-oriented plots for the final signal outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#171717",
    "dark_gray": "#4A4A4A",
    "gray": "#8A8A8A",
    "light_gray": "#D9D9D9",
    "red": "#C62828",
    "light_red": "#EF9A9A",
}

REPORT_CMAP = LinearSegmentedColormap.from_list(
    "product_hack_red_gray",
    (PALETTE["light_gray"], "#F4F4F4", PALETTE["light_red"], PALETTE["red"]),
)

SPEED_ORDER = ("fast_1_3d", "middle_5d", "slow_10_20d")
SPEED_LABELS = {
    "fast_1_3d": "Быстрые: 1–3 дня",
    "middle_5d": "Средние: 5 дней",
    "slow_10_20d": "Медленные: 10–20 дней",
}
SPEED_COLORS = {
    "fast_1_3d": PALETTE["red"],
    "middle_5d": PALETTE["gray"],
    "slow_10_20d": PALETTE["ink"],
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
    for family, family_label, color, marker in (
        ("G0", "Выгодно сейчас", PALETTE["red"], "o"),
        ("W1", "Окно закрывается", PALETTE["dark_gray"], "^"),
    ):
        points = subset.loc[subset.get("target_family", pd.Series(index=subset.index)).eq(family)]
        if not points.empty:
            points = points.merge(frame[["available_at", "rate"]], on="available_at", how="left")
            ax.scatter(points["available_at"], points["rate"], color=color, marker=marker,
                       s=42, label=family_label, zorder=3,
                       edgecolors="white", linewidths=0.5)
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


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"В {name} нет полей: {sorted(missing)}")


def _signal_rows(backtest_rows: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        backtest_rows,
        {"available_at", "currency", "horizon", "target_family", "signal", "benefit_bps"},
        "backtest_rows",
    )
    result = backtest_rows.loc[backtest_rows["signal"].astype(bool)].copy()
    result["available_at"] = pd.to_datetime(result["available_at"])
    result["benefit_bps"] = pd.to_numeric(result["benefit_bps"], errors="coerce")
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise").astype(int)
    result["speed"] = np.select(
        [result["horizon"].isin((1, 3)), result["horizon"].isin((10, 20))],
        ["fast_1_3d", "slow_10_20d"],
        default="middle_5d",
    )
    return result


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    value_format: str,
    colorbar_label: str,
) -> None:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=REPORT_CMAP, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), labels=[f"h={value}" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                "—" if not np.isfinite(value) else value_format.format(value),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=PALETTE["ink"],
            )
    ax.set_title(title, loc="left", color=PALETTE["ink"], weight="bold")
    ax.set_xlabel("Горизонт")
    ax.set_ylabel("Валюта")
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(colorbar_label)


def plot_policy_currency_overview(
    summary: pd.DataFrame,
    *,
    policy_name: str = "Текущая политика",
) -> plt.Figure:
    """Final holdout dashboard for one selected policy, split by currency."""
    required = {
        "scope", "currency", "lift", "signals_per_week", "precision",
        "mean_benefit_bps", "benefit_uplift_bps", "positive_benefit_rate",
    }
    _require_columns(summary, required, "summary")
    data = summary.loc[summary["scope"].eq("currency")].sort_values("currency")
    if data.empty:
        raise ValueError("В summary нет строк scope='currency'")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    panels = (
        ("lift", "Lift относительно случайного входа", "Lift", 1.0),
        ("signals_per_week", "Частота итоговых сигналов", "Сигналов в неделю", None),
        ("benefit_uplift_bps", "Дополнительный эффект относительно случайного входа", "Benefit uplift, BPS", 0.0),
        ("positive_benefit_rate", "Доля сигналов с положительным BPS", "Доля", 0.5),
    )
    for ax, (column, title, ylabel, reference) in zip(axes.flat, panels, strict=True):
        bars = ax.bar(data["currency"], data[column], color=PALETTE["red"], width=0.65)
        ax.bar_label(bars, fmt="%.2f" if column != "benefit_uplift_bps" else "%.1f", padding=3)
        if reference is not None:
            ax.axhline(reference, color=PALETTE["ink"], linestyle="--", linewidth=1)
        if column == "signals_per_week":
            ax.axhspan(1.0, 2.0, color=PALETTE["light_gray"], alpha=0.55)
        _style(ax, title)
        ax.set_ylabel(ylabel)
    figure.suptitle(f"Финальный holdout · {policy_name}", fontsize=16, weight="bold")
    return figure


def plot_policy_horizon_heatmaps(
    summary: pd.DataFrame,
    *,
    target_families: tuple[str, ...] = ("G0", "W1"),
) -> plt.Figure:
    """Heatmaps of lift and economic uplift for one policy."""
    required = {
        "scope", "currency", "horizon", "target_family", "lift", "benefit_uplift_bps",
    }
    _require_columns(summary, required, "summary")
    exact = summary.loc[
        summary["scope"].eq("currency+horizon+target_family")
        & summary["target_family"].isin(target_families)
    ].copy()
    combined = summary.loc[summary["scope"].eq("currency+horizon")].copy()
    if exact.empty or combined.empty:
        raise ValueError(
            "Для heatmap нужны строки scope='currency+horizon' и "
            "scope='currency+horizon+target_family'"
        )
    exact["horizon"] = pd.to_numeric(exact["horizon"], errors="raise").astype(int)
    combined["horizon"] = pd.to_numeric(combined["horizon"], errors="raise").astype(int)
    currencies = sorted(combined["currency"].unique())
    horizons = sorted(combined["horizon"].unique())
    panels = (("Все таргеты", combined), *(
        (family, exact.loc[exact["target_family"].eq(family)])
        for family in target_families
    ))
    figure, axes = plt.subplots(
        len(panels), 2,
        figsize=(17, 4.8 * len(panels)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, (panel_name, panel_data) in enumerate(panels):
        lift = panel_data.pivot(index="currency", columns="horizon", values="lift").reindex(
            index=currencies, columns=horizons
        )
        bps = panel_data.pivot(
            index="currency", columns="horizon", values="benefit_uplift_bps"
        ).reindex(index=currencies, columns=horizons)
        _annotated_heatmap(
            axes[row_index, 0], lift,
            title=f"{panel_name}: lift", value_format="{:.2f}", colorbar_label="Lift",
        )
        _annotated_heatmap(
            axes[row_index, 1], bps,
            title=f"{panel_name}: benefit uplift", value_format="{:.1f}", colorbar_label="BPS",
        )
    figure.suptitle("Качество политики по валютам, таргетам и горизонтам", fontsize=16, weight="bold")
    return figure


def plot_policy_speed_analysis(backtest_rows: pd.DataFrame) -> plt.Figure:
    """Show quality and contribution of fast, middle and slow final signals."""
    signals = _signal_rows(backtest_rows).dropna(subset=["benefit_bps"])
    currencies = sorted(signals["currency"].unique())
    grouped = (
        signals.groupby(["currency", "speed"], as_index=False)
        .agg(
            signal_count=("benefit_bps", "size"),
            mean_benefit_bps=("benefit_bps", "mean"),
            total_benefit_bps=("benefit_bps", "sum"),
            positive_benefit_rate=("benefit_bps", lambda values: float((values > 0).mean())),
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    positions = np.arange(len(currencies))
    width = 0.24
    for index, speed in enumerate(SPEED_ORDER):
        data = grouped.loc[grouped["speed"].eq(speed)].set_index("currency").reindex(currencies)
        offset = (index - 1) * width
        axes[0, 0].bar(
            positions + offset, data["mean_benefit_bps"], width,
            color=SPEED_COLORS[speed], label=SPEED_LABELS[speed],
        )
        axes[0, 1].bar(
            positions + offset, data["positive_benefit_rate"], width,
            color=SPEED_COLORS[speed], label=SPEED_LABELS[speed],
        )
    for ax, title, ylabel in (
        (axes[0, 0], "Средний BPS на сигнал", "BPS"),
        (axes[0, 1], "Доля сигналов с положительным эффектом", "Доля"),
    ):
        ax.set_xticks(positions, labels=currencies)
        ax.axhline(0 if ylabel == "BPS" else 0.5, color=PALETTE["gray"], linestyle="--", linewidth=1)
        _style(ax, title)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)

    count_matrix = grouped.pivot(index="currency", columns="speed", values="signal_count").reindex(
        index=currencies, columns=SPEED_ORDER
    ).fillna(0)
    effect_matrix = grouped.pivot(
        index="currency", columns="speed", values="total_benefit_bps"
    ).reindex(index=currencies, columns=SPEED_ORDER).fillna(0)
    count_matrix.rename(columns=SPEED_LABELS).plot(
        kind="bar", stacked=True, ax=axes[1, 0],
        color=[SPEED_COLORS[speed] for speed in SPEED_ORDER],
    )
    effect_matrix.rename(columns=SPEED_LABELS).plot(
        kind="bar", stacked=True, ax=axes[1, 1],
        color=[SPEED_COLORS[speed] for speed in SPEED_ORDER],
    )
    _style(axes[1, 0], "Состав потока по скорости сигналов")
    axes[1, 0].set_ylabel("Количество сигналов")
    max_signal_count = float(count_matrix.sum(axis=1).max())
    axes[1, 0].set_ylim(0, max(1.0, max_signal_count * 1.18))
    axes[1, 0].set_axisbelow(True)
    _style(axes[1, 1], "Вклад скорости сигналов в суммарный BPS")
    axes[1, 1].set_ylabel("Суммарный benefit, BPS")
    for ax in axes[1]:
        ax.set_xlabel("Валюта")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(frameon=False, fontsize=8)
    figure.suptitle("Быстрые и медленные сигналы выбранной политики", fontsize=16, weight="bold")
    return figure


def plot_policy_cumulative_bps(backtest_rows: pd.DataFrame) -> plt.Figure:
    """Plot cumulative realised opportunity BPS for the selected policy."""
    signals = _signal_rows(backtest_rows).dropna(subset=["benefit_bps"])
    daily = (
        signals.groupby(["available_at", "currency"], as_index=False)["benefit_bps"].sum()
        .sort_values("available_at")
    )
    figure, ax = plt.subplots(figsize=(15, 6))
    for currency, data in daily.groupby("currency", sort=True):
        ax.plot(
            data["available_at"], data["benefit_bps"].cumsum(),
            linewidth=1.25, color=PALETTE["gray"], alpha=0.65, label=currency,
        )
    overall = daily.groupby("available_at", as_index=False)["benefit_bps"].sum().sort_values("available_at")
    ax.plot(
        overall["available_at"], overall["benefit_bps"].cumsum(),
        linewidth=2.5, color=PALETTE["red"], label="Все валюты",
    )
    ax.axhline(0, color=PALETTE["ink"], linewidth=1)
    _style(ax, "Накопительный opportunity BPS финальных сигналов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Накопленный benefit, BPS")
    ax.legend(frameon=False, ncol=3)
    figure.tight_layout()
    return figure


def _quarterly_metrics(backtest_rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "available_at", "currency", "signal", "target_value", "benefit_bps",
        "_stratum_random_precision", "_stratum_random_benefit_bps",
    }
    _require_columns(backtest_rows, required, "backtest_rows")
    rows = backtest_rows.copy()
    rows["available_at"] = pd.to_datetime(rows["available_at"])
    rows["quarter"] = rows["available_at"].dt.to_period("Q").astype(str)
    output = []
    for (currency, quarter), sample in rows.groupby(["currency", "quarter"], sort=True):
        selected = sample.loc[sample["signal"].astype(bool)]
        signal_count = len(selected)
        true_positive = int(selected["target_value"].astype(bool).sum())
        precision = true_positive / signal_count if signal_count else np.nan
        baseline = (
            float(selected["_stratum_random_precision"].mean()) if signal_count else np.nan
        )
        benefit = pd.to_numeric(selected["benefit_bps"], errors="coerce").dropna()
        random_benefit = pd.to_numeric(
            selected["_stratum_random_benefit_bps"], errors="coerce"
        ).dropna()
        mean_benefit_bps = float(benefit.mean()) if len(benefit) else np.nan
        random_mean_benefit_bps = (
            float(random_benefit.mean()) if len(random_benefit) else np.nan
        )
        output.append({
            "currency": currency,
            "quarter": quarter,
            "signal_count": signal_count,
            "precision": precision,
            "random_precision": baseline,
            "lift": precision / baseline if baseline and baseline > 0 else np.nan,
            "mean_benefit_bps": mean_benefit_bps,
            "random_mean_benefit_bps": random_mean_benefit_bps,
            "benefit_uplift_bps": (
                mean_benefit_bps - random_mean_benefit_bps
                if pd.notna(mean_benefit_bps) and pd.notna(random_mean_benefit_bps)
                else np.nan
            ),
        })
    return pd.DataFrame(output)


def plot_policy_quarterly_stability(backtest_rows: pd.DataFrame) -> plt.Figure:
    """Plot time stability of matched-baseline lift for one policy."""
    quarterly = _quarterly_metrics(backtest_rows)
    currencies = sorted(quarterly["currency"].unique())
    figure, axes = plt.subplots(3, 2, figsize=(17, 13), sharex=True)
    axes = list(axes.flat)
    for ax, currency in zip(axes, currencies):
        data = quarterly.loc[quarterly["currency"].eq(currency)].sort_values("quarter")
        ax.plot(
            data["quarter"], data["lift"], marker="o",
            color=PALETTE["red"], linewidth=1.8,
        )
        ax.axhline(1, color=PALETTE["ink"], linestyle="--", linewidth=1)
        _style(ax, f"RUB/{currency}")
        ax.set_ylabel("Квартальный lift")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[len(currencies):]:
        ax.axis("off")
    figure.suptitle("Устойчивость lift по кварталам и валютам", fontsize=16, weight="bold")
    figure.tight_layout()
    return figure


def plot_policy_quarterly_benefit(backtest_rows: pd.DataFrame) -> plt.Figure:
    """Plot quarterly mean signal BPS against the matched random baseline."""
    quarterly = _quarterly_metrics(backtest_rows)
    currencies = sorted(quarterly["currency"].unique())
    figure, axes = plt.subplots(3, 2, figsize=(17, 13), sharex=True)
    axes = list(axes.flat)
    for ax, currency in zip(axes, currencies):
        data = quarterly.loc[quarterly["currency"].eq(currency)].sort_values("quarter")
        ax.plot(
            data["quarter"], data["mean_benefit_bps"], marker="o",
            color=PALETTE["red"], linewidth=1.8, label="Финальные сигналы",
        )
        ax.plot(
            data["quarter"], data["random_mean_benefit_bps"],
            color=PALETTE["dark_gray"], linestyle="--", linewidth=1.3,
            label="Случайный вход того же mix",
        )
        ax.axhline(0, color=PALETTE["ink"], linestyle=":", linewidth=1)
        _style(ax, f"RUB/{currency}")
        ax.set_ylabel("Средний benefit, BPS")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[len(currencies):]:
        ax.axis("off")
    figure.suptitle(
        "Выгода сигналов по кварталам и валютам",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout()
    return figure


def plot_policy_quarterly_signal_count(backtest_rows: pd.DataFrame) -> plt.Figure:
    """Plot the actual number of final signals in every currency-quarter."""
    quarterly = _quarterly_metrics(backtest_rows)
    currencies = sorted(quarterly["currency"].unique())
    figure, axes = plt.subplots(3, 2, figsize=(17, 13), sharex=True)
    axes = list(axes.flat)
    for ax, currency in zip(axes, currencies):
        data = quarterly.loc[quarterly["currency"].eq(currency)].sort_values("quarter")
        bars = ax.bar(
            data["quarter"], data["signal_count"],
            color=PALETTE["red"], width=0.62,
        )
        ax.bar_label(bars, fmt="%d", padding=3, color=PALETTE["ink"])
        _style(ax, f"RUB/{currency}")
        ax.set_ylabel("Количество сигналов")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0)
    for ax in axes[len(currencies):]:
        ax.axis("off")
    figure.suptitle(
        "Количество финальных сигналов по кварталам и валютам",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout()
    return figure


def _block_bootstrap_summary(
    backtest_rows: pd.DataFrame,
    *,
    repeats: int = 1000,
    block_weeks: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    required = {
        "available_at", "currency", "signal", "target_value", "benefit_bps",
        "_stratum_random_precision",
    }
    _require_columns(backtest_rows, required, "backtest_rows")
    rows = backtest_rows.copy()
    rows["available_at"] = pd.to_datetime(rows["available_at"])
    rows["week"] = rows["available_at"].dt.to_period("W-SUN").dt.start_time
    rows["_signal"] = rows["signal"].astype(bool).astype(int)
    rows["_tp"] = (rows["signal"].astype(bool) & rows["target_value"].astype(bool)).astype(int)
    rows["_expected_tp"] = rows["_signal"] * pd.to_numeric(
        rows["_stratum_random_precision"], errors="coerce"
    )
    benefit = pd.to_numeric(rows["benefit_bps"], errors="coerce")
    valid_benefit = rows["signal"].astype(bool) & benefit.notna()
    rows["_benefit_sum"] = np.where(valid_benefit, benefit, 0.0)
    rows["_benefit_count"] = valid_benefit.astype(int)
    stat_columns = ["_signal", "_tp", "_expected_tp", "_benefit_sum", "_benefit_count"]

    by_currency = rows.groupby(["currency", "week"], as_index=False)[stat_columns].sum()
    overall = rows.groupby("week", as_index=False)[stat_columns].sum().assign(currency="ALL")
    weekly = pd.concat([by_currency, overall], ignore_index=True)
    weeks = np.array(sorted(rows["week"].unique()))
    week_count = len(weeks)
    if not week_count:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(week_count / block_weeks))
    weights = np.zeros((repeats, week_count), dtype=int)
    for repeat in range(repeats):
        starts = rng.integers(0, week_count, size=block_count)
        positions = np.concatenate([
            (np.arange(start, start + block_weeks) % week_count) for start in starts
        ])[:week_count]
        weights[repeat] = np.bincount(positions, minlength=week_count)

    output = []
    for currency, sample in weekly.groupby("currency", sort=True):
        matrix = sample.set_index("week")[stat_columns].reindex(weeks, fill_value=0).to_numpy(float)
        sampled = weights @ matrix
        signals, tp, expected_tp, benefit_sum, benefit_count = sampled.T
        lift = np.divide(tp, expected_tp, out=np.full(repeats, np.nan), where=expected_tp > 0)
        mean_bps = np.divide(
            benefit_sum, benefit_count, out=np.full(repeats, np.nan), where=benefit_count > 0
        )
        point = matrix.sum(axis=0)
        output.append({
            "currency": currency,
            "lift": point[1] / point[2] if point[2] > 0 else np.nan,
            "lift_ci_low": float(np.nanquantile(lift, 0.025)),
            "lift_ci_high": float(np.nanquantile(lift, 0.975)),
            "mean_benefit_bps": point[3] / point[4] if point[4] > 0 else np.nan,
            "mean_bps_ci_low": float(np.nanquantile(mean_bps, 0.025)),
            "mean_bps_ci_high": float(np.nanquantile(mean_bps, 0.975)),
        })
    return pd.DataFrame(output)


def plot_policy_bootstrap_intervals(
    backtest_rows: pd.DataFrame,
    *,
    repeats: int = 1000,
    block_weeks: int = 4,
    seed: int = 42,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot matched-lift and BPS block-bootstrap intervals for one policy."""
    summary = _block_bootstrap_summary(
        backtest_rows, repeats=repeats, block_weeks=block_weeks, seed=seed
    )
    order = ["ALL", *sorted(summary.loc[summary["currency"].ne("ALL"), "currency"])]
    data = summary.set_index("currency").reindex(order)
    labels = ["Все" if value == "ALL" else value for value in data.index]
    positions = np.arange(len(data))
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    lift_error = np.vstack([
        data["lift"] - data["lift_ci_low"], data["lift_ci_high"] - data["lift"],
    ])
    axes[0].errorbar(
        positions, data["lift"], yerr=lift_error, fmt="o", capsize=6,
        color=PALETTE["red"], ecolor=PALETTE["dark_gray"], markersize=8,
    )
    axes[0].axhline(1, color=PALETTE["ink"], linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, labels=labels)
    axes[0].set_ylabel("Lift")
    _style(axes[0], "Lift и 95% block-bootstrap интервал")

    bps_error = np.vstack([
        data["mean_benefit_bps"] - data["mean_bps_ci_low"],
        data["mean_bps_ci_high"] - data["mean_benefit_bps"],
    ])
    axes[1].errorbar(
        positions, data["mean_benefit_bps"], yerr=bps_error, fmt="o", capsize=6,
        color=PALETTE["red"], ecolor=PALETTE["dark_gray"], markersize=8,
    )
    axes[1].axhline(0, color=PALETTE["ink"], linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, labels=labels)
    axes[1].set_ylabel("Средний benefit, BPS")
    _style(axes[1], "Средний BPS и 95% block-bootstrap интервал")
    figure.suptitle(
        f"Статистическая устойчивость · {repeats} итераций · блок {block_weeks} недели",
        fontsize=15,
        weight="bold",
    )
    return figure, summary.reset_index(drop=True)
