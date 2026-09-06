"""Causal client-delivery replay across Russian time zones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .moex_live import MOEX_TIMEZONE, evaluate_signal_relevance


RUSSIAN_TIME_ZONES: tuple[tuple[str, str, int], ...] = (
    ("MSK", "Europe/Moscow", 0),
    ("SAMT", "Europe/Samara", 1),
    ("YEKT", "Asia/Yekaterinburg", 2),
    ("OMST", "Asia/Omsk", 3),
    ("KRAT", "Asia/Krasnoyarsk", 4),
    ("IRKT", "Asia/Irkutsk", 5),
    ("YAKT", "Asia/Yakutsk", 6),
    ("VLAT", "Asia/Vladivostok", 7),
    ("MAGT", "Asia/Magadan", 8),
    ("PETT", "Asia/Kamchatka", 9),
)


@dataclass(frozen=True)
class ClientSimulationConfig:
    total_clients: int = 10_000
    average_transfer_rub: float = 20_000.0
    participation_rate: float = 1.0
    signal_hour_msk: int = 9
    quiet_start_hour: int = 21
    quiet_end_hour: int = 9
    max_market_price_age: str = "2h"
    timezone_shares: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.total_clients <= 0:
            raise ValueError("total_clients должен быть положительным")
        if self.average_transfer_rub <= 0:
            raise ValueError("average_transfer_rub должен быть положительным")
        if not 0.0 <= self.participation_rate <= 1.0:
            raise ValueError("participation_rate должен лежать в [0, 1]")
        for value in (
            self.signal_hour_msk, self.quiet_start_hour, self.quiet_end_hour
        ):
            if not 0 <= value <= 23:
                raise ValueError("Часы должны лежать в [0, 23]")
        if self.quiet_start_hour == self.quiet_end_hour:
            raise ValueError("Недопустимое окно не может занимать ноль часов")
        if pd.Timedelta(self.max_market_price_age) <= pd.Timedelta(0):
            raise ValueError("max_market_price_age должен быть положительным")


@dataclass
class ClientSimulationResult:
    config: ClientSimulationConfig
    client_allocation: pd.DataFrame
    delivery_details: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class ClientScenarioSweepResult:
    """Alternative delivery-time scenarios evaluated on the same signal stream."""

    signal_hours_msk: tuple[int, ...]
    scenario_results: dict[int, ClientSimulationResult]
    delivery_details: pd.DataFrame
    scenario_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame


def allocate_clients_by_timezone(
    total_clients: int = 10_000,
    shares: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Allocate an integer client population with the largest-remainder rule."""
    zone_rows = pd.DataFrame(
        RUSSIAN_TIME_ZONES,
        columns=["timezone", "iana_timezone", "utc_offset_from_msk"],
    )
    known = set(zone_rows["timezone"])
    if shares is None:
        weights = pd.Series(1.0, index=zone_rows["timezone"])
    else:
        unknown = set(shares).difference(known)
        if unknown:
            raise ValueError(f"Неизвестные часовые зоны: {sorted(unknown)}")
        weights = zone_rows["timezone"].map(
            lambda zone: float(shares.get(zone, 0.0))
        )
        weights.index = zone_rows["timezone"]
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Доли часовых зон должны быть конечными, неотрицательными")
    normalized = weights / weights.sum()
    exact = normalized * int(total_clients)
    counts = np.floor(exact).astype(int)
    remainder = int(total_clients) - int(counts.sum())
    if remainder:
        order = (exact - counts).sort_values(ascending=False, kind="stable").index
        counts.loc[order[:remainder]] += 1
    zone_rows["configured_share"] = zone_rows["timezone"].map(normalized)
    zone_rows["client_count"] = zone_rows["timezone"].map(counts).astype(int)
    zone_rows["realized_share"] = zone_rows["client_count"] / int(total_clients)
    return zone_rows


def _to_moscow(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(MOEX_TIMEZONE)
    return timestamp.tz_convert(MOEX_TIMEZONE)


def _is_quiet(hour: int, start: int, end: int) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def client_delivery_time(
    generated_at: object,
    iana_timezone: str,
    *,
    quiet_start_hour: int = 21,
    quiet_end_hour: int = 9,
) -> pd.Timestamp:
    """Return the first allowed delivery time, expressed in Moscow time."""
    generated_msk = _to_moscow(generated_at)
    local_zone = ZoneInfo(iana_timezone)
    local = generated_msk.tz_convert(local_zone)
    if not _is_quiet(local.hour, quiet_start_hour, quiet_end_hour):
        return generated_msk
    wake_date = local.date()
    if local.hour >= quiet_start_hour:
        wake_date += timedelta(days=1)
    wake_local = pd.Timestamp(
        datetime.combine(wake_date, time(hour=quiet_end_hour)), tz=local_zone
    )
    return wake_local.tz_convert(MOEX_TIMEZONE)


def build_client_delivery_schedule(
    signals: pd.DataFrame,
    *,
    config: ClientSimulationConfig = ClientSimulationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one aggregate delivery row per signal and time zone."""
    required = {"event_id", "available_at", "currency", "horizon"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"В signals отсутствуют поля: {sorted(missing)}")
    allocation = allocate_clients_by_timezone(
        config.total_clients, config.timezone_shares
    )
    events = signals.copy()
    event_dates = pd.to_datetime(events["available_at"]).dt.normalize()
    events["generated_at"] = event_dates.map(_to_moscow) + pd.Timedelta(
        hours=config.signal_hour_msk
    )
    events["_cross_key"] = 1
    zones = allocation.copy()
    zones["_cross_key"] = 1
    schedule = events.merge(zones, on="_cross_key", how="inner").drop(
        columns="_cross_key"
    )
    schedule["delivery_at"] = [
        client_delivery_time(
            generated,
            zone,
            quiet_start_hour=config.quiet_start_hour,
            quiet_end_hour=config.quiet_end_hour,
        )
        for generated, zone in zip(
            schedule["generated_at"], schedule["iana_timezone"], strict=True
        )
    ]
    schedule["delivery_delay_hours"] = (
        pd.to_datetime(schedule["delivery_at"], utc=True)
        - pd.to_datetime(schedule["generated_at"], utc=True)
    ).dt.total_seconds() / 3600.0
    return schedule, allocation


def _asof_market_price(
    requests: pd.DataFrame,
    hourly_prices: pd.DataFrame,
    *,
    request_time: str,
    suffix: str,
) -> pd.DataFrame:
    prices = hourly_prices.copy()
    prices["available_at"] = pd.to_datetime(prices["available_at"], utc=True)
    request_rows = requests.copy()
    request_rows[request_time] = pd.to_datetime(request_rows[request_time], utc=True)
    market_columns = [
        "currency", "available_at", "secid", "board", "quote_at", "fetched_at",
        "buy_price", "buy_price_source", "buy_price_is_executable",
        "sell_price", "sell_price_source", "sell_price_is_executable",
    ]
    output: list[pd.DataFrame] = []
    for currency, left in request_rows.groupby("currency", sort=False):
        right = prices.loc[prices["currency"].eq(currency), market_columns].copy()
        left = left.sort_values(request_time)
        if right.empty:
            joined = left.copy()
            for column in market_columns[2:]:
                joined[f"{column}_{suffix}"] = np.nan
            joined[f"available_at_{suffix}"] = pd.NaT
        else:
            rename = {
                column: f"{column}_{suffix}" for column in market_columns[1:]
            }
            right = right.drop(columns="currency")
            joined = pd.merge_asof(
                left,
                right.rename(columns=rename).sort_values(f"available_at_{suffix}"),
                left_on=request_time,
                right_on=f"available_at_{suffix}",
                direction="backward",
                allow_exact_matches=True,
            )
        output.append(joined)
    return pd.concat(output, ignore_index=True).sort_values("_schedule_order")


def _attach_backtest_outcomes(
    signals: pd.DataFrame, evaluation_rows: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "event_id", "target_value", "benefit_bps",
        "_stratum_random_precision", "_stratum_random_benefit_bps",
    }
    missing = required.difference(evaluation_rows.columns)
    if missing:
        raise ValueError(f"В evaluation_rows отсутствуют поля: {sorted(missing)}")
    outcomes = evaluation_rows.loc[
        evaluation_rows["event_id"].notna(), list(required)
    ].drop_duplicates("event_id")
    result = signals.merge(
        outcomes, on="event_id", how="inner", validate="one_to_one"
    )
    if result.empty:
        raise ValueError("Нет созревших отправленных сигналов для симуляции")
    return result


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid]))


def _summarize(details: pd.DataFrame, allocation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    client_map = allocation.set_index("timezone")["client_count"]
    for (zone, currency), sample in details.groupby(
        ["timezone", "currency"], sort=True
    ):
        accepted = sample.loc[sample["is_relevant"]].copy()
        weights = accepted["expected_client_transactions"]
        precision = _weighted_mean(accepted["target_value"], weights)
        random_precision = _weighted_mean(
            accepted["_stratum_random_precision"], weights
        )
        net_savings = float(accepted["net_client_savings_rub"].sum())
        positive_savings = float(
            accepted["positive_client_savings_rub"].sum()
        )
        mean_benefit_bps = _weighted_mean(
            accepted["realized_benefit_bps"], weights
        )
        rows.append({
            "timezone": zone,
            "currency": currency,
            "client_count": int(client_map.loc[zone]),
            "signals_considered": int(len(sample)),
            "signals_delayed": int(sample["delivery_delay_hours"].gt(0).sum()),
            "mean_delivery_delay_hours": float(
                sample["delivery_delay_hours"].mean()
            ),
            "max_delivery_delay_hours": float(
                sample["delivery_delay_hours"].max()
            ),
            "signals_accepted": int(len(accepted)),
            "signals_rejected": int(len(sample) - len(accepted)),
            "rejected_expired": int(
                sample["relevance_status"].eq("EXPIRED").sum()
            ),
            "rejected_stale_quote": int(
                sample["relevance_status"].eq("STALE_QUOTE").sum()
            ),
            "rejected_no_market_data": int(
                sample["relevance_status"].eq("NO_MARKET_DATA").sum()
            ),
            "rejected_consumed": int(
                sample["relevance_status"].eq("OPPORTUNITY_CONSUMED").sum()
            ),
            "signal_acceptance_rate": float(sample["is_relevant"].mean()),
            "expected_client_transactions": float(weights.sum()),
            "precision": precision,
            "random_precision": random_precision,
            "lift": (
                precision / random_precision
                if pd.notna(precision) and random_precision > 0 else np.nan
            ),
            "mean_realized_benefit_bps": mean_benefit_bps,
            "mean_realized_benefit_pct": (
                mean_benefit_bps / 100.0
                if pd.notna(mean_benefit_bps) else np.nan
            ),
            "positive_benefit_rate": _weighted_mean(
                accepted["realized_benefit_bps"].gt(0), weights
            ),
            "net_client_savings_rub": net_savings,
            "positive_client_savings_rub": positive_savings,
            "net_savings_per_client_rub": (
                net_savings / int(client_map.loc[zone])
                if int(client_map.loc[zone]) else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["timezone", "currency"]
    ).reset_index(drop=True)


def simulate_client_timezones(
    signals: pd.DataFrame,
    hourly_moex_prices: pd.DataFrame,
    *,
    evaluation_rows: pd.DataFrame,
    config: ClientSimulationConfig = ClientSimulationConfig(),
) -> ClientSimulationResult:
    """Replay signal delivery and relevance without expanding individual users.

    Each client is assumed to have one transfer of ``average_transfer_rub`` when
    an accepted signal arrives. ``participation_rate`` can scale that assumption
    later without changing delivery or relevance logic.
    """
    enriched = _attach_backtest_outcomes(signals, evaluation_rows)
    schedule, allocation = build_client_delivery_schedule(enriched, config=config)
    schedule = schedule.reset_index(drop=True)
    schedule["_schedule_order"] = np.arange(len(schedule))

    reference_requests = schedule.copy()
    reference_requests["reference_at"] = reference_requests["generated_at"]
    priced = _asof_market_price(
        reference_requests, hourly_moex_prices,
        request_time="reference_at", suffix="reference",
    )
    priced = _asof_market_price(
        priced, hourly_moex_prices,
        request_time="delivery_at", suffix="delivery",
    )

    decisions: list[dict] = []
    for row in priced.to_dict(orient="records"):
        expires_at = row["generated_at"] + pd.Timedelta(
            days=int(row["horizon"])
        )
        # The frozen pipeline already approved an immediate delivery. MOEX is
        # consulted only when the product quiet-hours rule delayed the client.
        if float(row["delivery_delay_hours"]) <= 0.0:
            decisions.append({
                "status": "DIRECT_DELIVERY",
                "is_relevant": True,
                "adverse_market_move_bps": 0.0,
                "remaining_expected_bps": float(row["expected_bps"]),
            })
            continue
        if _to_moscow(row["delivery_at"]) >= _to_moscow(expires_at):
            decisions.append({
                "status": "EXPIRED",
                "is_relevant": False,
                "adverse_market_move_bps": np.nan,
                "remaining_expected_bps": np.nan,
            })
            continue
        reference_price = pd.to_numeric(
            row.get("buy_price_reference"), errors="coerce"
        )
        delivery_price = pd.to_numeric(
            row.get("buy_price_delivery"), errors="coerce"
        )
        if (
            pd.isna(reference_price) or reference_price <= 0
            or pd.isna(delivery_price) or delivery_price <= 0
        ):
            decisions.append({
                "status": "NO_MARKET_DATA",
                "is_relevant": False,
                "adverse_market_move_bps": np.nan,
                "remaining_expected_bps": np.nan,
            })
            continue
        signal = row.copy()
        signal.update({
            "side": "BUY_FOREIGN",
            "generated_at": row["generated_at"],
            "issued_at": row["generated_at"],
            "expires_at": expires_at,
            "market_reference_secid": row.get("secid_reference"),
            "market_reference_price": row.get("buy_price_reference"),
        })
        current_quote = {
            "currency": row["currency"],
            "secid": row.get("secid_delivery"),
            "quote_at": row.get("quote_at_delivery"),
            "fetched_at": row.get("fetched_at_delivery"),
            "buy_price": row.get("buy_price_delivery"),
            "buy_price_source": row.get("buy_price_source_delivery"),
            "buy_price_is_executable": row.get(
                "buy_price_is_executable_delivery", False
            ),
            "sell_price": row.get("sell_price_delivery"),
            "sell_price_source": row.get("sell_price_source_delivery"),
            "sell_price_is_executable": row.get(
                "sell_price_is_executable_delivery", False
            ),
        }
        decision = evaluate_signal_relevance(
            signal,
            current_quote,
            checked_at=row["delivery_at"],
            max_quote_age=config.max_market_price_age,
            require_executable=False,
        )
        decisions.append(decision)

    priced["relevance_status"] = [item["status"] for item in decisions]
    priced["is_relevant"] = [item["is_relevant"] for item in decisions]
    priced["adverse_market_move_bps"] = [
        item["adverse_market_move_bps"] for item in decisions
    ]
    priced["remaining_expected_bps"] = [
        item["remaining_expected_bps"] for item in decisions
    ]
    priced["realized_benefit_bps"] = (
        pd.to_numeric(priced["benefit_bps"], errors="coerce")
        - pd.to_numeric(priced["adverse_market_move_bps"], errors="coerce")
    )
    priced["expected_client_transactions"] = (
        priced["client_count"] * config.participation_rate
        * priced["is_relevant"].astype(float)
    )
    priced["net_client_savings_rub"] = (
        priced["expected_client_transactions"]
        * config.average_transfer_rub
        * priced["realized_benefit_bps"] / 10_000.0
    )
    priced["positive_client_savings_rub"] = priced[
        "net_client_savings_rub"
    ].clip(lower=0.0)
    details = priced.sort_values("_schedule_order").drop(
        columns=["_schedule_order"], errors="ignore"
    ).reset_index(drop=True)
    return ClientSimulationResult(
        config=config,
        client_allocation=allocation,
        delivery_details=details,
        summary=_summarize(details, allocation),
    )


def aggregate_client_summary(
    summary: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Aggregate simulation rows while preserving metric denominators.

    Quality and benefit metrics are weighted by expected executed client
    transactions. Lift is always recomputed from aggregated precision rather
    than averaged across rows.
    """
    groups = list(group_columns)
    missing_groups = set(groups).difference(summary.columns)
    if missing_groups:
        raise ValueError(f"Нет колонок группировки: {sorted(missing_groups)}")
    required = {
        "client_count", "signals_considered", "signals_delayed",
        "signals_accepted", "signals_rejected", "expected_client_transactions",
        "precision", "random_precision", "mean_realized_benefit_bps",
        "positive_benefit_rate", "net_client_savings_rub",
        "positive_client_savings_rub",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"В summary отсутствуют поля: {sorted(missing)}")

    additive = [
        "signals_considered", "signals_delayed", "signals_accepted",
        "signals_rejected", "rejected_expired", "rejected_stale_quote",
        "rejected_no_market_data", "rejected_consumed",
        "expected_client_transactions", "net_client_savings_rub",
        "positive_client_savings_rub",
    ]
    rows: list[dict] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for key, sample in summary.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(groups) == 1 else tuple(key)
        row = dict(zip(groups, keys, strict=True))
        for column in additive:
            if column in sample:
                row[column] = float(pd.to_numeric(sample[column], errors="coerce").sum())
        row["client_count"] = int(sample["client_count"].max())
        weights = pd.to_numeric(
            sample["expected_client_transactions"], errors="coerce"
        ).fillna(0.0)
        for column in (
            "precision", "random_precision", "mean_realized_benefit_bps",
            "positive_benefit_rate",
        ):
            row[column] = _weighted_mean(
                pd.to_numeric(sample[column], errors="coerce"), weights
            )
        row["lift"] = (
            row["precision"] / row["random_precision"]
            if pd.notna(row["precision"]) and row["random_precision"] > 0
            else np.nan
        )
        row["mean_realized_benefit_pct"] = (
            row["mean_realized_benefit_bps"] / 100.0
            if pd.notna(row["mean_realized_benefit_bps"]) else np.nan
        )
        considered = row.get("signals_considered", 0.0)
        row["signal_acceptance_rate"] = (
            row.get("signals_accepted", 0.0) / considered
            if considered > 0 else np.nan
        )
        clients = row["client_count"]
        row["net_savings_per_client_rub"] = (
            row["net_client_savings_rub"] / clients if clients else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def simulate_client_signal_hours(
    signals: pd.DataFrame,
    hourly_moex_prices: pd.DataFrame,
    *,
    evaluation_rows: pd.DataFrame,
    signal_hours_msk: tuple[int, ...] = (9, 12, 15, 18),
    config: ClientSimulationConfig = ClientSimulationConfig(),
) -> ClientScenarioSweepResult:
    """Evaluate mutually exclusive signal-release hours on identical inputs.

    The aggregate result is an average policy scenario, not a claim that every
    signal is delivered four times. Additive counts and money are therefore
    divided by the number of release-hour scenarios; rates are pooled using
    their proper transaction denominators.
    """
    hours = tuple(int(hour) for hour in signal_hours_msk)
    if not hours or len(hours) != len(set(hours)):
        raise ValueError("signal_hours_msk должен содержать уникальные часы")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError("Часы выпуска должны лежать в [0, 23]")

    results: dict[int, ClientSimulationResult] = {}
    detail_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for hour in hours:
        scenario_config = replace(config, signal_hour_msk=hour)
        result = simulate_client_timezones(
            signals,
            hourly_moex_prices,
            evaluation_rows=evaluation_rows,
            config=scenario_config,
        )
        results[hour] = result
        details = result.delivery_details.copy()
        details.insert(0, "signal_hour_msk", hour)
        detail_parts.append(details)
        summary = result.summary.copy()
        summary.insert(0, "signal_hour_msk", hour)
        summary_parts.append(summary)

    all_details = pd.concat(detail_parts, ignore_index=True)
    scenario_summary = pd.concat(summary_parts, ignore_index=True)
    allocation = next(iter(results.values())).client_allocation
    aggregate = _summarize(all_details, allocation)

    # These are four alternative release policies, so report the expected
    # per-policy count/money rather than an artificial fourfold total.
    scenario_count = float(len(hours))
    averaged_columns = [
        "signals_considered", "signals_delayed", "signals_accepted",
        "signals_rejected", "rejected_expired", "rejected_stale_quote",
        "rejected_no_market_data", "rejected_consumed",
        "expected_client_transactions", "net_client_savings_rub",
        "positive_client_savings_rub", "net_savings_per_client_rub",
    ]
    for column in averaged_columns:
        aggregate[column] = aggregate[column] / scenario_count

    return ClientScenarioSweepResult(
        signal_hours_msk=hours,
        scenario_results=results,
        delivery_details=all_details,
        scenario_summary=scenario_summary,
        aggregate_summary=aggregate,
    )
