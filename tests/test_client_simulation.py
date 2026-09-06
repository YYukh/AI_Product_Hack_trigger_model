import pandas as pd

from src.client_simulation import (
    ClientSimulationConfig,
    allocate_clients_by_timezone,
    build_client_delivery_schedule,
    client_delivery_time,
    simulate_client_signal_hours,
    simulate_client_timezones,
)


def test_equal_allocation_is_integer_and_complete():
    allocation = allocate_clients_by_timezone(10_000)
    assert allocation["client_count"].sum() == 10_000
    assert allocation["client_count"].max() - allocation["client_count"].min() <= 1


def test_night_window_delays_only_until_local_nine():
    generated = pd.Timestamp("2026-01-10 15:00", tz="Europe/Moscow")
    msk = client_delivery_time(generated, "Europe/Moscow")
    yakt = client_delivery_time(generated, "Asia/Yakutsk")
    assert msk == generated
    assert yakt == generated + pd.Timedelta(hours=12)


def test_compact_simulation_uses_weights_not_client_expansion():
    signal = pd.DataFrame([{
        "event_id": "event-1", "available_at": pd.Timestamp("2026-01-10"),
        "currency": "KZT", "scenario": "GOOD_NOW", "target_family": "G0",
        "target": "target-g0", "horizon": 3, "expected_bps": 100.0,
        "confidence": 0.7,
    }])
    evaluation = pd.DataFrame([{
        "event_id": "event-1", "target_value": True, "benefit_bps": 80.0,
        "_stratum_random_precision": 0.2,
        "_stratum_random_benefit_bps": 5.0,
    }])
    times = pd.to_datetime([
        "2026-01-10 09:00:00+03:00", "2026-01-10 10:00:00+03:00",
    ], utc=True).tz_convert("Europe/Moscow")
    prices = pd.DataFrame({
        "currency": ["KZT", "KZT"], "secid": ["KZTRUB_TOM"] * 2,
        "board": ["CETS"] * 2, "candle_begin": times - pd.Timedelta(hours=1),
        "available_at": times, "open": [100.0, 100.0], "high": [100.0, 100.5],
        "low": [100.0, 100.0], "close": [100.0, 100.5], "volume": [1, 1],
        "value": [100, 100], "quote_at": times,
        "buy_price": [100.0, 100.5], "buy_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "buy_price_is_executable": [False, False], "sell_price": [100.0, 100.5],
        "sell_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "sell_price_is_executable": [False, False], "fetched_at": times,
        "source": ["MOEX_ISS_CANDLES"] * 2,
    })
    result = simulate_client_timezones(
        signal, prices, evaluation_rows=evaluation,
        config=ClientSimulationConfig(total_clients=110),
    )
    assert len(result.delivery_details) == 10
    assert result.client_allocation["client_count"].sum() == 110
    assert set(result.summary["currency"]) == {"KZT"}
    assert result.summary["expected_client_transactions"].sum() == 110


def test_nine_oclock_direct_issue_does_not_require_a_fresh_moex_candle():
    signal = pd.DataFrame([{
        "event_id": "event-1", "available_at": pd.Timestamp("2026-01-10"),
        "currency": "KZT", "scenario": "GOOD_NOW", "target_family": "G0",
        "target": "target-g0", "horizon": 3, "expected_bps": 100.0,
        "confidence": 0.7,
    }])
    evaluation = pd.DataFrame([{
        "event_id": "event-1", "target_value": True, "benefit_bps": 80.0,
        "_stratum_random_precision": 0.2,
        "_stratum_random_benefit_bps": 5.0,
    }])
    old_time = pd.to_datetime(
        ["2026-01-09 18:00:00+03:00"], utc=True
    ).tz_convert("Europe/Moscow")
    prices = pd.DataFrame({
        "currency": ["KZT"], "secid": ["KZTRUB_TOM"], "board": ["CETS"],
        "available_at": old_time, "quote_at": old_time, "fetched_at": old_time,
        "buy_price": [100.0], "buy_price_source": ["HOURLY_CLOSE_PROXY"],
        "buy_price_is_executable": [False], "sell_price": [100.0],
        "sell_price_source": ["HOURLY_CLOSE_PROXY"],
        "sell_price_is_executable": [False],
    })
    result = simulate_client_timezones(
        signal, prices, evaluation_rows=evaluation,
        config=ClientSimulationConfig(total_clients=100, signal_hour_msk=9),
    )
    details = result.delivery_details
    assert len(details) == 10
    assert details["provider_is_relevant"].all()
    assert set(details["provider_status"]) == {"DIRECT_ISSUE"}
    assert details["is_relevant"].all()

    # The same stale market state blocks a later send on the provider side,
    # before any client-time-zone delivery takes place.
    later = simulate_client_timezones(
        signal, prices, evaluation_rows=evaluation,
        config=ClientSimulationConfig(total_clients=100, signal_hour_msk=15),
    ).delivery_details
    assert not later["provider_is_relevant"].any()
    assert set(later["provider_status"]) == {"STALE_QUOTE"}
    assert set(later["relevance_status"]) == {"NOT_SENT_STALE_QUOTE"}


def test_signal_hour_sweep_keeps_scenarios_separate_and_averages_counts():
    signal = pd.DataFrame([{
        "event_id": "event-1", "available_at": pd.Timestamp("2026-01-10"),
        "currency": "KZT", "scenario": "GOOD_NOW", "target_family": "G0",
        "target": "target-g0", "horizon": 3, "expected_bps": 100.0,
        "confidence": 0.7,
    }])
    evaluation = pd.DataFrame([{
        "event_id": "event-1", "target_value": True, "benefit_bps": 80.0,
        "_stratum_random_precision": 0.2,
        "_stratum_random_benefit_bps": 5.0,
    }])
    quote_times = pd.to_datetime([
        "2026-01-10 09:00:00+03:00", "2026-01-10 12:00:00+03:00",
    ], utc=True).tz_convert("Europe/Moscow")
    prices = pd.DataFrame({
        "currency": ["KZT"] * 2, "secid": ["KZTRUB_TOM"] * 2,
        "board": ["CETS"] * 2, "available_at": quote_times,
        "quote_at": quote_times, "fetched_at": quote_times,
        "buy_price": [100.0, 100.0],
        "buy_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "buy_price_is_executable": [False] * 2,
        "sell_price": [100.0, 100.0],
        "sell_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "sell_price_is_executable": [False] * 2,
    })
    sweep = simulate_client_signal_hours(
        signal, prices, evaluation_rows=evaluation,
        signal_hours_msk=(9, 12),
        config=ClientSimulationConfig(total_clients=100),
    )
    assert set(sweep.scenario_results) == {9, 12}
    assert set(sweep.scenario_summary["signal_hour_msk"]) == {9, 12}
    assert len(sweep.delivery_details) == 20
    # An aggregate row represents one average release-time policy, not two sends.
    assert sweep.aggregate_summary["signals_considered"].eq(1.0).all()
    assert "mean_realized_benefit_pct" in sweep.aggregate_summary


def test_each_corridor_client_gets_at_most_one_assignment_per_half_month():
    dates = pd.to_datetime([
        "2026-01-02", "2026-01-05", "2026-01-10",
        "2026-01-16", "2026-01-20", "2026-01-27",
    ])
    signals = pd.DataFrame({
        "event_id": [f"event-{i}" for i in range(len(dates))],
        "available_at": dates, "currency": "KZT", "scenario": "GOOD_NOW",
        "target_family": "G0", "target": "target-g0", "horizon": 3,
        "expected_bps": 100.0, "confidence": 0.7,
    })
    evaluation = pd.DataFrame({
        "event_id": signals["event_id"], "target_value": True,
        "benefit_bps": 80.0, "_stratum_random_precision": 0.2,
        "_stratum_random_benefit_bps": 5.0,
    })
    quote_time = pd.to_datetime(
        ["2026-01-01 18:00:00+03:00"], utc=True
    ).tz_convert("Europe/Moscow")
    prices = pd.DataFrame({
        "currency": ["KZT"], "secid": ["KZTRUB_TOM"], "board": ["CETS"],
        "available_at": quote_time, "quote_at": quote_time,
        "fetched_at": quote_time, "buy_price": [100.0],
        "buy_price_source": ["HOURLY_CLOSE_PROXY"],
        "buy_price_is_executable": [False], "sell_price": [100.0],
        "sell_price_source": ["HOURLY_CLOSE_PROXY"],
        "sell_price_is_executable": [False],
    })
    result = simulate_client_timezones(
        signals, prices, evaluation_rows=evaluation,
        config=ClientSimulationConfig(total_clients=100, signal_hour_msk=9),
    )
    assigned = result.delivery_details.groupby(
        ["timezone", "month_half"], sort=True
    )["scheduled_client_transactions"].sum()
    assert assigned.eq(10).all()
    assert result.summary["potential_client_transactions"].sum() == 200


def test_later_send_is_checked_by_provider_then_delayed_client_again():
    signal = pd.DataFrame([{
        "event_id": "event-1", "available_at": pd.Timestamp("2026-01-10"),
        "currency": "KZT", "scenario": "GOOD_NOW", "target_family": "G0",
        "target": "target-g0", "horizon": 3, "expected_bps": 100.0,
        "confidence": 0.7,
    }])
    evaluation = pd.DataFrame([{
        "event_id": "event-1", "target_value": True, "benefit_bps": 80.0,
        "_stratum_random_precision": 0.2,
        "_stratum_random_benefit_bps": 5.0,
    }])
    quote_times = pd.to_datetime([
        "2026-01-10 09:00:00+03:00", "2026-01-10 15:00:00+03:00",
    ], utc=True).tz_convert("Europe/Moscow")
    prices = pd.DataFrame({
        "currency": ["KZT"] * 2, "secid": ["KZTRUB_TOM"] * 2,
        "board": ["CETS"] * 2, "available_at": quote_times,
        "quote_at": quote_times, "fetched_at": quote_times,
        "buy_price": [100.0, 100.2],
        "buy_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "buy_price_is_executable": [False] * 2, "sell_price": [100.0, 100.2],
        "sell_price_source": ["HOURLY_CLOSE_PROXY"] * 2,
        "sell_price_is_executable": [False] * 2,
    })
    details = simulate_client_timezones(
        signal, prices, evaluation_rows=evaluation,
        config=ClientSimulationConfig(total_clients=100, signal_hour_msk=15),
    ).delivery_details
    assert details["provider_is_relevant"].all()
    immediate = details["delivery_delay_hours"].eq(0)
    assert details.loc[immediate, "is_relevant"].all()
    assert not details.loc[~immediate, "is_relevant"].any()
    assert set(details.loc[~immediate, "relevance_status"]) == {"STALE_QUOTE"}
