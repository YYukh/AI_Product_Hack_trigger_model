import pandas as pd
import pytest

from src.moex_live import (
    check_signal_relevance,
    evaluate_signal_relevance,
    stamp_signals_with_moex_reference,
)


def _quote(*, offer: float, at: str = "2026-09-06 10:00:00") -> pd.DataFrame:
    timestamp = pd.Timestamp(at, tz="Europe/Moscow")
    return pd.DataFrame([{
        "currency": "KZT",
        "secid": "KZTRUB_TOM",
        "board": "CETS",
        "quote_at": timestamp,
        "fetched_at": timestamp,
        "buy_price": offer,
        "buy_price_source": "OFFER",
        "buy_price_is_executable": True,
        "sell_price": offer - 0.01,
        "sell_price_source": "BID",
        "sell_price_is_executable": True,
    }])


def _signal() -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": "event-1",
        "currency": "KZT",
        "available_at": pd.Timestamp("2026-09-06"),
        "as_of": pd.Timestamp("2026-09-06 09:00:00"),
        "horizon": 3,
        "expected_bps": 100.0,
        "confidence": 0.72,
    }])


def test_relevance_uses_remaining_bps_and_preserves_confidence():
    issued = stamp_signals_with_moex_reference(_signal(), _quote(offer=100.0))
    checked = check_signal_relevance(
        issued,
        _quote(offer=100.5, at="2026-09-06 10:15:00"),
        as_of="2026-09-06 10:20:00+03:00",
    ).iloc[0]

    assert checked["relevance_status"] == "ACTIVE"
    assert bool(checked["is_relevant"])
    assert checked["adverse_market_move_bps"] == pytest.approx(50.0)
    assert checked["remaining_expected_bps"] == pytest.approx(50.0)
    assert checked["confidence"] == pytest.approx(0.72)


def test_consumed_expired_and_stale_are_distinct_states():
    issued = stamp_signals_with_moex_reference(_signal(), _quote(offer=100.0))
    consumed = check_signal_relevance(
        issued, _quote(offer=101.1, at="2026-09-06 10:15:00"),
        as_of="2026-09-06 10:20:00+03:00",
    )
    expired = check_signal_relevance(
        issued, _quote(offer=100.0, at="2026-09-10 10:00:00"),
        as_of="2026-09-10 10:05:00+03:00",
    )
    stale = check_signal_relevance(
        issued, _quote(offer=100.0, at="2026-09-06 09:00:00"),
        as_of="2026-09-06 10:20:00+03:00",
    )

    assert consumed.loc[0, "relevance_status"] == "OPPORTUNITY_CONSUMED"
    assert expired.loc[0, "relevance_status"] == "EXPIRED"
    assert stale.loc[0, "relevance_status"] == "STALE_QUOTE"


def test_indicative_quote_cannot_be_stamped_as_executable():
    quote = _quote(offer=100.0)
    quote["buy_price_source"] = "LAST"
    quote["buy_price_is_executable"] = False
    with pytest.raises(ValueError, match="Нет исполнимой"):
        stamp_signals_with_moex_reference(_signal(), quote)


def test_single_record_production_contract():
    issued = stamp_signals_with_moex_reference(_signal(), _quote(offer=100.0))
    current = _quote(offer=100.5, at="2026-09-06 10:15:00")
    decision = evaluate_signal_relevance(
        issued.iloc[0], current.iloc[0],
        checked_at="2026-09-06 10:20:00+03:00",
    )

    assert decision["currency"] == "KZT"
    assert decision["status"] == "ACTIVE"
    assert decision["remaining_expected_bps"] == pytest.approx(50.0)
    assert decision["confidence"] == pytest.approx(0.72)
