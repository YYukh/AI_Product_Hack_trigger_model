"""Live MOEX quotes and causal signal-relevance checks.

The module deliberately keeps market freshness separate from model confidence.
Confidence describes the engine at signal time; relevance describes whether the
same economic opportunity is still executable now.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MOEX_TIMEZONE = "Europe/Moscow"
MOEX_ISS_URL = (
    "https://iss.moex.com/iss/engines/currency/markets/selt/"
    "securities/{secid}.json"
)


@dataclass(frozen=True)
class MoexInstrument:
    """MOEX instrument used to execute RUB -> recipient-currency conversion."""

    currency: str
    secid: str
    board: str = "CETS"


MOEX_INSTRUMENTS: dict[str, MoexInstrument] = {
    currency: MoexInstrument(currency, f"{currency}RUB_TOM")
    for currency in ("AMD", "KGS", "KZT", "TJS", "UZS")
}


def _number(value: Any) -> float:
    converted = pd.to_numeric(value, errors="coerce")
    return float(converted) if pd.notna(converted) else np.nan


def _positive(value: Any) -> float:
    converted = _number(value)
    return converted if np.isfinite(converted) and converted > 0 else np.nan


def _moscow_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(MOEX_TIMEZONE)
    return timestamp.tz_convert(MOEX_TIMEZONE)


def _quote_timestamp(row: Mapping[str, Any], fetched_at: pd.Timestamp) -> pd.Timestamp:
    trade_date = row.get("TRADEDATE")
    update_time = row.get("UPDATETIME") or row.get("SYSTIME")
    if trade_date and update_time:
        parsed = pd.to_datetime(f"{trade_date} {update_time}", errors="coerce")
    else:
        parsed = pd.to_datetime(update_time, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    parsed = pd.Timestamp(parsed)
    if parsed.tzinfo is None:
        return parsed.tz_localize(MOEX_TIMEZONE)
    return parsed.tz_convert(MOEX_TIMEZONE)


def _preferred_row(
    rows: Sequence[Mapping[str, Any]], instrument: MoexInstrument
) -> Mapping[str, Any]:
    matching = [
        row for row in rows
        if str(row.get("SECID", "")) == instrument.secid
        and str(row.get("BOARDID", "")) == instrument.board
    ]
    if not matching:
        matching = [
            row for row in rows
            if str(row.get("SECID", "")) == instrument.secid
        ]
    if not matching:
        raise ValueError(
            f"MOEX ISS не вернул {instrument.secid} на доске {instrument.board}"
        )
    return matching[0]


def _execution_price(row: Mapping[str, Any], side: str) -> tuple[float, str, bool]:
    """Choose an executable quote, retaining an explicit indicative fallback."""
    if side == "BUY_FOREIGN":
        executable = _positive(row.get("OFFER"))
        executable_name = "OFFER"
    elif side == "SELL_FOREIGN":
        executable = _positive(row.get("BID"))
        executable_name = "BID"
    else:
        raise ValueError("side должен быть BUY_FOREIGN или SELL_FOREIGN")
    if np.isfinite(executable):
        return executable, executable_name, True
    for column in ("MARKETPRICE", "LAST", "WAPRICE"):
        fallback = _positive(row.get(column))
        if np.isfinite(fallback):
            return fallback, column, False
    return np.nan, "UNAVAILABLE", False


def _parse_quote(
    payload: Mapping[str, Any],
    instrument: MoexInstrument,
    fetched_at: pd.Timestamp,
) -> dict[str, Any]:
    market_rows = payload.get("marketdata", [])
    security_rows = payload.get("securities", [])
    market = _preferred_row(market_rows, instrument)
    try:
        security = _preferred_row(security_rows, instrument)
    except ValueError:
        security = {}
    buy_price, buy_source, buy_executable = _execution_price(
        market, "BUY_FOREIGN"
    )
    sell_price, sell_source, sell_executable = _execution_price(
        market, "SELL_FOREIGN"
    )
    return {
        "currency": instrument.currency,
        "secid": instrument.secid,
        "board": str(market.get("BOARDID") or instrument.board),
        "short_name": security.get("SHORTNAME"),
        "lot_size": _number(security.get("LOTSIZE")),
        "bid": _positive(market.get("BID")),
        "offer": _positive(market.get("OFFER")),
        "last": _positive(market.get("LAST")),
        "market_price": _positive(market.get("MARKETPRICE")),
        "waprice": _positive(market.get("WAPRICE")),
        "buy_price": buy_price,
        "buy_price_source": buy_source,
        "buy_price_is_executable": buy_executable,
        "sell_price": sell_price,
        "sell_price_source": sell_source,
        "sell_price_is_executable": sell_executable,
        "number_of_trades": _number(market.get("NUMTRADES")),
        "trading_status": market.get("TRADINGSTATUS"),
        "quote_at": _quote_timestamp(market, fetched_at),
        "fetched_at": fetched_at,
        "source": "MOEX_ISS",
    }


async def load_current_moex_quotes(
    currencies: Sequence[str] = tuple(MOEX_INSTRUMENTS),
) -> pd.DataFrame:
    """Load one point-in-time MOEX FX snapshot through ``aiomoex``.

    The function is async by design and can be called directly with ``await``
    in Jupyter. An OFFER/BID is marked executable. MARKETPRICE/LAST/WAPRICE are
    retained only as indicative fallbacks and never silently treated as orders.
    """
    try:
        import aiohttp
        import aiomoex
    except ImportError as error:
        raise ImportError(
            "Для live MOEX установите зависимости: pip install aiomoex aiohttp"
        ) from error

    requested = [str(currency).upper() for currency in currencies]
    unknown = sorted(set(requested).difference(MOEX_INSTRUMENTS))
    if unknown:
        raise ValueError(f"Нет MOEX mapping для валют: {unknown}")

    async with aiohttp.ClientSession() as session:
        requests = []
        for currency in requested:
            instrument = MOEX_INSTRUMENTS[currency]
            arguments = {
                "iss.only": "securities,marketdata",
                "securities.columns": (
                    "SECID,BOARDID,SHORTNAME,LOTSIZE,DECIMALS"
                ),
                "marketdata.columns": (
                    "SECID,BOARDID,BID,OFFER,LAST,MARKETPRICE,WAPRICE,"
                    "NUMTRADES,TRADINGSTATUS,TRADEDATE,UPDATETIME,SYSTIME"
                ),
            }
            client = aiomoex.ISSClient(
                session,
                MOEX_ISS_URL.format(secid=instrument.secid),
                arguments,
            )
            requests.append(client.get())

        import asyncio

        payloads = await asyncio.gather(*requests)

    fetched_at = pd.Timestamp.now(tz=MOEX_TIMEZONE)
    quotes = [
        _parse_quote(payload, MOEX_INSTRUMENTS[currency], fetched_at)
        for currency, payload in zip(requested, payloads, strict=True)
    ]
    return pd.DataFrame(quotes).sort_values("currency").reset_index(drop=True)


async def load_historical_moex_hourly(
    currencies: Sequence[str] = tuple(MOEX_INSTRUMENTS),
    *,
    start: str | pd.Timestamp = "2025-01-01",
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load causal one-hour CETS candles for a future client replay.

    MOEX candles are trades, not historical order-book snapshots. Consequently
    ``close`` is exposed as an explicitly non-executable simulation proxy. Its
    ``quote_at`` is the candle end, so a replay cannot consume the close before
    the complete hour has elapsed.
    """
    try:
        import aiohttp
        import aiomoex
    except ImportError as error:
        raise ImportError(
            "Для истории MOEX установите зависимости: pip install aiomoex aiohttp"
        ) from error

    requested = [str(currency).upper() for currency in currencies]
    unknown = sorted(set(requested).difference(MOEX_INSTRUMENTS))
    if unknown:
        raise ValueError(f"Нет MOEX mapping для валют: {unknown}")
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    if end_at < start_at:
        raise ValueError("end должен быть не раньше start")

    async with aiohttp.ClientSession() as session:
        import asyncio

        requests = [
            aiomoex.get_board_candles(
                session,
                MOEX_INSTRUMENTS[currency].secid,
                interval=60,
                start=start_at.strftime("%Y-%m-%d"),
                end=end_at.strftime("%Y-%m-%d"),
                board=MOEX_INSTRUMENTS[currency].board,
                market="selt",
                engine="currency",
            )
            for currency in requested
        ]
        payloads = await asyncio.gather(*requests)

    frames: list[pd.DataFrame] = []
    numeric_columns = ("open", "high", "low", "close", "value", "volume")
    for currency, records in zip(requested, payloads, strict=True):
        if not records:
            continue
        frame = pd.DataFrame(records)
        frame.columns = [str(column).lower() for column in frame.columns]
        if "begin" not in frame or "close" not in frame:
            raise ValueError(
                f"Неожиданный формат часовых свечей {currency}: {frame.columns.tolist()}"
            )
        candle_begin = pd.to_datetime(frame["begin"], errors="coerce")
        if candle_begin.isna().any():
            raise ValueError(f"MOEX вернул невалидный timestamp свечи для {currency}")
        if candle_begin.dt.tz is None:
            candle_begin = candle_begin.dt.tz_localize(MOEX_TIMEZONE)
        else:
            candle_begin = candle_begin.dt.tz_convert(MOEX_TIMEZONE)
        frame["candle_begin"] = candle_begin
        frame["available_at"] = candle_begin + pd.Timedelta(hours=1)
        for column in numeric_columns:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        instrument = MOEX_INSTRUMENTS[currency]
        frame["currency"] = currency
        frame["secid"] = instrument.secid
        frame["board"] = instrument.board
        frame["quote_at"] = frame["available_at"]
        frame["fetched_at"] = frame["available_at"]
        frame["buy_price"] = frame["close"]
        frame["buy_price_source"] = "HOURLY_CLOSE_PROXY"
        frame["buy_price_is_executable"] = False
        frame["sell_price"] = frame["close"]
        frame["sell_price_source"] = "HOURLY_CLOSE_PROXY"
        frame["sell_price_is_executable"] = False
        frame["source"] = "MOEX_ISS_CANDLES"
        frames.append(frame)

    if not frames:
        raise ValueError("MOEX ISS не вернул часовых свечей для заданного периода")
    columns = [
        "currency", "secid", "board", "candle_begin", "available_at",
        "open", "high", "low", "close", "volume", "value", "quote_at",
        "buy_price", "buy_price_source", "buy_price_is_executable",
        "sell_price", "sell_price_source", "sell_price_is_executable",
        "fetched_at", "source",
    ]
    return (
        pd.concat(frames, ignore_index=True)[columns]
        .sort_values(["available_at", "currency"])
        .reset_index(drop=True)
    )


def stamp_signals_with_moex_reference(
    signals: pd.DataFrame,
    quotes_at_issue: pd.DataFrame,
    *,
    side: str = "BUY_FOREIGN",
    require_executable: bool = True,
    max_quote_age: str | pd.Timedelta = "30min",
) -> pd.DataFrame:
    """Attach immutable MOEX execution context when signals are emitted.

    This function belongs immediately after signal generation and before a
    signal is serialized/sent. It must not be applied retrospectively.
    """
    if signals.empty:
        return signals.copy()
    if quotes_at_issue["currency"].duplicated().any():
        raise ValueError("quotes_at_issue должен содержать одну строку на валюту")
    side = side.upper()
    if side not in {"BUY_FOREIGN", "SELL_FOREIGN"}:
        raise ValueError("side должен быть BUY_FOREIGN или SELL_FOREIGN")
    price_prefix = "buy" if side == "BUY_FOREIGN" else "sell"
    quote_columns = [
        "currency", "secid", "board", "quote_at", "fetched_at",
        f"{price_prefix}_price", f"{price_prefix}_price_source",
        f"{price_prefix}_price_is_executable",
    ]
    missing = set(quote_columns).difference(quotes_at_issue.columns)
    if missing:
        raise ValueError(f"В quotes_at_issue отсутствуют поля: {sorted(missing)}")

    result = signals.copy()
    result["_row_order"] = np.arange(len(result))
    result = result.merge(
        quotes_at_issue[quote_columns], on="currency", how="left",
        validate="many_to_one",
    ).sort_values("_row_order").drop(columns="_row_order")
    executable_column = f"{price_prefix}_price_is_executable"
    if require_executable and not result[executable_column].fillna(False).all():
        unavailable = sorted(
            result.loc[~result[executable_column].fillna(False), "currency"].unique()
        )
        raise ValueError(f"Нет исполнимой MOEX-котировки для: {unavailable}")
    if require_executable:
        quote_at = pd.to_datetime(result["quote_at"], utc=True, errors="coerce")
        fetched_at = pd.to_datetime(result["fetched_at"], utc=True, errors="coerce")
        fresh = quote_at.notna() & fetched_at.notna() & (
            fetched_at.sub(quote_at).between(pd.Timedelta(0), pd.Timedelta(max_quote_age))
        )
        if not fresh.all():
            stale = sorted(result.loc[~fresh, "currency"].unique())
            raise ValueError(f"MOEX-котировка устарела в момент выпуска: {stale}")

    generated_at = pd.to_datetime(
        result["as_of"] if "as_of" in result else result["available_at"]
    ).map(_moscow_timestamp)
    issued_at = pd.to_datetime(result["fetched_at"]).map(_moscow_timestamp)
    result["side"] = side
    result["market_reference_secid"] = result["secid"]
    result["market_reference_board"] = result["board"]
    result["market_reference_price"] = pd.to_numeric(
        result[f"{price_prefix}_price"], errors="coerce"
    )
    result["market_reference_price_source"] = result[
        f"{price_prefix}_price_source"
    ]
    result["market_reference_quote_at"] = pd.to_datetime(result["quote_at"])
    result["market_reference_fetched_at"] = pd.to_datetime(result["fetched_at"])
    result["generated_at"] = generated_at
    result["issued_at"] = issued_at
    result["expires_at"] = generated_at + pd.to_timedelta(
        result["horizon"], unit="D"
    )
    return result.drop(columns=[
        "secid", "board", "quote_at", "fetched_at",
        f"{price_prefix}_price", f"{price_prefix}_price_source",
        f"{price_prefix}_price_is_executable",
    ])


def check_signal_relevance(
    signals: pd.DataFrame,
    current_quotes: pd.DataFrame,
    *,
    as_of: Any | None = None,
    max_quote_age: str | pd.Timedelta = "30min",
    execution_buffer_bps: float = 0.0,
    require_executable: bool = True,
) -> pd.DataFrame:
    """Check whether issued signals retain positive expected economic value.

    For BUY_FOREIGN, an increase of the current OFFER is adverse. For
    SELL_FOREIGN, a decrease of the current BID is adverse. The original
    expected BPS is reduced by that signed adverse move. The model confidence
    remains unchanged because current price relevance is a separate concept.
    """
    required_signal = {
        "currency", "horizon", "expected_bps", "side", "expires_at",
        "market_reference_secid", "market_reference_price",
    }
    missing_signal = required_signal.difference(signals.columns)
    if missing_signal:
        raise ValueError(
            f"Сигналы не прошиты MOEX-контекстом: {sorted(missing_signal)}"
        )
    if current_quotes["currency"].duplicated().any():
        raise ValueError("current_quotes должен содержать одну строку на валюту")

    now = _moscow_timestamp(as_of or pd.Timestamp.now(tz=MOEX_TIMEZONE))
    age_limit = pd.Timedelta(max_quote_age)
    quote_columns = [
        "currency", "secid", "quote_at", "buy_price", "buy_price_source",
        "buy_price_is_executable", "sell_price", "sell_price_source",
        "sell_price_is_executable",
    ]
    missing_quote = set(quote_columns).difference(current_quotes.columns)
    if missing_quote:
        raise ValueError(f"В current_quotes отсутствуют поля: {sorted(missing_quote)}")

    result = signals.copy()
    result["_row_order"] = np.arange(len(result))
    result = result.merge(
        current_quotes[quote_columns], on="currency", how="left",
        validate="many_to_one",
    ).sort_values("_row_order").drop(columns="_row_order")

    output_rows: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        adverse_move_bps = np.nan
        side = str(row.get("side", "")).upper()
        price_prefix = "buy" if side == "BUY_FOREIGN" else "sell"
        current_price = _positive(row.get(f"{price_prefix}_price"))
        reference_price = _positive(row.get("market_reference_price"))
        expected_bps = _number(row.get("expected_bps"))
        quote_at_raw = row.get("quote_at")
        quote_at = (
            _moscow_timestamp(quote_at_raw)
            if pd.notna(quote_at_raw) else pd.NaT
        )
        expires_at = _moscow_timestamp(row["expires_at"])
        same_instrument = str(row.get("secid")) == str(
            row.get("market_reference_secid")
        )
        executable_raw = row.get(f"{price_prefix}_price_is_executable", False)
        executable = bool(pd.notna(executable_raw) and executable_raw)

        if not same_instrument:
            status = "INSTRUMENT_MISMATCH"
        elif now >= expires_at:
            status = "EXPIRED"
        elif not np.isfinite(current_price) or (require_executable and not executable):
            status = "NO_EXECUTABLE_QUOTE"
        elif pd.isna(quote_at) or now - quote_at > age_limit or quote_at > now:
            status = "STALE_QUOTE"
        elif not np.isfinite(reference_price):
            status = "NO_REFERENCE_PRICE"
        elif not np.isfinite(expected_bps):
            status = "NO_EXPECTED_BENEFIT"
        else:
            if side == "BUY_FOREIGN":
                adverse_move_bps = 10_000.0 * (
                    current_price / reference_price - 1.0
                )
            elif side == "SELL_FOREIGN":
                adverse_move_bps = 10_000.0 * (
                    1.0 - current_price / reference_price
                )
            else:
                adverse_move_bps = np.nan
                status = "UNKNOWN_SIDE"
            if np.isfinite(adverse_move_bps):
                remaining_bps = expected_bps - adverse_move_bps
                status = (
                    "ACTIVE" if remaining_bps > execution_buffer_bps
                    else "OPPORTUNITY_CONSUMED"
                )

        remaining_bps = (
            expected_bps - adverse_move_bps
            if np.isfinite(expected_bps) and np.isfinite(adverse_move_bps)
            else np.nan
        )
        row.update({
            "checked_at": now,
            "current_market_price": current_price,
            "current_market_price_source": row.get(f"{price_prefix}_price_source"),
            "current_market_price_is_executable": executable,
            "current_quote_at": quote_at,
            "adverse_market_move_bps": adverse_move_bps,
            "remaining_expected_bps": remaining_bps,
            "relevance_status": status,
            "is_relevant": status == "ACTIVE",
        })
        output_rows.append(row)

    output = pd.DataFrame(output_rows)
    return output.drop(
        columns=[column for column in quote_columns if column != "currency"],
        errors="ignore",
    )


def evaluate_signal_relevance(
    last_sent_signal: Mapping[str, Any] | pd.Series,
    current_moex_quote: Mapping[str, Any] | pd.Series,
    *,
    checked_at: Any | None = None,
    max_quote_age: str | pd.Timedelta = "30min",
    execution_buffer_bps: float = 0.0,
    require_executable: bool = True,
) -> dict[str, Any]:
    """Production contract: one last signal + one current quote -> decision.

    Both inputs are JSON-like records. ``current_moex_quote`` can be a row from
    :func:`load_current_moex_quotes`. For a historical replay it can be a row
    from :func:`load_historical_moex_hourly`, but that deliberate approximation
    requires ``require_executable=False``.
    """
    signal_record = dict(last_sent_signal)
    quote_record = dict(current_moex_quote)
    if str(signal_record.get("currency")) != str(quote_record.get("currency")):
        raise ValueError("Валюта текущей цены не совпадает с валютой сигнала")
    decision_at = (
        checked_at
        or quote_record.get("fetched_at")
        or quote_record.get("quote_at")
    )
    if decision_at is None or pd.isna(decision_at):
        raise ValueError("Нужен checked_at либо timestamp текущей MOEX-котировки")
    result = check_signal_relevance(
        pd.DataFrame([signal_record]),
        pd.DataFrame([quote_record]),
        as_of=decision_at,
        max_quote_age=max_quote_age,
        execution_buffer_bps=execution_buffer_bps,
        require_executable=require_executable,
    )
    decision = result.iloc[0].to_dict()
    return {
        "event_id": decision.get("event_id"),
        "currency": decision["currency"],
        "checked_at": decision["checked_at"],
        "expires_at": decision["expires_at"],
        "reference_price": decision["market_reference_price"],
        "current_price": decision["current_market_price"],
        "current_price_source": decision["current_market_price_source"],
        "price_is_executable": decision["current_market_price_is_executable"],
        "expected_bps_at_issue": decision["expected_bps"],
        "adverse_market_move_bps": decision["adverse_market_move_bps"],
        "remaining_expected_bps": decision["remaining_expected_bps"],
        "confidence": decision.get("confidence"),
        "status": decision["relevance_status"],
        "is_relevant": bool(decision["is_relevant"]),
    }
