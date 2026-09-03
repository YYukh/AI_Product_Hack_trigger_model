"""Загрузка официальных дневных курсов Банка России."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests


CBR_BASE_URL = "https://www.cbr.ru/scripts"

# Порядок словаря задаёт порядок столбцов итогового датафрейма.
CURRENCIES = {
    "AMD": "R01060",
    "KZT": "R01335",
    "KGS": "R01370",
    "TJS": "R01670",
    "UZS": "R01717",
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
}


def build_session() -> requests.Session:
    """Создать HTTP-сессию для запросов к ЦБ."""
    session = requests.Session()
    session.headers.update({"User-Agent": "ProductHack-CBR-Research/0.1"})
    return session


def get_xml(
    session: requests.Session,
    endpoint: str,
    params: dict[str, str],
    *,
    timeout: int = 30,
    retries: int = 3,
) -> bytes:
    """Получить валидный XML с повторными попытками."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(
                f"{CBR_BASE_URL}/{endpoint}", params=params, timeout=timeout
            )
            response.raise_for_status()
            ET.fromstring(response.content)
            return response.content
        except (requests.RequestException, ET.ParseError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Не удалось получить {endpoint} после {retries} попыток"
    ) from last_error


def parse_history_records(content: bytes, currency: str) -> pd.DataFrame:
    """Преобразовать XML валюты в temporal-записи без потери номинала."""
    root = ET.fromstring(content)
    observations = []

    for record in root.findall("Record"):
        effective_date = pd.to_datetime(record.attrib["Date"], dayfirst=True)
        nominal = int(record.findtext("Nominal"))
        raw_rate = float(
            record.findtext("Value").replace(" ", "").replace(",", ".")
        )
        observations.append(
            {
                "currency": currency,
                "effective_date": effective_date,
                "nominal": nominal,
                "raw_rate": raw_rate,
                "normalized_rate": raw_rate / nominal,
            }
        )

    if not observations:
        raise ValueError(f"ЦБ вернул пустую историю для {currency}")

    return (
        pd.DataFrame(observations)
        .sort_values("effective_date")
        .reset_index(drop=True)
    )


def parse_history(content: bytes, currency: str) -> pd.Series:
    """Получить нормированный Series-интерфейс."""
    records = parse_history_records(content, currency)
    return (
        records
        .set_index("effective_date")["normalized_rate"]
        .rename(currency)
        .astype("float64")
    )


def load_cbr_history(
    start_date: date,
    end_date: date,
    *,
    currencies: dict[str, str] | None = None,
    raw_dir: Path | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Загрузить long temporal-историю с полями исходного наблюдения.

    Исторический XML не содержит точного времени публикации. Поэтому
    ``publication_timestamp`` — явно маркированная консервативная proxy:
    00:00 следующего календарного дня, совпадающая с принятой в проекте
    ``available_at``. Это не выдаётся за фактический timestamp ЦБ.
    """
    currency_ids = currencies or CURRENCIES
    http = session or build_session()
    own_session = session is None
    frames = []
    ingestion_timestamp = pd.Timestamp.now(tz="UTC")

    try:
        for currency, cbr_id in currency_ids.items():
            params = {
                "date_req1": start_date.strftime("%d/%m/%Y"),
                "date_req2": end_date.strftime("%d/%m/%Y"),
                "VAL_NM_RQ": cbr_id,
            }
            content = get_xml(http, "XML_dynamic.asp", params)
            records = parse_history_records(content, currency)
            records["cbr_id"] = cbr_id
            frames.append(records)

            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{currency}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.xml"
                (raw_dir / filename).write_bytes(content)
    finally:
        if own_session:
            http.close()

    history = pd.concat(frames, ignore_index=True)
    history["available_at"] = history["effective_date"] + pd.Timedelta(days=1)
    history["publication_timestamp"] = history["available_at"]
    history["publication_timestamp_is_proxy"] = True
    history["ingestion_timestamp"] = ingestion_timestamp
    history["source"] = "CBR_XML_dynamic"
    history["is_update_day"] = True

    return (
        history[
            [
                "currency",
                "cbr_id",
                "effective_date",
                "publication_timestamp",
                "publication_timestamp_is_proxy",
                "available_at",
                "ingestion_timestamp",
                "source",
                "nominal",
                "raw_rate",
                "normalized_rate",
                "is_update_day",
            ]
        ]
        .sort_values(["available_at", "currency"])
        .reset_index(drop=True)
    )


def load_cbr_rates(
    start_date: date,
    end_date: date,
    *,
    currencies: dict[str, str] | None = None,
    raw_dir: Path | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Загрузить курсы ЦБ в широком формате.

    Строки — даты доступности в 00:00 следующего календарного дня.
    Столбцы — валюты. Значения — RUB за одну единицу валюты.
    """
    currency_ids = currencies or CURRENCIES
    history = load_cbr_history(
        start_date,
        end_date,
        currencies=currency_ids,
        raw_dir=raw_dir,
        session=session,
    )
    rates = history.pivot(
        index="available_at",
        columns="currency",
        values="normalized_rate",
    )
    rates = rates.reindex(columns=list(currency_ids)).sort_index()
    rates.columns.name = None
    rates.index.name = "available_at"
    return rates
