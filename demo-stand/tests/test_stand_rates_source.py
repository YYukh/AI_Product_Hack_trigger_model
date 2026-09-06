"""api/data_access.py — курсы из парсера с откатом на файл."""
import httpx
import pytest
import respx

from api import config
from api import data_access as da

PARSER = "http://parser.test"

WIDE_OK = {
    "source": "mldata_snapshot",
    "columns": ["AMD", "KZT", "KGS", "TJS", "UZS", "USD", "EUR", "CNY"],
    "rows": [
        {"date": "2026-08-12", "AMD": 0.24, "KZT": 0.19, "KGS": 0.99,
         "TJS": 8.9, "UZS": 0.0079, "USD": 90.0, "EUR": 98.0, "CNY": 12.0},
        {"date": "2026-08-13", "AMD": 0.24, "KZT": 0.19, "KGS": 0.99,
         "TJS": 8.97, "UZS": 0.0079, "USD": 90.1, "EUR": 98.1, "CNY": 12.1},
    ],
}


@pytest.fixture(autouse=True)
def _reset():
    da.rates.cache_clear()
    da._rates_source.update(active="file", error=None)
    old = config.RATES_URL
    yield
    config.RATES_URL = old
    da.rates.cache_clear()


@respx.mock
def test_rates_from_parser_when_configured():
    config.RATES_URL = PARSER
    respx.get(f"{PARSER}/rates/wide").mock(return_value=httpx.Response(200, json=WIDE_OK))
    df = da.rates()
    assert da.rates_source()["active"] == "parser"
    assert set(df["corridor"]) == {"RUB_AMD", "RUB_KZT", "RUB_KGS", "RUB_TJS", "RUB_UZS"}
    tjs = df[(df["corridor"] == "RUB_TJS") & (df["date"].astype(str) == "2026-08-13")]
    assert abs(float(tjs["rate"].iloc[0]) - 8.97) < 1e-9


@respx.mock
def test_rates_falls_back_to_file_on_parser_error():
    config.RATES_URL = PARSER
    respx.get(f"{PARSER}/rates/wide").mock(return_value=httpx.Response(500))
    df = da.rates()                      # не бросает
    assert da.rates_source()["active"] == "file"
    assert da.rates_source()["error"]
    assert len(df) > 0 and {"corridor", "date", "rate", "is_stale"} <= set(df.columns)


def test_rates_reads_file_when_not_configured():
    config.RATES_URL = ""
    df = da.rates()
    assert da.rates_source()["active"] == "file"
    assert len(df) > 100
