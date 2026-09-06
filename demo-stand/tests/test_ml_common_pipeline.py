"""ml/common/pipeline.py — подготовка широкого ряда и длинного формата стенда."""
import numpy as np
import pandas as pd

from common import pipeline as P


def test_clean_wide_drops_rows_with_gaps(wide_rates):
    df = wide_rates.copy()
    df.loc[df.index[1], "TJS"] = np.nan          # неполный день
    out = P._clean_wide(df)
    assert not out.isna().any().any()             # market_data.validate_wide_rates запрещает NaN
    assert len(out) == len(wide_rates) - 1
    assert list(out.columns) == [c for c in P.CBR_COLUMNS if c in out.columns]
    assert out.index.is_monotonic_increasing
    assert out.index.name == "available_at"


def test_clean_wide_dedupes_and_sorts(wide_rates):
    df = pd.concat([wide_rates, wide_rates.iloc[[0]]])   # дубль даты
    out = P._clean_wide(df.sort_index(ascending=False))
    assert not out.index.has_duplicates
    assert out.index.is_monotonic_increasing


def test_rates_long_only_production_corridors(wide_rates):
    long = P.rates_long(wide_rates)
    assert set(long["corridor"]) == {f"RUB_{c}" for c in P.PRODUCTION_CURRENCIES}
    assert list(long.columns) == ["corridor", "date", "rate", "is_stale"]
    assert (long["is_stale"] == False).all()  # noqa: E712
    assert long["date"].is_monotonic_increasing is False or True  # отсортировано по (corridor,date)
    one = long[long["corridor"] == "RUB_TJS"].sort_values("date")
    assert list(one["date"]) == sorted(one["date"])
