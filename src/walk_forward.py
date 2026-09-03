"""Расширяющиеся walk-forward разбиения без temporal leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    test_year: int
    horizon: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_periodic_walk_forward_folds(
    data: pd.DataFrame,
    *,
    horizon: int,
    first_test_date: str | pd.Timestamp,
    test_months: int = 12,
    date_column: str = "available_at",
) -> list[WalkForwardFold]:
    """Expanding walk-forward с test-периодом заданной длины в месяцах."""
    if horizon <= 0:
        raise ValueError("horizon должен быть положительным")
    if test_months <= 0:
        raise ValueError("test_months должен быть положительным")
    if date_column not in data.columns:
        raise KeyError(f"Нет столбца {date_column!r}")

    dates = pd.to_datetime(data[date_column])
    if dates.isna().any():
        raise ValueError("В данных есть пустые даты")

    max_date = dates.max()
    test_start = pd.Timestamp(first_test_date)
    folds: list[WalkForwardFold] = []
    fold_id = 1

    while test_start <= max_date:
        next_test_start = test_start + pd.DateOffset(months=test_months)
        test_end_exclusive = min(next_test_start, max_date + pd.Timedelta(days=1))

        test_mask = dates.ge(test_start) & dates.lt(test_end_exclusive)
        train_mask = dates + pd.Timedelta(days=horizon) < test_start
        train_index = data.index[train_mask].to_numpy()
        test_index = data.index[test_mask].to_numpy()

        if len(train_index) and len(test_index):
            train_dates = dates.loc[train_index]
            test_dates = dates.loc[test_index]
            if not (
                train_dates.max() + pd.Timedelta(days=horizon)
                < test_dates.min()
            ):
                raise AssertionError("Нарушен purge между train и test")

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    test_year=test_start.year,
                    horizon=horizon,
                    train_index=train_index,
                    test_index=test_index,
                    train_start=train_dates.min(),
                    train_end=train_dates.max(),
                    test_start=test_dates.min(),
                    test_end=test_dates.max(),
                )
            )
            fold_id += 1

        test_start = next_test_start

    if not folds:
        raise ValueError("Не удалось построить periodic walk-forward folds")
    return folds


def make_walk_forward_folds(
    data: pd.DataFrame,
    *,
    horizon: int,
    first_test_year: int = 2023,
    date_column: str = "available_at",
) -> list[WalkForwardFold]:
    """Создать expanding folds и purge по длине будущего outcome.

    Строка train допустима, только если её label window полностью
    заканчивается до начала test: ``date + horizon < test_start``.
    """
    if horizon <= 0:
        raise ValueError("horizon должен быть положительным")
    if date_column not in data.columns:
        raise KeyError(f"Нет столбца {date_column!r}")

    dates = pd.to_datetime(data[date_column])
    if dates.isna().any():
        raise ValueError("В данных есть пустые даты")

    max_date = dates.max()
    folds: list[WalkForwardFold] = []

    for fold_id, test_year in enumerate(
        range(first_test_year, max_date.year + 1),
        start=1,
    ):
        nominal_test_start = pd.Timestamp(test_year, 1, 1)
        nominal_test_end = pd.Timestamp(test_year, 12, 31)

        test_mask = dates.between(
            nominal_test_start,
            min(nominal_test_end, max_date),
            inclusive="both",
        )
        train_mask = (
            dates + pd.Timedelta(days=horizon)
            < nominal_test_start
        )

        train_index = data.index[train_mask].to_numpy()
        test_index = data.index[test_mask].to_numpy()

        if len(train_index) == 0 or len(test_index) == 0:
            continue

        train_dates = dates.loc[train_index]
        test_dates = dates.loc[test_index]

        if not (
            train_dates.max() + pd.Timedelta(days=horizon)
            < test_dates.min()
        ):
            raise AssertionError("Нарушен purge между train и test")

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                test_year=test_year,
                horizon=horizon,
                train_index=train_index,
                test_index=test_index,
                train_start=train_dates.min(),
                train_end=train_dates.max(),
                test_start=test_dates.min(),
                test_end=test_dates.max(),
            )
        )

    if not folds:
        raise ValueError("Не удалось построить walk-forward folds")

    return folds


def folds_summary(folds: list[WalkForwardFold]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "horizon": fold.horizon,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": len(fold.train_index),
            "test_rows": len(fold.test_index),
        }
        for fold in folds
    )

