"""Интерпретируемые бинарные индикаторы и их простые комбинации."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorRule:
    family: str
    feature: str
    operator: str
    threshold: float

    @property
    def name(self) -> str:
        threshold = f"{self.threshold:g}".replace("-", "m").replace(".", "p")
        operator = {"le": "le", "ge": "ge"}[self.operator]
        return f"{self.family}__{self.feature}__{operator}_{threshold}"

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.feature not in data.columns:
            raise KeyError(f"Нет признака {self.feature!r}")
        values = data[self.feature]
        if self.operator == "le":
            prediction = values.le(self.threshold)
        elif self.operator == "ge":
            prediction = values.ge(self.threshold)
        else:
            raise ValueError(f"Неизвестный operator: {self.operator}")
        return prediction.fillna(False).to_numpy(dtype=bool)


@dataclass(frozen=True)
class IndicatorCandidate:
    rules: tuple[IndicatorRule, ...]
    logic: str = "SINGLE"

    @property
    def name(self) -> str:
        if self.logic == "SINGLE":
            return self.rules[0].name
        separator = f"__{self.logic}__"
        return separator.join(rule.name for rule in self.rules)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(rule.family for rule in self.rules)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        predictions = [rule.predict(data) for rule in self.rules]
        if self.logic == "SINGLE":
            return predictions[0]
        if self.logic == "AND":
            return np.logical_and.reduce(predictions)
        if self.logic == "OR":
            return np.logical_or.reduce(predictions)
        raise ValueError(f"Неизвестная логика: {self.logic}")


def create_indicator_rules(
    indicator_type: str,
    feature: str,
    operator: str,
    thresholds: tuple[float, ...] | list[float],
) -> list[IndicatorRule]:
    """Создать варианты одного индикатора с разными порогами."""
    if operator not in {"le", "ge"}:
        raise ValueError("operator должен быть 'le' или 'ge'")
    if not thresholds:
        raise ValueError("Нужен хотя бы один threshold")
    return [
        IndicatorRule(
            family=indicator_type,
            feature=feature,
            operator=operator,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]


def default_indicator_rules() -> list[IndicatorRule]:
    """Детерминированная небольшая grid-библиотека baseline rules."""
    rules: list[IndicatorRule] = []

    for window in (30, 90, 180):
        for threshold in (0.10, 0.20):
            rules.append(
                IndicatorRule(
                    "level_low",
                    f"percentile_{window}d",
                    "le",
                    threshold,
                )
            )
        for threshold in (25.0, 50.0):
            rules.append(
                IndicatorRule(
                    "near_low",
                    f"distance_from_low_{window}d_bps",
                    "le",
                    threshold,
                )
            )

    for horizon in (1, 3, 5, 10, 20):
        for threshold in (-50.0, 0.0):
            rules.append(
                IndicatorRule(
                    "momentum_down",
                    f"return_{horizon}d_bps",
                    "le",
                    threshold,
                )
            )
        for threshold in (0.0, 50.0):
            rules.append(
                IndicatorRule(
                    "momentum_up",
                    f"return_{horizon}d_bps",
                    "ge",
                    threshold,
                )
            )

    for count in (2, 3, 4):
        rules.append(
            IndicatorRule("down_streak", "consecutive_down", "ge", float(count))
        )
        rules.append(
            IndicatorRule("up_streak", "consecutive_up", "ge", float(count))
        )

    for window in (3, 5, 10):
        rules.append(
            IndicatorRule(
                "trend_down",
                f"slope_{window}d_bps_per_day",
                "le",
                0.0,
            )
        )
        rules.append(
            IndicatorRule(
                "trend_up",
                f"slope_{window}d_bps_per_day",
                "ge",
                0.0,
            )
        )

    for threshold in (1.0, 1.5):
        rules.append(
            IndicatorRule(
                "high_volatility",
                "volatility_ratio_7d_30d",
                "ge",
                threshold,
            )
        )

    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise AssertionError("Grid содержит повторяющиеся rules")
    return rules


def build_indicator_candidates(
    rules: list[IndicatorRule] | None = None,
    *,
    combination_logic: tuple[str, ...] = ("AND", "OR"),
) -> list[IndicatorCandidate]:
    """Построить singles и все pairwise-комбинации разных семейств."""
    base_rules = rules or default_indicator_rules()
    candidates = [IndicatorCandidate((rule,)) for rule in base_rules]

    for logic in combination_logic:
        if logic not in {"AND", "OR"}:
            raise ValueError("Допустимы только AND и OR")

        for left_index, left in enumerate(base_rules):
            for right in base_rules[left_index + 1:]:
                # Варианты одного семейства — альтернативные настройки
                # одного evidence, а не независимые evidence.
                if left.family == right.family:
                    continue
                candidates.append(
                    IndicatorCandidate((left, right), logic=logic)
                )

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise AssertionError("Получены повторяющиеся candidates")
    return candidates


def required_indicator_features(
    candidates: list[IndicatorCandidate],
) -> list[str]:
    return sorted(
        {
            rule.feature
            for candidate in candidates
            for rule in candidate.rules
        }
    )


def prediction_matrix(
    data: pd.DataFrame,
    candidates: list[IndicatorCandidate],
) -> np.ndarray:
    """Посчитать candidates, переиспользуя каждый простой rule один раз."""
    if not candidates:
        raise ValueError("Список candidates не должен быть пустым")

    rule_predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        for rule in candidate.rules:
            if rule.name not in rule_predictions:
                rule_predictions[rule.name] = rule.predict(data)

    columns = []
    for candidate in candidates:
        predictions = [
            rule_predictions[rule.name]
            for rule in candidate.rules
        ]
        if candidate.logic == "SINGLE":
            columns.append(predictions[0])
        elif candidate.logic == "AND":
            columns.append(np.logical_and.reduce(predictions))
        elif candidate.logic == "OR":
            columns.append(np.logical_or.reduce(predictions))
        else:
            raise ValueError(f"Неизвестная логика: {candidate.logic}")
    return np.column_stack(columns)

