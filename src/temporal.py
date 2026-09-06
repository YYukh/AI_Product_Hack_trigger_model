"""Dynamic and auditable temporal boundaries for research and production replay."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .config import YuraPipelineConfig


@dataclass(frozen=True)
class TemporalPlan:
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    base_oos_start: pd.Timestamp
    selector_validation_start: pd.Timestamp
    holdout_start: pd.Timestamp
    maximum_horizon_days: int

    @property
    def selector_validation_end(self) -> pd.Timestamp:
        return self.holdout_start

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        config: YuraPipelineConfig,
    ) -> "TemporalPlan":
        if "available_at" not in data:
            raise KeyError("Для TemporalPlan нужен available_at")
        dates = pd.to_datetime(data["available_at"], errors="raise").dropna()
        if dates.empty:
            raise ValueError("Нельзя построить TemporalPlan по пустым данным")
        data_start = dates.min().to_period("M").start_time
        data_end = dates.max()
        derived_base = data_start + pd.DateOffset(months=config.train_months)
        base_oos = (
            pd.Timestamp(config.base_oos_start)
            if config.base_oos_start is not None else derived_base
        )
        warmup_end = base_oos + pd.DateOffset(
            months=config.confidence_warmup_months
        )
        validation = (
            pd.Timestamp(config.arbiter_validation_start)
            if config.arbiter_validation_start is not None
            else warmup_end
        )
        holdout = (
            pd.Timestamp(config.holdout_start)
            if config.holdout_start is not None
            else validation + pd.DateOffset(
                months=config.selector_validation_months
            )
        )
        if not data_start < base_oos <= validation < holdout <= data_end:
            raise ValueError(
                "Некорректная временная схема: требуется "
                "data_start < base_oos <= validation < holdout <= data_end"
            )
        return cls(
            data_start=data_start,
            data_end=data_end,
            base_oos_start=base_oos,
            selector_validation_start=validation,
            holdout_start=holdout,
            maximum_horizon_days=max(config.horizons),
        )

    def resolve(self, config: YuraPipelineConfig) -> YuraPipelineConfig:
        """Return an immutable runtime config understood by legacy engines."""
        return replace(
            config,
            base_oos_start=self.base_oos_start.isoformat(),
            arbiter_validation_start=self.selector_validation_start.isoformat(),
            holdout_start=self.holdout_start.isoformat(),
        )

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "data_start": self.data_start,
            "data_end": self.data_end,
            "base_oos_start": self.base_oos_start,
            "selector_validation_start": self.selector_validation_start,
            "selector_validation_end": self.selector_validation_end,
            "holdout_start": self.holdout_start,
            "maximum_horizon_days": self.maximum_horizon_days,
        }])
