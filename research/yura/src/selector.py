"""Replaceable selector boundary between evidence aggregation and policy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from .arbiter import FittedArbiter, fit_arbiter, select_opportunities
from .config import YuraPipelineConfig
from .policy import SignalPolicyConfig


@runtime_checkable
class OpportunitySelector(Protocol):
    def fit(
        self,
        candidates: pd.DataFrame,
        *,
        evaluation_universe: pd.DataFrame,
        config: YuraPipelineConfig,
    ) -> tuple[object, pd.DataFrame]: ...

    def select(self, candidates: pd.DataFrame, fitted: object) -> pd.DataFrame: ...

    def policy_config(self, fitted: object) -> SignalPolicyConfig: ...


class ThresholdSelector:
    """Small validation-audited selector supplied as the default baseline."""

    def fit(
        self,
        candidates: pd.DataFrame,
        *,
        evaluation_universe: pd.DataFrame,
        config: YuraPipelineConfig,
    ) -> tuple[FittedArbiter, pd.DataFrame]:
        return fit_arbiter(
            candidates, evaluation_universe=evaluation_universe, config=config
        )

    def select(
        self,
        candidates: pd.DataFrame,
        fitted: FittedArbiter,
    ) -> pd.DataFrame:
        return select_opportunities(candidates, fitted.config)

    def policy_config(self, fitted: FittedArbiter) -> SignalPolicyConfig:
        return SignalPolicyConfig(
            cooldown_days=fitted.config.cooldown_days,
            max_signals_per_7d=fitted.config.max_signals_per_7d,
        )
