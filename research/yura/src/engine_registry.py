"""Explicit extension point for independent rule and ML evidence engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .config import YuraPipelineConfig
from .models import build_benefit_regressor, build_classifier
from .rules import RULE_LIBRARY, RuleCandidate


ModelBuilder = Callable[[YuraPipelineConfig], object]


@dataclass(frozen=True)
class MLModelSpec:
    """Named probability estimator trained under the configured ML scope."""

    name: str
    builder: ModelBuilder


@dataclass(frozen=True)
class EngineRegistry:
    """Immutable list of engines used by one reproducible replay."""

    rule_library: Mapping[str, tuple[RuleCandidate, ...]]
    ml_models: tuple[MLModelSpec, ...]
    benefit_model_builder: ModelBuilder

    def __post_init__(self) -> None:
        names = [model.name for model in self.ml_models]
        if not names or len(names) != len(set(names)):
            raise ValueError("ML engine names должны быть непустыми и уникальными")


def default_engine_registry() -> EngineRegistry:
    return EngineRegistry(
        rule_library=RULE_LIBRARY,
        ml_models=(MLModelSpec("pooled_hgb", build_classifier),),
        benefit_model_builder=build_benefit_regressor,
    )
