"""Compact alternative FX signal pipeline."""

from .config import YuraPipelineConfig
from .evidence import (
    EvidenceResult, add_causal_relative_scores, aggregate_engine_evidence,
    build_evidence_matrix,
)
from .engine_registry import EngineRegistry, MLModelSpec, default_engine_registry
from .pipeline import YuraPipelineResult, run_yura_pipeline
from .policy import SignalPolicyConfig, apply_signal_policy
from .reporting import build_action_summary, compare_backtest_summaries
from .selector import OpportunitySelector, ThresholdSelector
from .targets import build_yura_targets
from .temporal import TemporalPlan

__all__ = [
    "YuraPipelineConfig", "TemporalPlan", "EvidenceResult",
    "aggregate_engine_evidence", "build_evidence_matrix",
    "add_causal_relative_scores", "SignalPolicyConfig", "apply_signal_policy",
    "EngineRegistry", "MLModelSpec", "default_engine_registry",
    "YuraPipelineResult", "run_yura_pipeline",
    "OpportunitySelector", "ThresholdSelector",
    "build_yura_targets",
    "build_action_summary", "compare_backtest_summaries",
]
