"""Frozen, deliberately small configuration of the alternative pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YuraPipelineConfig:
    currencies: tuple[str, ...] = ("AMD", "KZT", "KGS", "TJS", "UZS")
    target_families: tuple[str, ...] = ("G0", "W1")
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)

    # Dates are derived from the available dataset unless explicitly frozen.
    base_oos_start: str | None = None
    arbiter_validation_start: str | None = None
    holdout_start: str | None = None
    train_months: int = 36
    retrain_months: int = 6
    rule_confidence_months: int = 24
    rule_confidence_prior_strength: float = 20.0
    confidence_warmup_months: int = 12
    selector_validation_months: int = 36
    relative_rank_months: int = 36

    # Original W1: the rate is currently in the low part of its trailing
    # corridor and deteriorates by the specified amount after h days.
    w1_deterioration_bps: float = 75.0
    w1_low_percentile: float = 0.15

    # Rule parameter fitting. Frequency is per currency and concerns base
    # candidates, not the final push budget.
    rule_min_signals_per_week: float = 0.25
    rule_max_signals_per_week: float = 4.00

    # Small fixed models: these values are not optimized.
    ml_learning_rate: float = 0.06
    ml_max_iter: int = 80
    ml_max_leaf_nodes: int = 7
    ml_min_samples_leaf: int = 30
    ml_l2_regularization: float = 2.0
    # pooled: legacy shared model/calibration;
    # hybrid: shared model + balanced strata + currency/horizon calibration;
    # per_currency: one classifier per family and currency, horizons pooled.
    ml_scope: str = "pooled"
    random_state: int = 42

    # Validation-only selection grid: a small set of transparent global
    # policies; there are no currency/family/horizon-specific thresholds.
    rule_lift_thresholds: tuple[float, ...] = (1.30,)
    ml_lift_thresholds: tuple[float, ...] = (1.30,)
    # Positive expected uplift is an economic admissibility condition, not
    # another hyperparameter fitted to the target.
    min_expected_bps_options: tuple[float, ...] = (0.0,)
    decision_score_thresholds: tuple[float, ...] = (0.00,)
    cooldown_options: tuple[int, ...] = (2,)
    max_signals_per_7d: int = 2
    min_average_signals_per_week: float = 1.0
    max_average_signals_per_week: float = 2.0
    validation_quarterly_lift_floor: float = 1.30

    # Learned opportunity selectors use three strictly ordered intervals
    # inside selector validation: estimator fit -> probability calibration ->
    # threshold validation.  The final holdout is never used by any of them.
    learned_selector_calibration_months: int = 12
    learned_selector_validation_months: int = 12
    learned_selector_thresholds: tuple[float, ...] = (
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    )

    def __post_init__(self) -> None:
        if self.train_months <= 0 or self.retrain_months <= 0:
            raise ValueError("train_months и retrain_months должны быть положительными")
        if self.rule_confidence_months <= 0:
            raise ValueError("rule_confidence_months должен быть положительным")
        if self.max_signals_per_7d <= 0:
            raise ValueError("max_signals_per_7d должен быть положительным")
        if min(
            self.confidence_warmup_months,
            self.selector_validation_months,
            self.relative_rank_months,
        ) <= 0:
            raise ValueError("Все временные окна должны быть положительными")
        if self.w1_deterioration_bps < 0:
            raise ValueError("w1_deterioration_bps не может быть отрицательным")
        if not 0.0 <= self.w1_low_percentile <= 1.0:
            raise ValueError("w1_low_percentile должен лежать в [0, 1]")
        if self.ml_scope not in {"pooled", "hybrid", "per_currency"}:
            raise ValueError("ml_scope должен быть pooled, hybrid или per_currency")
        if self.validation_quarterly_lift_floor <= 0:
            raise ValueError("validation_quarterly_lift_floor должен быть положительным")
        if min(
            self.learned_selector_calibration_months,
            self.learned_selector_validation_months,
        ) <= 0:
            raise ValueError("Окна learned selector должны быть положительными")
        if (
            self.learned_selector_calibration_months
            + self.learned_selector_validation_months
            >= self.selector_validation_months
        ):
            raise ValueError(
                "До calibration и validation learned selector должен оставаться "
                "непустой fit-период"
            )
        if not self.learned_selector_thresholds or any(
            not 0.0 < value < 1.0 for value in self.learned_selector_thresholds
        ):
            raise ValueError("Пороги learned selector должны лежать строго между 0 и 1")
        if (
            self.arbiter_validation_start is not None
            and self.holdout_start is not None
            and self.arbiter_validation_start >= self.holdout_start
        ):
            raise ValueError("validation должна начинаться раньше holdout")
