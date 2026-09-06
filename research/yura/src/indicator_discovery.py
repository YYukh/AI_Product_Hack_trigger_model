"""Production-shaped walk-forward laboratory for indicator hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd

from src.indicators import IndicatorRule


@dataclass(frozen=True)
class IndicatorDiscoveryConfig:
    train_months: int = 36
    test_months: int = 6
    discovery_oos_months: int = 48
    min_train_signals_per_week: float = 0.25
    max_train_signals_per_week: float = 4.0
    top_rules_per_family: int = 2
    max_pair_families: int = 12
    precision_prior_strength: float = 20.0
    min_oos_signals: int = 20

    def __post_init__(self) -> None:
        if min(self.train_months, self.test_months, self.discovery_oos_months) <= 0:
            raise ValueError("Временные окна discovery должны быть положительными")
        if not 0 <= self.min_train_signals_per_week <= self.max_train_signals_per_week:
            raise ValueError("Некорректный диапазон частоты base rules")
        if min(self.top_rules_per_family, self.max_pair_families, self.min_oos_signals) <= 0:
            raise ValueError("Ограничения библиотеки/support должны быть > 0")


@dataclass
class IndicatorDiscoveryResult:
    temporal_plan: pd.DataFrame
    catalogue: pd.DataFrame
    fold_results: pd.DataFrame
    leaderboard: pd.DataFrame
    best_by_configuration: pd.DataFrame
    family_summary: pd.DataFrame


def build_discovery_temporal_plan(
    data: pd.DataFrame,
    config: IndicatorDiscoveryConfig,
) -> pd.DataFrame:
    dates = pd.to_datetime(data["available_at"], errors="raise")
    data_start = dates.min().to_period("M").start_time
    oos_start = data_start + pd.DateOffset(months=config.train_months)
    desired_end = oos_start + pd.DateOffset(months=config.discovery_oos_months)
    available_end = dates.max().to_period("M").start_time
    discovery_end = min(desired_end, available_end)
    if discovery_end <= oos_start:
        raise ValueError("Недостаточно истории для discovery OOS")
    rows = []
    fold_start = oos_start
    fold_id = 1
    while fold_start < discovery_end:
        fold_end = min(
            fold_start + pd.DateOffset(months=config.test_months), discovery_end
        )
        rows.append({
            "fold_id": fold_id,
            "train_start": fold_start - pd.DateOffset(months=config.train_months),
            "test_start": fold_start,
            "test_end": fold_end,
            "discovery_end": discovery_end,
            "reserved_after": discovery_end,
        })
        fold_start = fold_end
        fold_id += 1
    return pd.DataFrame(rows)


def _weeks(dates: pd.Series) -> float:
    if dates.empty:
        return 0.0
    values = pd.to_datetime(dates)
    return max((values.max() - values.min()).days + 1, 1) / 7.0


def _selection_statistics(
    predictions: np.ndarray,
    target: np.ndarray,
    benefit: np.ndarray,
    *,
    weeks: float,
    baseline: float,
    prior_strength: float,
) -> pd.DataFrame:
    matrix = np.asarray(predictions, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    count = matrix.sum(axis=0).astype(int)
    true_positive = (matrix & target[:, None]).sum(axis=0).astype(int)
    precision = np.divide(
        true_positive, count, out=np.zeros_like(count, dtype=float), where=count > 0
    )
    smoothed_precision = (
        true_positive + prior_strength * baseline
    ) / (count + prior_strength)
    finite_benefit = np.isfinite(benefit)
    benefit_count = (matrix & finite_benefit[:, None]).sum(axis=0)
    benefit_sum = np.where(
        matrix & finite_benefit[:, None], benefit[:, None], 0.0
    ).sum(axis=0)
    mean_benefit = np.divide(
        benefit_sum, benefit_count,
        out=np.full(len(count), -np.inf, dtype=float), where=benefit_count > 0,
    )
    return pd.DataFrame({
        "signal_count": count,
        "true_positive": true_positive,
        "precision": precision,
        "smoothed_lift": smoothed_precision / baseline if baseline > 0 else 0.0,
        "signals_per_week": count / max(weeks, 1e-9),
        "mean_benefit_bps": mean_benefit,
    })


def _rank_positions(
    statistics: pd.DataFrame,
    positions: np.ndarray,
    config: IndicatorDiscoveryConfig,
    limit: int,
) -> list[int]:
    sample = statistics.loc[positions].copy()
    feasible = sample.loc[sample["signals_per_week"].between(
        config.min_train_signals_per_week,
        config.max_train_signals_per_week,
        inclusive="both",
    )]
    if feasible.empty:
        return []
    ordered = feasible.sort_values(
        ["smoothed_lift", "mean_benefit_bps", "signal_count"],
        ascending=[False, False, False],
    )
    return ordered.head(limit).index.astype(int).tolist()


def _fold_record(
    *,
    base: dict,
    strategy_kind: str,
    strategy_name: str,
    logic: str,
    selected_spec: str,
    train_metrics: pd.Series,
    test_prediction: np.ndarray,
    test_target: np.ndarray,
    test_benefit: np.ndarray,
    test_weeks: float,
) -> dict:
    prediction = np.asarray(test_prediction, dtype=bool)
    target = np.asarray(test_target, dtype=bool)
    benefit = np.asarray(test_benefit, dtype=float)
    finite = np.isfinite(benefit)
    signal_count = int(prediction.sum())
    true_positive = int(np.sum(prediction & target))
    benefit_mask = prediction & finite
    return {
        **base,
        "strategy_kind": strategy_kind,
        "strategy_name": strategy_name,
        "logic": logic,
        "selected_spec": selected_spec,
        "train_smoothed_lift": float(train_metrics["smoothed_lift"]),
        "train_signals_per_week": float(train_metrics["signals_per_week"]),
        "test_observations": len(target),
        "test_positive_count": int(target.sum()),
        "test_signal_count": signal_count,
        "test_true_positive": true_positive,
        "test_signal_benefit_sum": float(np.nansum(benefit[benefit_mask])),
        "test_signal_benefit_count": int(benefit_mask.sum()),
        "test_random_benefit_sum": float(np.nansum(benefit[finite])),
        "test_random_benefit_count": int(finite.sum()),
        "test_weeks": float(test_weeks),
    }


def _wilson_lower(successes: pd.Series, trials: pd.Series) -> pd.Series:
    successes = successes.astype(float)
    trials = trials.astype(float)
    proportion = successes / trials.replace(0, np.nan)
    z = 1.959963984540054
    denominator = 1.0 + z ** 2 / trials.replace(0, np.nan)
    centre = proportion + z ** 2 / (2.0 * trials.replace(0, np.nan))
    spread = z * np.sqrt(
        proportion * (1.0 - proportion) / trials.replace(0, np.nan)
        + z ** 2 / (4.0 * trials.replace(0, np.nan) ** 2)
    )
    return ((centre - spread) / denominator).fillna(0.0)


def summarize_discovery_folds(
    folds: pd.DataFrame,
    *,
    min_oos_signals: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = [
        "currency", "target_family", "target", "horizon",
        "strategy_kind", "strategy_name", "logic",
    ]
    rows = []
    for group_keys, sample in folds.groupby(keys, sort=True):
        observations = int(sample["test_observations"].sum())
        positives = int(sample["test_positive_count"].sum())
        signals = int(sample["test_signal_count"].sum())
        true_positive = int(sample["test_true_positive"].sum())
        precision = true_positive / signals if signals else 0.0
        baseline = positives / observations if observations else 0.0
        fold_baseline = sample["test_positive_count"] / sample["test_observations"]
        fold_precision = sample["test_true_positive"] / sample["test_signal_count"].replace(0, np.nan)
        fold_lift = fold_precision / fold_baseline.replace(0, np.nan)
        signal_benefit_count = int(sample["test_signal_benefit_count"].sum())
        random_benefit_count = int(sample["test_random_benefit_count"].sum())
        mean_benefit = (
            sample["test_signal_benefit_sum"].sum() / signal_benefit_count
            if signal_benefit_count else np.nan
        )
        random_benefit = (
            sample["test_random_benefit_sum"].sum() / random_benefit_count
            if random_benefit_count else np.nan
        )
        row = {
            **dict(zip(keys, group_keys)),
            "oos_fold_count": int(sample["fold_id"].nunique()),
            "oos_nonempty_fold_count": int(sample["test_signal_count"].gt(0).sum()),
            "oos_observations": observations,
            "oos_positive_count": positives,
            "oos_signal_count": signals,
            "oos_true_positive": true_positive,
            "oos_precision": precision,
            "oos_random_precision": baseline,
            "oos_lift": precision / baseline if baseline > 0 else 0.0,
            "oos_precision_lcb95": 0.0,
            "oos_lift_lcb95": 0.0,
            "oos_signals_per_week": signals / max(sample["test_weeks"].sum(), 1e-9),
            "oos_mean_benefit_bps": mean_benefit,
            "oos_random_mean_benefit_bps": random_benefit,
            "oos_benefit_uplift_bps": mean_benefit - random_benefit,
            "fold_lift_median": float(fold_lift.median()) if fold_lift.notna().any() else 0.0,
            "fold_lift_p25": float(fold_lift.quantile(0.25)) if fold_lift.notna().any() else 0.0,
            "fold_share_lift_gt_1": float(fold_lift.gt(1.0).mean()),
            "fold_share_lift_ge_1p3": float(fold_lift.ge(1.3).mean()),
            "parameter_stability": float(
                sample["selected_spec"].value_counts(normalize=True).iloc[0]
            ),
            "confidence_method": "aggregated_walk_forward_oos_precision",
        }
        rows.append(row)
    leaderboard = pd.DataFrame(rows)
    leaderboard["oos_precision_lcb95"] = _wilson_lower(
        leaderboard["oos_true_positive"], leaderboard["oos_signal_count"]
    )
    leaderboard["oos_lift_lcb95"] = np.divide(
        leaderboard["oos_precision_lcb95"], leaderboard["oos_random_precision"],
        out=np.zeros(len(leaderboard)),
        where=leaderboard["oos_random_precision"].to_numpy() > 0,
    )
    enough = leaderboard["oos_signal_count"].ge(min_oos_signals)
    strong = (
        enough & leaderboard["oos_lift_lcb95"].ge(1.30)
        & leaderboard["oos_benefit_uplift_bps"].gt(0)
        & leaderboard["fold_share_lift_gt_1"].ge(0.75)
    )
    promising = (
        enough & leaderboard["oos_lift"].ge(1.30)
        & leaderboard["oos_benefit_uplift_bps"].gt(0)
        & leaderboard["fold_share_lift_gt_1"].ge(0.50)
    )
    leaderboard["quality_group"] = np.select(
        [strong, promising, enough],
        ["strong", "promising", "weak"],
        default="insufficient_support",
    )
    leaderboard = leaderboard.sort_values(
        ["target_family", "target", "horizon", "currency",
         "oos_lift_lcb95", "oos_benefit_uplift_bps"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    config_keys = ["currency", "target_family", "target", "horizon"]
    best = (
        leaderboard.sort_values(
            [*config_keys, "oos_lift_lcb95", "oos_benefit_uplift_bps", "oos_signal_count"],
            ascending=[True, True, True, True, False, False, False],
        )
        .drop_duplicates(config_keys)
        .reset_index(drop=True)
    )
    family_summary = (
        leaderboard.groupby(
            ["target_family", "strategy_kind", "strategy_name"], sort=True
        )
        .agg(
            tested_configurations=("target", "size"),
            median_oos_lift=("oos_lift", "median"),
            median_lift_lcb95=("oos_lift_lcb95", "median"),
            median_benefit_uplift_bps=("oos_benefit_uplift_bps", "median"),
            median_fold_stability=("fold_share_lift_gt_1", "median"),
            strong_share=("quality_group", lambda values: float(values.eq("strong").mean())),
            promising_or_better_share=(
                "quality_group",
                lambda values: float(values.isin(["strong", "promising"]).mean()),
            ),
        )
        .reset_index()
        .sort_values(
            ["target_family", "strong_share", "median_lift_lcb95"],
            ascending=[True, False, False],
        )
    )
    return leaderboard, best, family_summary


def run_rule_indicator_discovery(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    rules: list[IndicatorRule],
    currencies: tuple[str, ...],
    config: IndicatorDiscoveryConfig = IndicatorDiscoveryConfig(),
) -> IndicatorDiscoveryResult:
    """Fit rule parameters per fold and evaluate singles plus pairwise AND/OR."""
    required = {"available_at", "currency", "rate", *target_registry["name"]}
    required.update(rule.feature for rule in rules)
    if missing := required.difference(data.columns):
        raise KeyError(f"Discovery data не содержит: {sorted(missing)}")
    prepared = data.loc[data["currency"].isin(currencies)].copy()
    prepared["available_at"] = pd.to_datetime(prepared["available_at"])
    prepared = prepared.sort_values(["available_at", "currency"]).reset_index(drop=True)
    plan = build_discovery_temporal_plan(prepared, config)
    rule_matrix = np.column_stack([rule.predict(prepared) for rule in rules])
    families: dict[str, np.ndarray] = {}
    for family in sorted({rule.family for rule in rules}):
        families[family] = np.asarray(
            [index for index, rule in enumerate(rules) if rule.family == family],
            dtype=int,
        )
    catalogue = pd.DataFrame({
        "rule_index": range(len(rules)),
        "rule_name": [rule.name for rule in rules],
        "indicator_family": [rule.family for rule in rules],
        "feature": [rule.feature for rule in rules],
        "operator": [rule.operator for rule in rules],
        "threshold": [rule.threshold for rule in rules],
    })
    dates = prepared["available_at"]
    fold_rows: list[dict] = []

    for definition in target_registry.itertuples(index=False):
        target_name = str(definition.name)
        horizon = int(definition.horizon)
        benefit_column = f"local_advantage_{horizon}d_bps"
        for currency in currencies:
            currency_mask = prepared["currency"].eq(currency).to_numpy()
            for fold in plan.itertuples(index=False):
                maturity = dates.add(pd.Timedelta(days=horizon))
                train_mask = (
                    currency_mask
                    & dates.ge(fold.train_start).to_numpy()
                    & maturity.lt(fold.test_start).to_numpy()
                    & prepared[target_name].notna().to_numpy()
                )
                test_mask = (
                    currency_mask
                    & dates.ge(fold.test_start).to_numpy()
                    & dates.lt(fold.test_end).to_numpy()
                    & maturity.lt(fold.discovery_end).to_numpy()
                    & prepared[target_name].notna().to_numpy()
                )
                if not train_mask.any() or not test_mask.any():
                    continue
                train_target = prepared.loc[train_mask, target_name].astype(bool).to_numpy()
                test_target = prepared.loc[test_mask, target_name].astype(bool).to_numpy()
                baseline = float(train_target.mean())
                if not 0.0 < baseline < 1.0:
                    continue
                train_benefit = pd.to_numeric(
                    prepared.loc[train_mask, benefit_column], errors="coerce"
                ).to_numpy(float)
                test_benefit = pd.to_numeric(
                    prepared.loc[test_mask, benefit_column], errors="coerce"
                ).to_numpy(float)
                train_predictions = rule_matrix[train_mask]
                test_predictions = rule_matrix[test_mask]
                train_weeks = _weeks(prepared.loc[train_mask, "available_at"])
                test_weeks = _weeks(prepared.loc[test_mask, "available_at"])
                statistics = _selection_statistics(
                    train_predictions, train_target, train_benefit,
                    weeks=train_weeks, baseline=baseline,
                    prior_strength=config.precision_prior_strength,
                )
                top_by_family = {
                    family: _rank_positions(
                        statistics, positions, config, config.top_rules_per_family
                    )
                    for family, positions in families.items()
                }
                top_by_family = {
                    family: positions for family, positions in top_by_family.items()
                    if positions
                }
                # Pair search is screened on train only. It keeps the search
                # broad while avoiding a quadratic explosion across dozens of
                # weak families; the test fold is never consulted.
                pair_families = sorted(
                    top_by_family,
                    key=lambda family: float(
                        statistics.loc[top_by_family[family][0], "smoothed_lift"]
                    ),
                    reverse=True,
                )[:config.max_pair_families]
                base = {
                    "fold_id": int(fold.fold_id),
                    "train_start": fold.train_start,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "currency": currency,
                    "target_family": str(definition.family),
                    "target": target_name,
                    "horizon": horizon,
                }
                for family, positions in top_by_family.items():
                    winner = positions[0]
                    fold_rows.append(_fold_record(
                        base=base, strategy_kind="single",
                        strategy_name=family, logic="SINGLE",
                        selected_spec=rules[winner].name,
                        train_metrics=statistics.loc[winner],
                        test_prediction=test_predictions[:, winner],
                        test_target=test_target, test_benefit=test_benefit,
                        test_weeks=test_weeks,
                    ))

                for left_family, right_family in combinations(pair_families, 2):
                    candidate_pairs = list(product(
                        top_by_family[left_family], top_by_family[right_family]
                    ))
                    for logic in ("AND", "OR"):
                        pair_train = np.column_stack([
                            (
                                train_predictions[:, left] & train_predictions[:, right]
                                if logic == "AND"
                                else train_predictions[:, left] | train_predictions[:, right]
                            )
                            for left, right in candidate_pairs
                        ])
                        pair_stats = _selection_statistics(
                            pair_train, train_target, train_benefit,
                            weeks=train_weeks, baseline=baseline,
                            prior_strength=config.precision_prior_strength,
                        )
                        feasible = pair_stats.loc[pair_stats["signals_per_week"].between(
                            config.min_train_signals_per_week,
                            config.max_train_signals_per_week,
                            inclusive="both",
                        )]
                        if feasible.empty:
                            continue
                        winner_position = int(feasible.sort_values(
                            ["smoothed_lift", "mean_benefit_bps", "signal_count"],
                            ascending=[False, False, False],
                        ).index[0])
                        left, right = candidate_pairs[winner_position]
                        pair_test = (
                            test_predictions[:, left] & test_predictions[:, right]
                            if logic == "AND"
                            else test_predictions[:, left] | test_predictions[:, right]
                        )
                        strategy_name = f"{left_family}+{right_family}"
                        fold_rows.append(_fold_record(
                            base=base, strategy_kind="combination",
                            strategy_name=strategy_name, logic=logic,
                            selected_spec=f"{rules[left].name} {logic} {rules[right].name}",
                            train_metrics=pair_stats.loc[winner_position],
                            test_prediction=pair_test, test_target=test_target,
                            test_benefit=test_benefit, test_weeks=test_weeks,
                        ))

    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        raise ValueError("Rule discovery не создал ни одного OOS результата")
    leaderboard, best, family_summary = summarize_discovery_folds(
        folds, min_oos_signals=config.min_oos_signals
    )
    return IndicatorDiscoveryResult(
        temporal_plan=plan,
        catalogue=catalogue,
        fold_results=folds,
        leaderboard=leaderboard,
        best_by_configuration=best,
        family_summary=family_summary,
    )
