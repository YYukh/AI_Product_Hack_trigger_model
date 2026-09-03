"""Target Registry для GOOD_NOW и WINDOW_CLOSING."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .outcomes import DEFAULT_HORIZONS


DEFAULT_G1_TOLERANCES_BPS = (25, 50, 100)
DEFAULT_W0_DETERIORATION_BPS = 75
DEFAULT_W1_LOW_PERCENTILE = 0.15


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    family: str
    scenario: str
    horizon: int
    description: str
    tolerance_bps: float | None = None
    deterioration_bps: float | None = None
    low_percentile: float | None = None


def _nullable_binary(
    condition: pd.Series,
    valid: pd.Series,
) -> pd.Series:
    target = pd.Series(pd.NA, index=condition.index, dtype="Int8")
    target.loc[valid] = condition.loc[valid].astype("int8")
    return target


def build_targets(
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    g1_tolerances_bps: tuple[int, ...] = DEFAULT_G1_TOLERANCES_BPS,
    w0_deterioration_bps: int = DEFAULT_W0_DETERIORATION_BPS,
    w1_low_percentile: float = DEFAULT_W1_LOW_PERCENTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создать простые targets из заранее рассчитанных outcomes.

    G0 — exact local minimum в окне ±h.
    G1 — future regret не выше допуска.
    W0 — курс через h дней стал хуже минимум на заданное число bps.
    W1 — W0 при условии, что в T курс находился в выгодной зоне.

    G2 остаётся continuous outcome ``local_advantage_*`` и намеренно не
    превращается здесь в бинарный target.
    """
    if not 0 <= w1_low_percentile <= 1:
        raise ValueError("w1_low_percentile должен лежать в [0, 1]")

    result = outcomes.copy()
    definitions: list[TargetDefinition] = []

    for horizon in horizons:
        regret_column = f"future_best_regret_{horizon}d_bps"
        future_return_column = f"future_return_{horizon}d_bps"
        centered_min_column = f"centered_min_rate_{horizon}d"

        required = {
            regret_column,
            future_return_column,
            centered_min_column,
            "percentile_90d",
            "rate",
        }
        missing = required.difference(result.columns)
        if missing:
            raise KeyError(f"Не хватает outcome/feature полей: {sorted(missing)}")

        g0_name = f"target_g0_exact_min_h{horizon}d"
        g0_valid = result[centered_min_column].notna()
        g0_condition = np.isclose(
            result["rate"],
            result[centered_min_column],
            rtol=0,
            atol=1e-12,
        )
        result[g0_name] = _nullable_binary(
            pd.Series(g0_condition, index=result.index),
            g0_valid,
        )
        definitions.append(
            TargetDefinition(
                name=g0_name,
                family="G0",
                scenario="GOOD_NOW",
                horizon=horizon,
                description="Exact local minimum in ±h calendar days",
            )
        )

        for tolerance in g1_tolerances_bps:
            g1_name = f"target_g1_regret_le_{tolerance}bps_h{horizon}d"
            g1_valid = result[regret_column].notna()
            result[g1_name] = _nullable_binary(
                result[regret_column].le(tolerance),
                g1_valid,
            )
            definitions.append(
                TargetDefinition(
                    name=g1_name,
                    family="G1",
                    scenario="GOOD_NOW",
                    horizon=horizon,
                    tolerance_bps=float(tolerance),
                    description="Future regret is not above tolerance",
                )
            )

        w0_name = f"target_w0_deterioration_{w0_deterioration_bps}bps_h{horizon}d"
        w0_valid = result[future_return_column].notna()
        w0_condition = result[future_return_column].gt(w0_deterioration_bps)
        result[w0_name] = _nullable_binary(w0_condition, w0_valid)
        definitions.append(
            TargetDefinition(
                name=w0_name,
                family="W0",
                scenario="WINDOW_CLOSING",
                horizon=horizon,
                deterioration_bps=float(w0_deterioration_bps),
                description="Rate deteriorates by threshold within h days",
            )
        )

        percentile_label = str(w1_low_percentile).replace(".", "p")
        w1_name = (
            f"target_w1_lowpct_{percentile_label}_"
            f"deterioration_{w0_deterioration_bps}bps_h{horizon}d"
        )
        w1_valid = w0_valid & result["percentile_90d"].notna()
        w1_condition = (
            result["percentile_90d"].le(w1_low_percentile)
            & w0_condition
        )
        result[w1_name] = _nullable_binary(w1_condition, w1_valid)
        definitions.append(
            TargetDefinition(
                name=w1_name,
                family="W1",
                scenario="WINDOW_CLOSING",
                horizon=horizon,
                deterioration_bps=float(w0_deterioration_bps),
                low_percentile=float(w1_low_percentile),
                description="Previously favourable and then deteriorates",
            )
        )

    registry = pd.DataFrame(asdict(item) for item in definitions)
    registry = registry.sort_values(
        ["scenario", "family", "horizon", "tolerance_bps"],
        na_position="first",
    ).reset_index(drop=True)

    return result, registry


def target_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("target_")]

