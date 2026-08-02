"""Analyst-style derived ratios.

A gradient booster can discover interactions on its own, so these features add
little to raw AUC. They earn their place for a different reason: they are the
quantities a credit analyst reasons in, and they are what a declined customer
can actually be told. "Your wallet balance covers about two days of your normal
spending" is a sentence; ``avg_daily_balance_kes = 412`` is not.

Every ratio is defined so that it stays finite on the whole domain, including
customers with no savings, no borrowing history and no inflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mkopo.data.schema import DERIVED_FEATURES

EPS = 1e-6


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with the derived ratio block appended."""
    out = frame.copy()

    daily_spend = out["monthly_inflow_kes"] / 30.0
    out["buffer_days"] = (out["avg_daily_balance_kes"] / (daily_spend + 1.0)).clip(0, 120).round(4)

    out["liquidity_stress_index"] = (
        (out["days_with_zero_balance_30d"] / 30.0) * out["balance_volatility_cv"]
    ).round(4)

    out["betting_to_savings_ratio"] = (
        out["betting_spend_ratio"] / (out["savings_balance_ratio"] + 0.02)
    ).clip(0, 50).round(4)

    out["credit_hunger_index"] = (
        out["active_lender_count"]
        * (1.0 + out["loan_stacking_velocity_90d"])
        / (1.0 + out["prior_loan_count"])
    ).round(4)

    # Nan-aware blend: a thin file simply contributes fewer components rather
    # than being penalised with a zero it did not earn.
    discipline_parts = out[
        ["utility_payment_punctuality", "rent_payment_regularity", "prior_ontime_repayment_ratio"]
    ]
    out["obligation_discipline_index"] = discipline_parts.mean(axis=1, skipna=True).round(4)

    out["income_stability_index"] = (
        out["inflow_regularity_score"] * (1.0 + 0.5 * out["salary_deposit_flag"])
    ).round(4)

    free_cash = (out["net_cash_flow_ratio"].clip(lower=0.0) + 0.02) * out["monthly_inflow_kes"]
    exposure = out["total_credit_exposure_ratio"] * out["monthly_inflow_kes"]
    out["exposure_to_capacity_ratio"] = (exposure / (free_cash + 1.0)).clip(0, 200).round(4)

    missing = [c for c in DERIVED_FEATURES if c not in out.columns]
    if missing:  # pragma: no cover - guards against schema drift
        raise RuntimeError(f"derived features declared but not computed: {missing}")

    # NaN is legitimate (thin files); infinity never is.
    if np.isinf(out[list(DERIVED_FEATURES)].to_numpy(dtype=float)).any():
        raise RuntimeError("derived features produced infinite values")

    return out
