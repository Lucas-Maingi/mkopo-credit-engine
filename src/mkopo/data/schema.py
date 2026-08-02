"""Feature contract for the Mkopo book.

One place defines what a feature is called, what family it belongs to and
whether the model is allowed to see it. Everything downstream — the WOE binner,
the API request schema, the fairness audit, the data dictionary — reads from
here, so the contract cannot drift between training and serving.
"""

from __future__ import annotations

from dataclasses import dataclass

ID_COLUMN = "customer_id"
DATE_COLUMN = "application_date"
TARGET = "default_flag"

#: Observed only for loans the legacy policy approved. NaN elsewhere.
APPROVAL_COLUMN = "approved"

#: Counterfactual outcome for rejected applicants. Lives in a separate oracle
#: file and is NEVER available to a model — it exists purely so that reject
#: inference can be honestly validated instead of merely asserted.
ORACLE_TARGET = "true_default_flag"


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    dtype: str  # "numeric" | "categorical"
    description: str
    #: Higher value is expected to mean *lower* risk (+1), *higher* risk (-1),
    #: or no monotone prior (0). Drives monotonic constraints and sanity tests.
    risk_direction: int = 0
    protected: bool = False
    modelled: bool = True


FEATURES: tuple[Feature, ...] = (
    # ---------------------------------------------------------------- identity
    Feature("age", "demographic", "numeric", "Age in years at application", 0, modelled=True),
    Feature("gender", "demographic", "categorical", "Self-reported gender", 0, protected=True, modelled=False),
    Feature("region", "demographic", "categorical", "Kenyan region of residence", 0, protected=True, modelled=False),
    Feature("age_band", "demographic", "categorical", "Banded age, audit only", 0, protected=True, modelled=False),
    Feature("segment", "demographic", "categorical", "Livelihood segment (salaried, trader, gig, farmer, student)", 0),
    Feature("tenure_days", "identity", "numeric", "Days since SIM registration", +1),
    Feature("device_tier", "identity", "categorical", "Handset tier: feature, entry, mid, premium", 0),
    Feature("sim_swap_count_12m", "identity", "numeric", "SIM swaps in the last 12 months (fraud proxy)", -1),
    # ------------------------------------------------------------------ income
    Feature("monthly_inflow_kes", "income", "numeric", "Mean monthly money-in, KES", +1),
    Feature("inflow_txn_count_30d", "income", "numeric", "Count of inbound transactions, 30d", +1),
    Feature("inflow_regularity_score", "income", "numeric", "1 - CV of gaps between inflows; rhythm of income", +1),
    Feature("salary_deposit_flag", "income", "numeric", "Recurring same-payer credit consistent with payroll", +1),
    Feature("inflow_concentration_hhi", "income", "numeric", "Herfindahl index of inflow sources", 0),
    Feature("income_growth_3m", "income", "numeric", "3-month trend in monthly inflow", +1),
    # --------------------------------------------------------------- cash flow
    Feature("avg_daily_balance_kes", "cashflow", "numeric", "Mean end-of-day wallet balance, KES", +1),
    Feature("min_balance_kes", "cashflow", "numeric", "Minimum end-of-day balance in 30d", +1),
    Feature("balance_volatility_cv", "cashflow", "numeric", "Coefficient of variation of daily balance", -1),
    Feature("days_with_zero_balance_30d", "cashflow", "numeric", "Days the wallet was empty", -1),
    Feature("net_cash_flow_ratio", "cashflow", "numeric", "(inflow - outflow) / inflow", +1),
    Feature("month_end_liquidity_ratio", "cashflow", "numeric", "Last-5-day balance vs monthly mean", +1),
    # -------------------------------------------------------- bills & spending
    Feature("utility_bill_count_90d", "obligations", "numeric", "Paybill utility payments in 90d", +1),
    Feature("utility_payment_punctuality", "obligations", "numeric", "1 = always on the same day of cycle, 0 = erratic", +1),
    Feature("school_fees_payment_flag", "obligations", "numeric", "Any school-fees paybill in 180d", +1),
    Feature("rent_payment_regularity", "obligations", "numeric", "Regularity of a recurring rent-sized outflow", +1),
    Feature("airtime_topup_count_30d", "spending", "numeric", "Airtime top-ups in 30d", 0),
    Feature("airtime_topup_avg_kes", "spending", "numeric", "Mean airtime top-up size, KES", +1),
    Feature("merchant_till_spend_ratio", "spending", "numeric", "Share of outflow spent at tills/merchants", +1),
    Feature("p2p_send_receive_ratio", "spending", "numeric", "P2P sent divided by P2P received", 0),
    Feature("savings_deposit_count_90d", "spending", "numeric", "Deposits into a savings/lock product, 90d", +1),
    Feature("savings_balance_ratio", "spending", "numeric", "Savings balance over monthly inflow", +1),
    Feature("insurance_premium_flag", "spending", "numeric", "Any recurring insurance premium", +1),
    # -------------------------------------------------------- risk behaviours
    Feature("betting_txn_count_30d", "behaviour", "numeric", "Paybill transactions to betting merchants, 30d", -1),
    Feature("betting_spend_ratio", "behaviour", "numeric", "Betting spend as a share of outflow", -1),
    Feature("betting_spend_trend_90d", "behaviour", "numeric", "Growth in betting spend over 90d", -1),
    Feature("night_txn_ratio", "behaviour", "numeric", "Share of transactions between 22:00 and 04:00", -1),
    Feature("weekend_txn_ratio", "behaviour", "numeric", "Share of transactions on Sat/Sun", 0),
    Feature("txn_time_entropy", "behaviour", "numeric", "Shannon entropy of transaction hour-of-day", 0),
    Feature("distinct_counterparties_30d", "behaviour", "numeric", "Unique counterparties transacted with", 0),
    Feature("new_counterparty_ratio", "behaviour", "numeric", "Share of counterparties never seen before", -1),
    Feature("max_single_outflow_ratio", "behaviour", "numeric", "Largest single outflow over monthly inflow", -1),
    Feature("failed_txn_ratio", "behaviour", "numeric", "Share of transactions failing for insufficient funds", -1),
    Feature("reversal_count_90d", "behaviour", "numeric", "Reversed transactions in 90d", -1),
    # ----------------------------------------------------------- credit history
    Feature("prior_loan_count", "credit", "numeric", "Digital loans taken before this application", 0),
    Feature("prior_ontime_repayment_ratio", "credit", "numeric", "Share of prior loans repaid on time", +1),
    Feature("days_since_last_loan", "credit", "numeric", "Days since the most recent digital loan", 0),
    Feature("overdraft_utilisation", "credit", "numeric", "Overdraft drawn over limit (Fuliza-style)", -1),
    Feature("active_lender_count", "credit", "numeric", "Distinct digital lenders currently owed", -1),
    Feature("loan_stacking_velocity_90d", "credit", "numeric", "New loans opened per month over 90d", -1),
    Feature("max_dpd_last_180d", "credit", "numeric", "Worst days-past-due observed in 180d", -1),
    Feature("total_credit_exposure_ratio", "credit", "numeric", "Outstanding balances over monthly inflow", -1),
    # ------------------------------------------------------- mobility / network
    Feature("distinct_agent_locations_90d", "mobility", "numeric", "Distinct agent tills used for cash in/out", 0),
    Feature("home_agent_concentration", "mobility", "numeric", "Share of agent activity at the single top agent", +1),
    Feature("travel_radius_km", "mobility", "numeric", "Radius covered by agent locations used", 0),
    Feature("contact_stability_score", "mobility", "numeric", "Stability of the top-10 counterparty set over 6m", +1),
)

FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

MODELLED_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.modelled)
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    f.name for f in FEATURES if f.modelled and f.dtype == "numeric"
)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    f.name for f in FEATURES if f.modelled and f.dtype == "categorical"
)
PROTECTED_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.protected)

#: Features whose sign a reviewer would insist on; used for monotonic
#: constraints in the GBM and asserted in the test suite.
MONOTONE_PRIORS: dict[str, int] = {
    f.name: f.risk_direction for f in FEATURES if f.modelled and f.risk_direction != 0
}

#: Features that are simply absent for a customer with no borrowing history.
#: Missingness here is informative and must survive into the model, not be
#: silently mean-imputed.
THIN_FILE_NULLABLE: tuple[str, ...] = (
    "prior_ontime_repayment_ratio",
    "days_since_last_loan",
    "max_dpd_last_180d",
    "income_growth_3m",
    "betting_spend_trend_90d",
    "contact_stability_score",
)

FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(f.family for f in FEATURES))


def feature_frame():
    """Return the feature contract as a DataFrame (used to render the docs)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "feature": f.name,
                "family": f.family,
                "type": f.dtype,
                "modelled": f.modelled,
                "protected": f.protected,
                "risk_direction": {1: "lower risk", -1: "higher risk", 0: "-"}[f.risk_direction],
                "description": f.description,
            }
            for f in FEATURES
        ]
    )
