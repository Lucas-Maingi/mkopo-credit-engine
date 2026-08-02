"""Synthetic mobile-money book generator.

Why this file is long
---------------------
The easy way to fake credit data is to sample 40 independent columns and then
define the target as a linear function of them. That produces a dataset where a
gradient booster hits 0.97 AUC, SHAP is meaningless and every downstream claim
about the model is worthless.

This generator instead builds a small **causal graph**:

    livelihood segment --+
    region / gender -----+--> latent income level, income stability, discipline
                         |          |
                         |          +--> observable cash-flow behaviour
                         |          +--> observable bill / savings behaviour
                         |          +--> observable risk behaviour (betting, stacking)
                         |                       |
                         +-----------------------+--> latent repayment capacity --> default

Observable features are *noisy consequences* of the latents, so a model can
recover only part of the signal. The achievable AUC lands in the 0.78–0.84 band
that real digital-lending scorecards actually live in.

Three further properties exist deliberately, because the rest of the repository
depends on them:

1. **Selection bias.** A crude legacy rules engine approves ~62% of applicants.
   Only approved loans have an observed outcome. This is the bias that
   :mod:`mkopo.models.reject_inference` corrects.
2. **A macro shock.** From month 13 of 18, incomes fall and betting rises, so the
   out-of-time window genuinely drifts and PSI monitoring has something to find.
3. **A counterfactual oracle.** The true outcome of *rejected* applicants is
   written to a separate file that no model may read. It exists so reject
   inference can be measured rather than asserted.

Structural inequality is included on purpose: women skew toward trading and
farming segments with lower recorded inflows, and remote regions have thinner
files. Gender and region are never model inputs — the point is that the
:mod:`mkopo.evaluation.fairness` audit must still find the proxy effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mkopo.config import LEGACY_APPROVAL_RATE, N_CUSTOMERS, RANDOM_SEED, RAW_DIR
from mkopo.data.schema import (
    DATE_COLUMN,
    ID_COLUMN,
    MODELLED_FEATURES,
    ORACLE_TARGET,
    TARGET,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Population structure
# --------------------------------------------------------------------------------------

SEGMENTS = ("salaried_formal", "salaried_informal", "small_trader", "boda_gig", "farmer", "student")
SEGMENT_WEIGHTS = np.array([0.16, 0.14, 0.24, 0.16, 0.20, 0.10])

REGIONS = ("Nairobi", "Central", "Rift Valley", "Western", "Nyanza", "Eastern", "Coast", "North Eastern")
REGION_WEIGHTS = np.array([0.19, 0.13, 0.23, 0.12, 0.12, 0.12, 0.07, 0.02])
#: Multiplier on median monthly inflow — urban wage premium, remote-region discount.
REGION_INCOME_MULT = np.array([1.55, 1.10, 0.95, 0.85, 0.85, 0.88, 1.00, 0.62])
#: Multiplier on network tenure — later mobile-money adoption in remote counties.
REGION_TENURE_MULT = np.array([1.20, 1.10, 0.98, 0.92, 0.92, 0.95, 1.05, 0.70])

DEVICE_TIERS = ("feature_phone", "entry_android", "mid_android", "premium")

#: median monthly inflow (KES) and inflow-regularity beta parameters per segment
SEGMENT_INCOME_MEDIAN = {
    "salaried_formal": 52_000.0,
    "salaried_informal": 26_000.0,
    "small_trader": 34_000.0,
    "boda_gig": 22_000.0,
    "farmer": 17_000.0,
    "student": 9_000.0,
}
SEGMENT_INCOME_SIGMA = {
    "salaried_formal": 0.55,
    "salaried_informal": 0.70,
    "small_trader": 0.85,
    "boda_gig": 0.65,
    "farmer": 0.90,
    "student": 0.75,
}
#: Beta(a, b) for income stability in [0, 1]; salaried is rhythmic, farming is seasonal.
SEGMENT_STABILITY_BETA = {
    "salaried_formal": (9.0, 2.0),
    "salaried_informal": (4.5, 3.0),
    "small_trader": (4.0, 3.5),
    "boda_gig": (3.5, 3.5),
    "farmer": (2.0, 4.5),
    "student": (2.5, 4.0),
}

N_MONTHS = 18
#: Months from which the macro squeeze applies (inclusive).
SHOCK_START_MONTH = 13
#: Last four months are held out as the out-of-time validation window.
OOT_START_MONTH = 14


@dataclass
class GeneratorConfig:
    n_customers: int = N_CUSTOMERS
    seed: int = RANDOM_SEED
    target_bad_rate: float = 0.132
    approval_rate: float = LEGACY_APPROVAL_RATE
    #: Irreducible noise in the default process — the part of repayment that no
    #: amount of transaction history can explain (illness, a family emergency, a
    #: failed harvest). Tuned so the achievable out-of-time AUC lands in the
    #: 0.78-0.84 band that real digital-lending scorecards live in; drop it and
    #: the dataset becomes a toy where everything scores 0.95.
    outcome_noise_sd: float = 3.2
    #: During the macro squeeze, households take idiosyncratic hits that no
    #: transaction history predicts — a lost job, a failed harvest, a hospital
    #: bill. This is what makes the out-of-time window genuinely harder rather
    #: than merely shifted, so model decay is real and monitoring has a job.
    shock_idiosyncratic_sd: float = 1.9
    reference_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-06-30"))


# --------------------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _z(x: np.ndarray) -> np.ndarray:
    """Standardise, robust to zero variance."""
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


def _solve_intercept(linear: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept that makes ``mean(sigmoid(linear + b))`` hit target."""
    lo, hi = -25.0, 25.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _sigmoid(linear + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _beta(rng: np.random.Generator, a: float, b: float, size: int) -> np.ndarray:
    return rng.beta(a, b, size=size)


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the through-the-door application book and its counterfactual oracle."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    log.info("generating %s applications (seed=%s)", f"{n:,}", cfg.seed)

    # ---------------------------------------------------------------- demographics
    seg_idx = rng.choice(len(SEGMENTS), size=n, p=SEGMENT_WEIGHTS)
    segment = np.array(SEGMENTS, dtype=object)[seg_idx]

    # Gender is not independent of livelihood: trading and farming skew female,
    # boda/gig work skews male. This is what creates a proxy pathway later.
    p_female = np.select(
        [
            segment == "small_trader",
            segment == "farmer",
            segment == "boda_gig",
            segment == "salaried_formal",
        ],
        [0.62, 0.55, 0.12, 0.42],
        default=0.50,
    )
    gender = np.where(rng.random(n) < p_female, "F", "M")

    reg_idx = rng.choice(len(REGIONS), size=n, p=REGION_WEIGHTS)
    region = np.array(REGIONS, dtype=object)[reg_idx]

    age = np.clip(rng.gamma(shape=7.0, scale=4.4, size=n) + 18, 18, 78).round().astype(int)
    age = np.where(segment == "student", np.clip(age * 0.55 + 8, 18, 30).round().astype(int), age)
    age_band = pd.cut(
        age, bins=[17, 24, 34, 44, 54, 100], labels=["18-24", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    # ------------------------------------------------------------------ latents
    # Financial discipline: the unobservable trait that drives saving, bill
    # punctuality, restraint on betting and, ultimately, repayment.
    discipline = rng.normal(0, 1, n)
    discipline += 0.18 * (gender == "F")  # documented microfinance regularity
    discipline += 0.012 * (age - 32) / 10
    discipline = _z(discipline)

    income_median = np.array([SEGMENT_INCOME_MEDIAN[s] for s in segment])
    income_sigma = np.array([SEGMENT_INCOME_SIGMA[s] for s in segment])
    income_median = income_median * REGION_INCOME_MULT[reg_idx]
    income_median = income_median * np.where(gender == "F", 0.88, 1.0)  # recorded-inflow gap

    stability_ab = np.array([SEGMENT_STABILITY_BETA[s] for s in segment])
    income_stability = rng.beta(stability_ab[:, 0], stability_ab[:, 1])
    income_stability = np.clip(income_stability + 0.06 * discipline, 0.01, 0.99)

    shock_exposure = np.clip(_beta(rng, 2.0, 3.0, n) + 0.25 * (segment == "farmer"), 0, 1)

    # --------------------------------------------------------------------- time
    month_idx = rng.integers(0, N_MONTHS, size=n)
    application_date = cfg.reference_date - pd.to_timedelta(
        (N_MONTHS - 1 - month_idx) * 30 + rng.integers(0, 30, size=n), unit="D"
    )
    in_shock = (month_idx >= SHOCK_START_MONTH).astype(float)
    shock_intensity = in_shock * (0.4 + 0.6 * (month_idx - SHOCK_START_MONTH) / 4.0)

    # --------------------------------------------------------------- identity
    tenure_days = np.clip(
        rng.gamma(shape=2.6, scale=520, size=n) * REGION_TENURE_MULT[reg_idx], 5, 6000
    )
    # A fifth of the book is deliberately thin-file.
    thin = rng.random(n) < 0.20
    tenure_days = np.where(thin, rng.uniform(5, 90, n), tenure_days).round()

    log_income = np.log(income_median) + income_sigma * rng.normal(0, 1, n)
    log_income -= 0.10 * shock_intensity * (1 + shock_exposure)  # macro squeeze
    monthly_inflow_kes = np.exp(log_income).round(-1)
    income_z = _z(np.log(monthly_inflow_kes))

    device_p = np.stack(
        [
            _sigmoid(1.4 - 1.5 * income_z),
            _sigmoid(0.9 - 0.4 * income_z),
            _sigmoid(-0.6 + 0.9 * income_z),
            _sigmoid(-2.2 + 1.4 * income_z),
        ],
        axis=1,
    )
    device_p = device_p / device_p.sum(axis=1, keepdims=True)
    device_tier = np.array(DEVICE_TIERS, dtype=object)[
        (device_p.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1).clip(0, 3)
    ]

    sim_swap_count_12m = rng.poisson(np.clip(0.22 - 0.06 * discipline, 0.02, None), n)

    # ------------------------------------------------------------------- income
    inflow_txn_count_30d = np.clip(
        rng.poisson(np.clip(4 + 26 * (1 - income_stability) + 6 * income_z, 1, None), n), 1, 400
    )
    inflow_txn_count_30d = np.where(
        np.isin(segment, ["salaried_formal", "salaried_informal"]),
        np.clip(rng.poisson(3.0, n) + 1, 1, 40),
        inflow_txn_count_30d,
    )
    inflow_regularity_score = np.clip(
        income_stability + rng.normal(0, 0.09, n) - 0.05 * shock_intensity, 0, 1
    ).round(4)
    p_salary = _sigmoid(-1.6 + 3.2 * (segment == "salaried_formal") + 1.1 * (segment == "salaried_informal") + 1.4 * (income_stability - 0.5))
    salary_deposit_flag = (rng.random(n) < p_salary).astype(int)
    inflow_concentration_hhi = np.clip(
        0.18 + 0.55 * income_stability + 0.22 * salary_deposit_flag + rng.normal(0, 0.08, n), 0.02, 1.0
    ).round(4)
    income_growth_3m = np.clip(
        rng.normal(0.015, 0.16, n) + 0.05 * discipline - 0.13 * shock_intensity * (1 + shock_exposure),
        -0.9,
        1.5,
    ).round(4)

    # ---------------------------------------------------------------- cash flow
    balance_ratio = np.clip(
        0.055 + 0.10 * _sigmoid(1.1 * discipline) + 0.05 * income_stability + rng.normal(0, 0.025, n),
        0.004,
        0.6,
    )
    avg_daily_balance_kes = (monthly_inflow_kes * balance_ratio).round(-1)
    balance_volatility_cv = np.clip(
        1.35 - 0.55 * income_stability - 0.16 * discipline + 0.18 * shock_intensity + rng.normal(0, 0.16, n),
        0.05,
        4.0,
    ).round(4)
    zero_lambda = np.clip(
        9.5 - 5.2 * income_stability - 2.4 * discipline - 1.6 * income_z + 2.2 * shock_intensity, 0.05, None
    )
    days_with_zero_balance_30d = np.clip(rng.poisson(zero_lambda, n), 0, 30)
    min_balance_kes = (
        avg_daily_balance_kes * np.clip(rng.beta(1.4, 5.0, n) * (1 - days_with_zero_balance_30d / 31), 0, 1)
    ).round(-1)
    net_cash_flow_ratio = np.clip(
        0.045 + 0.075 * discipline + 0.05 * income_stability - 0.06 * shock_intensity + rng.normal(0, 0.07, n),
        -0.65,
        0.65,
    ).round(4)
    month_end_liquidity_ratio = np.clip(
        0.72 + 0.30 * discipline + 0.20 * income_stability + rng.normal(0, 0.20, n), 0.02, 3.0
    ).round(4)

    # ------------------------------------------------------- bills, savings, spend
    bill_propensity = _sigmoid(-0.35 + 0.85 * discipline + 0.75 * income_z + 0.5 * (age - 30) / 15)
    utility_bill_count_90d = rng.poisson(np.clip(9.5 * bill_propensity, 0, None), n)
    utility_payment_punctuality = np.where(
        utility_bill_count_90d > 0,
        np.clip(0.45 + 0.28 * discipline + 0.18 * income_stability + rng.normal(0, 0.13, n), 0, 1),
        0.0,
    ).round(4)
    school_fees_payment_flag = (
        rng.random(n) < _sigmoid(-1.5 + 0.9 * income_z + 1.3 * ((age > 28) & (age < 55)))
    ).astype(int)
    rent_payment_regularity = np.clip(
        np.where(
            rng.random(n) < _sigmoid(-0.4 + 0.8 * income_z + 1.2 * (region == "Nairobi")),
            0.5 + 0.3 * discipline + 0.2 * income_stability + rng.normal(0, 0.14, n),
            0.0,
        ),
        0,
        1,
    ).round(4)
    airtime_topup_count_30d = np.clip(
        rng.poisson(np.clip(7.5 - 2.2 * income_z, 0.4, None), n), 0, 90
    )
    airtime_topup_avg_kes = np.clip(
        np.exp(3.15 + 0.55 * income_z + rng.normal(0, 0.45, n)), 5, 5000
    ).round(0)
    merchant_till_spend_ratio = np.clip(
        0.16 + 0.20 * income_z * 0.5 + 0.11 * discipline + 0.20 * (region == "Nairobi") + rng.normal(0, 0.10, n),
        0,
        0.95,
    ).round(4)
    p2p_send_receive_ratio = np.clip(
        np.exp(rng.normal(-0.05, 0.55, n) - 0.20 * net_cash_flow_ratio * 3), 0.02, 20
    ).round(4)
    savings_lambda = np.clip(4.2 * _sigmoid(1.3 * discipline + 0.55 * income_z - 0.4), 0, None)
    savings_deposit_count_90d = rng.poisson(savings_lambda, n)
    savings_balance_ratio = np.where(
        savings_deposit_count_90d > 0,
        np.clip(np.exp(rng.normal(-2.1, 0.85, n) + 0.55 * discipline), 0, 6),
        0.0,
    ).round(4)
    insurance_premium_flag = (
        rng.random(n) < _sigmoid(-2.6 + 1.1 * income_z + 0.75 * discipline)
    ).astype(int)

    # ------------------------------------------------------------ risk behaviour
    # Betting participation is high in this market; the *escalation* is the risk.
    p_bet = _sigmoid(
        -0.55
        - 0.85 * discipline
        + 0.55 * (gender == "M")
        + 0.60 * (age < 32)
        - 0.15 * income_z
        + 0.45 * shock_intensity
    )
    bets = rng.random(n) < p_bet
    betting_txn_count_30d = np.where(
        bets, np.clip(rng.negative_binomial(2.0, 0.16, n), 1, 300), 0
    )
    betting_spend_ratio = np.where(
        bets,
        np.clip(rng.beta(1.7, 11.0, n) * (1 - 0.35 * discipline) + 0.05 * shock_intensity, 0, 0.85),
        0.0,
    ).round(4)
    betting_spend_trend_90d = np.where(
        bets, np.clip(rng.normal(0.06, 0.34, n) - 0.16 * discipline + 0.22 * shock_intensity, -1, 3), 0.0
    ).round(4)

    night_txn_ratio = np.clip(
        0.085 - 0.045 * discipline + 0.075 * betting_spend_ratio * 3 + 0.05 * (age < 30) + rng.normal(0, 0.035, n),
        0,
        0.85,
    ).round(4)
    weekend_txn_ratio = np.clip(rng.normal(0.285, 0.075, n) + 0.05 * (segment == "small_trader"), 0, 1).round(4)
    txn_time_entropy = np.clip(
        2.05 + 0.55 * (1 - income_stability) + 0.25 * night_txn_ratio * 3 + rng.normal(0, 0.22, n), 0.4, 4.6
    ).round(4)
    distinct_counterparties_30d = np.clip(
        rng.poisson(np.clip(6 + 14 * (1 - inflow_concentration_hhi) + 4 * income_z, 1, None), n), 1, 300
    )
    new_counterparty_ratio = np.clip(
        0.30 - 0.10 * discipline - 0.00004 * tenure_days + rng.normal(0, 0.09, n), 0, 1
    ).round(4)
    max_single_outflow_ratio = np.clip(
        np.exp(rng.normal(-1.5, 0.62, n) + 0.28 * (1 - income_stability)), 0.005, 3.5
    ).round(4)
    failed_lambda = np.clip(
        0.55 + 0.11 * days_with_zero_balance_30d - 0.20 * discipline + 0.35 * shock_intensity, 0.01, None
    )
    failed_txn_ratio = np.clip(
        rng.gamma(2.0, failed_lambda / 2.0, n) / 22.0, 0, 0.9
    ).round(4)
    reversal_count_90d = rng.poisson(np.clip(0.35 + 0.9 * failed_txn_ratio * 4, 0.01, None), n)

    # ----------------------------------------------------------- credit history
    borrow_appetite = _sigmoid(
        -0.35 - 0.45 * discipline + 0.55 * (tenure_days > 400) + 0.35 * betting_spend_ratio * 3 - 0.25 * income_z
    )
    prior_loan_count = np.where(
        tenure_days < 90, rng.binomial(2, 0.25, n), rng.poisson(9.0 * borrow_appetite, n)
    )
    has_history = prior_loan_count > 0
    prior_ontime_repayment_ratio = np.where(
        has_history,
        np.clip(0.72 + 0.16 * discipline + 0.07 * income_stability + rng.normal(0, 0.13, n), 0, 1),
        np.nan,
    )
    days_since_last_loan = np.where(
        has_history, np.clip(rng.gamma(1.7, 26, n), 1, 900).round(), np.nan
    )
    overdraft_utilisation = np.clip(
        0.30 + 0.30 * borrow_appetite - 0.18 * discipline + 0.20 * shock_intensity + rng.normal(0, 0.16, n),
        0,
        1.6,
    ).round(4)
    active_lender_count = np.clip(
        rng.poisson(np.clip(2.4 * borrow_appetite + 0.6 * shock_intensity, 0, None), n) * has_history, 0, 9
    )
    loan_stacking_velocity_90d = np.clip(
        (active_lender_count / 3.0) * rng.gamma(2.0, 0.55, n), 0, 8
    ).round(4)
    max_dpd_last_180d = np.where(
        has_history,
        np.clip(
            np.where(
                rng.random(n) < np.clip(0.55 - 0.45 * prior_ontime_repayment_ratio, 0.02, 0.95),
                rng.gamma(1.6, 17, n),
                0.0,
            ),
            0,
            180,
        ).round(),
        np.nan,
    )
    total_credit_exposure_ratio = np.clip(
        (0.10 + 0.28 * borrow_appetite) * (1 + active_lender_count * 0.32) * rng.gamma(3.0, 0.34, n),
        0,
        4.0,
    ).round(4)

    # ---------------------------------------------------------------- mobility
    distinct_agent_locations_90d = np.clip(
        rng.poisson(np.clip(3.4 + 4.5 * (1 - inflow_concentration_hhi) + 1.6 * (region == "Nairobi"), 0.3, None), n),
        0,
        70,
    )
    home_agent_concentration = np.clip(
        0.68 - 0.035 * distinct_agent_locations_90d + 0.07 * discipline + rng.normal(0, 0.11, n), 0.02, 1.0
    ).round(4)
    travel_radius_km = np.clip(
        np.exp(rng.normal(1.9, 0.85, n) + 0.35 * (segment == "boda_gig") - 0.25 * (region == "Nairobi")), 0.2, 600
    ).round(2)
    contact_stability_score = np.clip(
        0.62 + 0.14 * discipline + 0.00003 * tenure_days - 0.30 * new_counterparty_ratio + rng.normal(0, 0.10, n),
        0,
        1,
    ).round(4)

    # ---------------------------------------------------------------------------
    # Latent repayment capacity -> default
    # ---------------------------------------------------------------------------
    capacity = (
        0.85 * discipline
        + 0.70 * _z(income_stability)
        + 0.45 * income_z
        - 0.55 * _z(total_credit_exposure_ratio)
    )

    linear = (
        -0.78 * _z(capacity)
        + 0.46 * _z(betting_spend_ratio)
        + 0.22 * _z(betting_spend_trend_90d)
        + 0.40 * _z(loan_stacking_velocity_90d)
        + 0.34 * _z(overdraft_utilisation)
        + 0.33 * _z(failed_txn_ratio)
        + 0.26 * _z(balance_volatility_cv)
        + 0.24 * _z(days_with_zero_balance_30d)
        - 0.30 * _z(np.nan_to_num(prior_ontime_repayment_ratio, nan=0.72))
        + 0.28 * _z(np.nan_to_num(max_dpd_last_180d, nan=0.0))
        - 0.22 * _z(savings_deposit_count_90d)
        - 0.20 * _z(utility_payment_punctuality)
        + 0.16 * _z(night_txn_ratio)
        - 0.14 * _z(month_end_liquidity_ratio)
        + 0.12 * _z(new_counterparty_ratio)
        + 0.55 * shock_intensity * (0.5 + shock_exposure)
        # Concept drift: under stress a liquidity buffer starts to matter more
        # than a clean betting record. The relationship itself moves, which is
        # exactly what a champion model cannot see coming.
        + shock_intensity
        * (
            0.45 * _z(overdraft_utilisation)
            - 0.40 * _z(savings_balance_ratio)
            - 0.35 * _z(month_end_liquidity_ratio)
        )
        + cfg.shock_idiosyncratic_sd * shock_intensity * shock_exposure * rng.normal(0, 1, n)
        + cfg.outcome_noise_sd * rng.normal(0, 1, n)
    )
    linear += _solve_intercept(linear, cfg.target_bad_rate)
    pd_true = _sigmoid(linear)
    true_default = (rng.random(n) < pd_true).astype(int)

    # ---------------------------------------------------------------------------
    # Legacy rules engine: crude, correlated with risk but far from optimal
    # ---------------------------------------------------------------------------
    legacy_score = (
        1.10 * (tenure_days > 180)
        + 0.95 * (monthly_inflow_kes > 15_000)
        + 0.80 * np.nan_to_num(prior_ontime_repayment_ratio, nan=0.35) * 1.4
        - 1.30 * (np.nan_to_num(max_dpd_last_180d, nan=0.0) > 30)
        - 0.75 * (betting_txn_count_30d > 12)
        - 0.85 * (days_with_zero_balance_30d > 12)
        + 0.60 * salary_deposit_flag
        - 0.55 * (active_lender_count >= 3)
        + rng.normal(0, 0.85, n)  # the operational noise every rules engine has
    )
    cutoff = np.quantile(legacy_score, 1 - cfg.approval_rate)
    approved = (legacy_score >= cutoff).astype(int)

    # ---------------------------------------------------------------------------
    # Assemble
    # ---------------------------------------------------------------------------
    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"MK{i:07d}" for i in range(n)],
            DATE_COLUMN: application_date,
            "application_month": month_idx,
            "gender": gender,
            "region": region,
            "age": age,
            "age_band": age_band,
            "segment": segment,
            "tenure_days": tenure_days,
            "device_tier": device_tier,
            "sim_swap_count_12m": sim_swap_count_12m,
            "monthly_inflow_kes": monthly_inflow_kes,
            "inflow_txn_count_30d": inflow_txn_count_30d,
            "inflow_regularity_score": inflow_regularity_score,
            "salary_deposit_flag": salary_deposit_flag,
            "inflow_concentration_hhi": inflow_concentration_hhi,
            "income_growth_3m": income_growth_3m,
            "avg_daily_balance_kes": avg_daily_balance_kes,
            "min_balance_kes": min_balance_kes,
            "balance_volatility_cv": balance_volatility_cv,
            "days_with_zero_balance_30d": days_with_zero_balance_30d,
            "net_cash_flow_ratio": net_cash_flow_ratio,
            "month_end_liquidity_ratio": month_end_liquidity_ratio,
            "utility_bill_count_90d": utility_bill_count_90d,
            "utility_payment_punctuality": utility_payment_punctuality,
            "school_fees_payment_flag": school_fees_payment_flag,
            "rent_payment_regularity": rent_payment_regularity,
            "airtime_topup_count_30d": airtime_topup_count_30d,
            "airtime_topup_avg_kes": airtime_topup_avg_kes,
            "merchant_till_spend_ratio": merchant_till_spend_ratio,
            "p2p_send_receive_ratio": p2p_send_receive_ratio,
            "savings_deposit_count_90d": savings_deposit_count_90d,
            "savings_balance_ratio": savings_balance_ratio,
            "insurance_premium_flag": insurance_premium_flag,
            "betting_txn_count_30d": betting_txn_count_30d,
            "betting_spend_ratio": betting_spend_ratio,
            "betting_spend_trend_90d": betting_spend_trend_90d,
            "night_txn_ratio": night_txn_ratio,
            "weekend_txn_ratio": weekend_txn_ratio,
            "txn_time_entropy": txn_time_entropy,
            "distinct_counterparties_30d": distinct_counterparties_30d,
            "new_counterparty_ratio": new_counterparty_ratio,
            "max_single_outflow_ratio": max_single_outflow_ratio,
            "failed_txn_ratio": failed_txn_ratio,
            "reversal_count_90d": reversal_count_90d,
            "prior_loan_count": prior_loan_count,
            "prior_ontime_repayment_ratio": prior_ontime_repayment_ratio,
            "days_since_last_loan": days_since_last_loan,
            "overdraft_utilisation": overdraft_utilisation,
            "active_lender_count": active_lender_count,
            "loan_stacking_velocity_90d": loan_stacking_velocity_90d,
            "max_dpd_last_180d": max_dpd_last_180d,
            "total_credit_exposure_ratio": total_credit_exposure_ratio,
            "distinct_agent_locations_90d": distinct_agent_locations_90d,
            "home_agent_concentration": home_agent_concentration,
            "travel_radius_km": travel_radius_km,
            "contact_stability_score": contact_stability_score,
            "approved": approved,
        }
    )

    # Thin files genuinely lack long-window features. Missingness is informative
    # and must not be quietly imputed away downstream.
    thin_mask = frame["tenure_days"] < 90
    for col in ("income_growth_3m", "betting_spend_trend_90d", "contact_stability_score"):
        frame.loc[thin_mask, col] = np.nan

    # The observed target: only approved applications ever repay or default.
    frame[TARGET] = np.where(approved == 1, true_default, np.nan)

    oracle = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN],
            ORACLE_TARGET: true_default,
            "pd_true": pd_true.round(6),
            "latent_capacity": capacity.round(4),
            "approved": approved,
        }
    )

    _log_summary(frame, oracle)
    return frame, oracle


def _log_summary(frame: pd.DataFrame, oracle: pd.DataFrame) -> None:
    approved = frame["approved"] == 1
    log.info("applications: %s", f"{len(frame):,}")
    log.info("modelled features: %d", len(MODELLED_FEATURES))
    log.info("approval rate: %.1f%%", 100 * approved.mean())
    log.info("observed bad rate (approved book): %.2f%%", 100 * frame.loc[approved, TARGET].mean())
    log.info(
        "counterfactual bad rate (rejected): %.2f%%",
        100 * oracle.loc[oracle["approved"] == 0, ORACLE_TARGET].mean(),
    )
    log.info("through-the-door bad rate: %.2f%%", 100 * oracle[ORACLE_TARGET].mean())
    log.info("thin-file share: %.1f%%", 100 * (frame["tenure_days"] < 90).mean())
    oot = frame["application_month"] >= OOT_START_MONTH
    log.info("out-of-time window: %s rows (%.1f%%)", f"{oot.sum():,}", 100 * oot.mean())


def main(_argv: list[str] | None = None) -> int:
    cfg = GeneratorConfig()
    frame, oracle = generate(cfg)

    app_path = RAW_DIR / "applications.parquet"
    oracle_path = RAW_DIR / "oracle.parquet"
    frame.to_parquet(app_path, index=False)
    oracle.to_parquet(oracle_path, index=False)
    log.info("wrote %s and %s", app_path, oracle_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
