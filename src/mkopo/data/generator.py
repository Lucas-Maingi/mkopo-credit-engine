"""Synthetic mobile-money book generator.

Why this file is long
---------------------
The easy way to fake credit data is to sample 40 independent columns and then
define the target as a linear function of them. That produces a dataset where a
gradient booster hits 0.97 AUC, every feature has an Information Value above 1.0,
SHAP is meaningless and every downstream claim about the model is worthless.

This generator instead builds a small causal graph:

    livelihood segment --+
    region / gender -----+--> latent traits: income level, income stability,
                         |    financial discipline, liquidity stress,
                         |    gambling propensity, credit hunger
                         |          |
                         |          +--> observable cash-flow behaviour
                         |          +--> observable bill / savings behaviour
                         |          +--> observable risk behaviour
                         |          +--> observable credit history
                         |                       |
                         +-----------------------+--> default

The critical property is that **default depends on the latent traits, never on
the observed columns**. Each observed column is a noisy measurement of a trait,
carrying roughly as much idiosyncratic noise as signal. A single feature is
therefore weak-to-medium on its own (Information Value in the 0.05-0.30 band a
credit reviewer expects), while a model that pools thirty of them recovers the
traits well enough to reach the 0.78-0.84 AUC band that real digital-lending
scorecards live in. Getting this relationship right is the whole point: a
dataset where one feature has IV above 1.0 is a dataset where nothing
downstream can be believed.

Three further properties exist deliberately, because the rest of the repository
depends on them:

1. **Selection bias.** A crude legacy rules engine approves ~62% of applicants.
   Only approved loans have an observed outcome. This is the bias that
   :mod:`mkopo.models.reject_inference` corrects.
2. **A macro shock.** From month 13 of 18 incomes fall and betting rises, and the
   relationships themselves move: under stress a liquidity buffer starts to
   matter more than a clean betting record. So the out-of-time window is
   genuinely harder, model decay is real, and PSI monitoring has a job.
3. **A counterfactual oracle.** The true outcome of *rejected* applicants is
   written to a separate file that no model may read. It exists so reject
   inference can be measured rather than asserted.

Structural inequality is included on purpose: women skew toward trading and
farming segments with lower recorded inflows, and remote regions have thinner
files and later mobile-money adoption. Gender and region are never model inputs
- the point is that the :mod:`mkopo.evaluation.fairness` audit must still find
the proxy effect.
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
    ORACLE_TARGET,
    RAW_FEATURES,
    TARGET,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Population structure
# --------------------------------------------------------------------------------------

SEGMENTS = ("salaried_formal", "salaried_informal", "small_trader", "boda_gig", "farmer", "student")
SEGMENT_WEIGHTS = np.array([0.16, 0.14, 0.24, 0.16, 0.20, 0.10])

REGIONS = (
    "Nairobi",
    "Central",
    "Rift Valley",
    "Western",
    "Nyanza",
    "Eastern",
    "Coast",
    "North Eastern",
)
REGION_WEIGHTS = np.array([0.19, 0.13, 0.23, 0.12, 0.12, 0.12, 0.07, 0.02])
#: Multiplier on median monthly inflow: urban wage premium, remote-region discount.
REGION_INCOME_MULT = np.array([1.55, 1.10, 0.95, 0.85, 0.85, 0.88, 1.00, 0.62])
#: Multiplier on network tenure: later mobile-money adoption in remote counties.
REGION_TENURE_MULT = np.array([1.20, 1.10, 0.98, 0.92, 0.92, 0.95, 1.05, 0.70])

DEVICE_TIERS = ("feature_phone", "entry_android", "mid_android", "premium")

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
#: Beta(a, b) for income stability in [0, 1]; salaried is rhythmic, farming seasonal.
SEGMENT_STABILITY_BETA = {
    "salaried_formal": (9.0, 2.0),
    "salaried_informal": (4.5, 3.0),
    "small_trader": (4.0, 3.5),
    "boda_gig": (3.5, 3.5),
    "farmer": (2.0, 4.5),
    "student": (2.5, 4.0),
}

N_MONTHS = 18
#: Month from which the macro squeeze applies (inclusive).
SHOCK_START_MONTH = 13
#: Last four months are held out as the out-of-time validation window.
OOT_START_MONTH = 14


@dataclass
class GeneratorConfig:
    n_customers: int = N_CUSTOMERS
    seed: int = RANDOM_SEED
    target_bad_rate: float = 0.132
    approval_rate: float = LEGACY_APPROVAL_RATE
    #: Irreducible noise in the default process: the part of repayment no amount
    #: of transaction history can explain (illness, a family emergency, a failed
    #: harvest). Tuned together with the per-feature measurement noise so the
    #: achievable out-of-time AUC lands in the 0.78-0.84 band.
    outcome_noise_sd: float = 1.25
    #: Under the macro squeeze households take idiosyncratic hits nothing
    #: predicts. This makes the out-of-time window genuinely harder rather than
    #: merely shifted, so model decay is real and monitoring has something to do.
    shock_idiosyncratic_sd: float = 1.15
    #: Global multiplier on how noisily each latent trait is observed. Raising it
    #: weakens every feature at once; it is the single dial controlling how much
    #: of the truth the transaction log actually reveals.
    measurement_noise: float = 1.15
    reference_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-06-30"))


# --------------------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _z(x: np.ndarray) -> np.ndarray:
    """Standardise, robust to zero variance."""
    x = np.asarray(x, dtype=float)
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


class _Observer:
    """Turns a latent trait into a noisy observable.

    ``noise`` is the ratio of idiosyncratic to systematic variation. At 1.0 the
    column is roughly half signal; at 1.8 it is a weak hint. Nothing in this book
    is observed cleanly, because nothing in a transaction log ever is.
    """

    def __init__(self, rng: np.random.Generator, n: int, scale: float = 1.0) -> None:
        self.rng = rng
        self.n = n
        self.scale = scale

    def __call__(self, signal: np.ndarray, noise: float = 1.0) -> np.ndarray:
        return _z(_z(signal) + self.scale * noise * self.rng.normal(0, 1, self.n))


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the through-the-door application book and its counterfactual oracle."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    obs = _Observer(rng, n, cfg.measurement_noise)
    log.info("generating %s applications (seed=%s)", f"{n:,}", cfg.seed)

    # ------------------------------------------------------------- demographics
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

    # ---------------------------------------------------------------- the clock
    month_idx = rng.integers(0, N_MONTHS, size=n)
    application_date = cfg.reference_date - pd.to_timedelta(
        (N_MONTHS - 1 - month_idx) * 30 + rng.integers(0, 30, size=n), unit="D"
    )
    in_shock = (month_idx >= SHOCK_START_MONTH).astype(float)
    shock = in_shock * (0.4 + 0.6 * (month_idx - SHOCK_START_MONTH) / 4.0)

    # ============================== LATENT TRAITS ==============================
    # Nothing in this block is written to the application file. Default depends
    # on these and only these; every column the model sees is a noisy shadow.

    #: Financial discipline: the unobservable trait behind saving, bill
    #: punctuality, restraint on betting and, ultimately, repayment.
    discipline = _z(rng.normal(0, 1, n) + 0.18 * (gender == "F") + 0.012 * (age - 32) / 10)

    income_median = np.array([SEGMENT_INCOME_MEDIAN[s] for s in segment])
    income_sigma = np.array([SEGMENT_INCOME_SIGMA[s] for s in segment])
    income_median = income_median * REGION_INCOME_MULT[reg_idx]
    income_median = income_median * np.where(gender == "F", 0.88, 1.0)  # recorded-inflow gap

    stability_ab = np.array([SEGMENT_STABILITY_BETA[s] for s in segment])
    income_stability = np.clip(
        rng.beta(stability_ab[:, 0], stability_ab[:, 1]) + 0.06 * discipline, 0.01, 0.99
    )
    stability_z = _z(income_stability)

    shock_exposure = np.clip(rng.beta(2.0, 3.0, n) + 0.25 * (segment == "farmer"), 0, 1)

    log_income = np.log(income_median) + income_sigma * rng.normal(0, 1, n)
    log_income = log_income - 0.10 * shock * (1 + shock_exposure)
    monthly_inflow_kes = np.exp(log_income).round(-1)
    income_z = _z(log_income)

    #: Liquidity stress: living hand to mouth. Drives empty wallets, bounced
    #: transactions, volatile balances and overdraft reliance.
    stress = _z(
        -0.50 * discipline
        - 0.45 * stability_z
        - 0.40 * income_z
        + 0.55 * shock * (1 + shock_exposure)
        + 0.95 * rng.normal(0, 1, n)
    )

    #: Gambling propensity. Participation is common in this market; escalation is
    #: what carries the risk.
    gambling = _z(
        -0.45 * discipline
        + 0.30 * (gender == "M")
        + 0.28 * (age < 32)
        + 0.20 * stress
        + 0.30 * shock
        + 1.00 * rng.normal(0, 1, n)
    )

    #: Credit hunger: appetite to stack short-term loans across several lenders.
    hunger = _z(
        -0.35 * discipline
        + 0.45 * stress
        + 0.25 * gambling
        - 0.15 * income_z
        + 0.90 * rng.normal(0, 1, n)
    )

    # ============================ OBSERVED COLUMNS =============================

    # ----------------------------------------------------------------- identity
    tenure_days = np.clip(
        rng.gamma(shape=2.6, scale=520, size=n) * REGION_TENURE_MULT[reg_idx], 5, 6000
    )
    thin = rng.random(n) < 0.20
    tenure_days = np.where(thin, rng.uniform(5, 90, n), tenure_days).round()

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
    sim_swap_count_12m = rng.poisson(np.clip(0.22 - 0.05 * obs(-discipline, 1.8), 0.02, None), n)

    # ------------------------------------------------------------------- income
    inflow_txn_count_30d = np.where(
        np.isin(segment, ["salaried_formal", "salaried_informal"]),
        np.clip(rng.poisson(3.0, n) + 1, 1, 40),
        np.clip(rng.poisson(np.clip(6 + 14 * _sigmoid(obs(-stability_z, 1.3)), 1, None), n), 1, 400),
    )
    inflow_regularity_score = np.clip(
        0.52 + 0.20 * obs(stability_z, 1.15) - 0.04 * shock, 0, 1
    ).round(4)
    salary_deposit_flag = (
        rng.random(n)
        < _sigmoid(
            -1.5
            + 2.9 * (segment == "salaried_formal")
            + 1.0 * (segment == "salaried_informal")
            + 0.5 * obs(stability_z, 1.6)
        )
    ).astype(int)
    inflow_concentration_hhi = np.clip(
        0.42 + 0.16 * obs(stability_z, 1.5) + 0.14 * salary_deposit_flag, 0.02, 1.0
    ).round(4)
    income_growth_3m = np.clip(
        0.015 + 0.075 * obs(discipline, 2.0) - 0.13 * shock * (1 + shock_exposure), -0.9, 1.5
    ).round(4)

    # ---------------------------------------------------------------- cash flow
    balance_ratio = np.clip(np.exp(-2.85 + 0.42 * obs(-stress, 1.25)), 0.004, 0.6)
    avg_daily_balance_kes = (monthly_inflow_kes * balance_ratio).round(-1)
    balance_volatility_cv = np.clip(np.exp(0.02 + 0.34 * obs(stress, 1.35)), 0.05, 4.0).round(4)
    days_with_zero_balance_30d = np.clip(
        rng.poisson(np.clip(np.exp(0.75 + 0.62 * obs(stress, 1.30)), 0.05, None), n), 0, 30
    )
    min_balance_kes = (
        avg_daily_balance_kes
        * np.clip(rng.beta(1.4, 5.0, n) * (1 - days_with_zero_balance_30d / 31), 0, 1)
    ).round(-1)
    net_cash_flow_ratio = np.clip(0.045 + 0.085 * obs(-stress, 1.45), -0.65, 0.65).round(4)
    month_end_liquidity_ratio = np.clip(
        np.exp(-0.16 + 0.30 * obs(-stress, 1.40)), 0.02, 3.0
    ).round(4)

    # ----------------------------------------------------- bills, savings, spend
    bill_propensity = _sigmoid(-0.35 + 0.75 * obs(discipline, 1.5) + 0.70 * income_z)
    utility_bill_count_90d = rng.poisson(np.clip(9.5 * bill_propensity, 0, None), n)
    utility_payment_punctuality = np.where(
        utility_bill_count_90d > 0,
        np.clip(0.55 + 0.17 * obs(discipline, 1.55), 0, 1),
        0.0,
    ).round(4)
    school_fees_payment_flag = (
        rng.random(n) < _sigmoid(-1.5 + 0.9 * income_z + 1.3 * ((age > 28) & (age < 55)))
    ).astype(int)
    rent_payment_regularity = np.clip(
        np.where(
            rng.random(n) < _sigmoid(-0.4 + 0.8 * income_z + 1.2 * (region == "Nairobi")),
            0.58 + 0.19 * obs(discipline, 1.65),
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
        0.22 + 0.075 * obs(discipline, 1.7) + 0.06 * income_z + 0.18 * (region == "Nairobi"),
        0,
        0.95,
    ).round(4)
    p2p_send_receive_ratio = np.clip(
        np.exp(rng.normal(-0.05, 0.55, n) + 0.10 * obs(stress, 1.8)), 0.02, 20
    ).round(4)
    savings_deposit_count_90d = rng.poisson(
        np.clip(np.exp(0.55 + 0.50 * obs(discipline, 1.35) + 0.30 * income_z), 0, None), n
    )
    savings_balance_ratio = np.where(
        savings_deposit_count_90d > 0,
        np.clip(np.exp(-2.1 + 0.45 * obs(discipline, 1.5) + rng.normal(0, 0.7, n)), 0, 6),
        0.0,
    ).round(4)
    insurance_premium_flag = (
        rng.random(n) < _sigmoid(-2.6 + 1.1 * income_z + 0.55 * obs(discipline, 1.7))
    ).astype(int)

    # ------------------------------------------------------------ risk behaviour
    p_bet = _sigmoid(-0.30 + 0.95 * obs(gambling, 1.10))
    bets = rng.random(n) < p_bet
    betting_txn_count_30d = np.where(bets, np.clip(rng.negative_binomial(2.0, 0.16, n), 1, 300), 0)
    betting_spend_ratio = np.where(
        bets,
        np.clip(np.exp(-3.1 + 0.55 * obs(gambling, 1.25) + rng.normal(0, 0.5, n)), 0, 0.85),
        0.0,
    ).round(4)
    betting_spend_trend_90d = np.where(
        bets, np.clip(0.05 + 0.22 * obs(gambling, 1.7) + 0.20 * shock, -1, 3), 0.0
    ).round(4)
    night_txn_ratio = np.clip(
        0.10 + 0.035 * obs(gambling - 0.5 * discipline, 1.55) + 0.04 * (age < 30), 0, 0.85
    ).round(4)
    weekend_txn_ratio = np.clip(
        rng.normal(0.285, 0.075, n) + 0.05 * (segment == "small_trader"), 0, 1
    ).round(4)
    txn_time_entropy = np.clip(
        2.25 + 0.30 * obs(-stability_z + 0.4 * gambling, 1.6) + rng.normal(0, 0.15, n), 0.4, 4.6
    ).round(4)
    distinct_counterparties_30d = np.clip(
        rng.poisson(np.clip(9 + 8 * _sigmoid(obs(-stability_z, 1.6)) + 4 * income_z, 1, None), n),
        1,
        300,
    )
    new_counterparty_ratio = np.clip(
        0.28 + 0.075 * obs(-discipline, 1.6) - 0.00003 * tenure_days, 0, 1
    ).round(4)
    max_single_outflow_ratio = np.clip(
        np.exp(-1.5 + 0.22 * obs(stress, 1.9) + rng.normal(0, 0.45, n)), 0.005, 3.5
    ).round(4)
    failed_txn_ratio = np.clip(
        np.exp(-3.5 + 0.55 * obs(stress, 1.30) + rng.normal(0, 0.35, n)), 0, 0.9
    ).round(4)
    reversal_count_90d = rng.poisson(np.clip(0.35 + 6.0 * failed_txn_ratio, 0.01, None), n)

    # ----------------------------------------------------------- credit history
    appetite = _sigmoid(0.35 * obs(hunger, 1.30) + 0.55 * (tenure_days > 400))
    prior_loan_count = np.where(
        tenure_days < 90, rng.binomial(2, 0.25, n), rng.poisson(9.0 * appetite, n)
    )
    has_history = prior_loan_count > 0
    prior_ontime_repayment_ratio = np.where(
        has_history,
        np.clip(0.78 + 0.115 * obs(discipline - 0.4 * stress, 1.30), 0, 1),
        np.nan,
    )
    days_since_last_loan = np.where(
        has_history, np.clip(rng.gamma(1.7, 26, n), 1, 900).round(), np.nan
    )
    overdraft_utilisation = np.clip(0.42 + 0.155 * obs(stress + 0.5 * hunger, 1.30), 0, 1.6).round(4)
    active_lender_count = np.clip(
        rng.poisson(np.clip(np.exp(-0.35 + 0.55 * obs(hunger, 1.35)), 0, None), n) * has_history,
        0,
        9,
    )
    loan_stacking_velocity_90d = np.clip(
        (active_lender_count / 3.0) * rng.gamma(2.0, 0.55, n), 0, 8
    ).round(4)
    max_dpd_last_180d = np.where(
        has_history,
        np.clip(
            np.where(
                rng.random(n) < np.clip(0.28 + 0.16 * obs(stress, 1.45), 0.02, 0.95),
                rng.gamma(1.6, 17, n),
                0.0,
            ),
            0,
            180,
        ).round(),
        np.nan,
    )
    total_credit_exposure_ratio = np.clip(
        np.exp(-1.55 + 0.42 * obs(hunger, 1.40) + rng.normal(0, 0.45, n)), 0, 4.0
    ).round(4)

    # ------------------------------------------------------- mobility / network
    distinct_agent_locations_90d = np.clip(
        rng.poisson(
            np.clip(
                4.0 + 3.0 * _sigmoid(obs(-stability_z, 1.8)) + 1.6 * (region == "Nairobi"),
                0.3,
                None,
            ),
            n,
        ),
        0,
        70,
    )
    home_agent_concentration = np.clip(
        0.62 - 0.030 * distinct_agent_locations_90d + 0.05 * obs(discipline, 1.8), 0.02, 1.0
    ).round(4)
    travel_radius_km = np.clip(
        np.exp(
            rng.normal(1.9, 0.85, n) + 0.35 * (segment == "boda_gig") - 0.25 * (region == "Nairobi")
        ),
        0.2,
        600,
    ).round(2)
    contact_stability_score = np.clip(
        0.60 + 0.10 * obs(discipline, 1.7) - 0.20 * new_counterparty_ratio, 0, 1
    ).round(4)

    # ============================== THE OUTCOME ================================
    # Depends on latent traits only. No observed column appears here, so no
    # feature can act as a shortcut to the answer.
    capacity = _z(0.55 * discipline + 0.50 * stability_z + 0.45 * income_z - 0.40 * hunger)
    linear = (
        -0.95 * capacity
        + 0.60 * stress
        + 0.42 * gambling
        + 0.38 * hunger
        - 0.28 * discipline
        # Concept drift: under stress a liquidity buffer starts to matter more
        # than a clean betting record. The relationship itself moves, which is
        # exactly what a deployed champion cannot see coming.
        + shock * (0.55 * stress - 0.30 * gambling)
        + cfg.shock_idiosyncratic_sd * shock * shock_exposure * rng.normal(0, 1, n)
        + cfg.outcome_noise_sd * rng.normal(0, 1, n)
    )
    linear = linear + _solve_intercept(linear, cfg.target_bad_rate)
    pd_true = _sigmoid(linear)
    true_default = (rng.random(n) < pd_true).astype(int)

    # ---------------------------------------------------------------------------
    # Legacy rules engine: crude, correlated with risk but far from optimal
    # ---------------------------------------------------------------------------
    legacy_score = (
        1.10 * (tenure_days > 180)
        + 0.95 * (monthly_inflow_kes > 15_000)
        + 1.10 * np.nan_to_num(prior_ontime_repayment_ratio, nan=0.35)
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
            "latent_stress": stress.round(4),
            "approved": approved,
        }
    )

    _validate(frame)
    _log_summary(frame, oracle)
    return frame, oracle


def _validate(frame: pd.DataFrame) -> None:
    """Fail loudly rather than silently shipping a broken book."""
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"generator did not produce declared features: {missing}")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise RuntimeError("generator produced infinite values")


def _log_summary(frame: pd.DataFrame, oracle: pd.DataFrame) -> None:
    approved = frame["approved"] == 1
    log.info("applications: %s", f"{len(frame):,}")
    log.info("raw features: %d", len(RAW_FEATURES))
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
    frame, oracle = generate(GeneratorConfig())
    app_path = RAW_DIR / "applications.parquet"
    oracle_path = RAW_DIR / "oracle.parquet"
    frame.to_parquet(app_path, index=False)
    oracle.to_parquet(oracle_path, index=False)
    log.info("wrote %s and %s", app_path, oracle_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
