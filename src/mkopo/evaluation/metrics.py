"""Discrimination, calibration and stability metrics.

Credit risk has its own vocabulary and it is not the one a Kaggle notebook uses.
A lending committee asks for Gini and KS, wants to see the rank ordering hold
across deciles, and treats a Population Stability Index above 0.25 as a reason to
stop using the model. All of that lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from mkopo.config import PSI_AMBER, PSI_GREEN

EPS = 1e-10


# --------------------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------------------


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))


def gini(y_true, y_score) -> float:
    """Somers' D, the coefficient every credit committee actually quotes."""
    return 2.0 * auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Maximum separation between the cumulative good and bad distributions."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    bads = np.cumsum(y_sorted) / max(y_sorted.sum(), EPS)
    goods = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), EPS)
    return float(np.max(np.abs(bads - goods)))


@dataclass
class Discrimination:
    auc: float
    gini: float
    ks: float
    n: int
    bad_rate: float

    def as_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "gini": round(self.gini, 4),
            "ks": round(self.ks, 4),
            "n": self.n,
            "bad_rate": round(self.bad_rate, 4),
        }


def discrimination(y_true, y_score) -> Discrimination:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return Discrimination(
        auc=auc(y_true, y_score),
        gini=gini(y_true, y_score),
        ks=ks_statistic(y_true, y_score),
        n=int(len(y_true)),
        bad_rate=float(y_true.mean()),
    )


# --------------------------------------------------------------------------------------
# Rank ordering
# --------------------------------------------------------------------------------------


def rank_ordering_table(y_true, y_score, *, n_bands: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Decile table: does risk actually fall monotonically as the score improves?

    ``ascending=False`` puts the riskiest band first, which is how a decline
    strategy is read.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "s": np.asarray(y_score, dtype=float)})
    frame = frame.sort_values("s", ascending=ascending).reset_index(drop=True)
    frame["band"] = pd.qcut(frame.index, n_bands, labels=range(1, n_bands + 1))

    agg = frame.groupby("band", observed=True).agg(
        count=("y", "size"), bads=("y", "sum"), min_score=("s", "min"), max_score=("s", "max")
    )
    agg["goods"] = agg["count"] - agg["bads"]
    agg["bad_rate"] = agg["bads"] / agg["count"]
    total_bads = max(agg["bads"].sum(), EPS)
    total = max(agg["count"].sum(), EPS)
    agg["cum_bad_capture"] = agg["bads"].cumsum() / total_bads
    agg["cum_population"] = agg["count"].cumsum() / total
    agg["lift"] = agg["bad_rate"] / (total_bads / total)
    return agg.reset_index()


def is_rank_ordered(table: pd.DataFrame, *, tolerance: float = 0.0) -> bool:
    """True when the bad rate falls monotonically across bands."""
    rates = table["bad_rate"].to_numpy()
    return bool((np.diff(rates) <= tolerance).all())


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


@dataclass
class Calibration:
    brier: float
    log_loss: float
    expected_bad_rate: float
    observed_bad_rate: float
    #: Ratio of predicted to actual defaults. 1.0 is perfectly calibrated;
    #: anything outside 0.9-1.1 means the pricing built on this model is wrong.
    calibration_ratio: float
    expected_calibration_error: float

    def as_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in self.__dict__.items()}


def calibration(y_true, y_prob, *, n_bins: int = 10) -> Calibration:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    expected = float(y_prob.mean())
    observed = float(y_true.mean())

    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    ece = float((grouped["n"] / len(frame) * (grouped["obs"] - grouped["pred"]).abs()).sum())

    return Calibration(
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        expected_bad_rate=expected,
        observed_bad_rate=observed,
        calibration_ratio=float(expected / max(observed, EPS)),
        expected_calibration_error=ece,
    )


def calibration_table(y_true, y_prob, *, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    out = frame.groupby("bin", observed=True).agg(
        count=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out.reset_index()


# --------------------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------------------


def psi(expected, actual, *, n_bins: int = 10, edges: np.ndarray | None = None) -> float:
    """Population Stability Index between a reference and a current sample.

    Convention: < 0.10 stable, 0.10-0.25 investigate, > 0.25 the population has
    moved far enough that the model should not be trusted unmonitored.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    if edges is None:
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) == 0:
        return 0.0

    e_counts = np.bincount(np.digitize(expected, edges, right=True), minlength=len(edges) + 1)
    a_counts = np.bincount(np.digitize(actual, edges, right=True), minlength=len(edges) + 1)
    e_share = np.clip(e_counts / e_counts.sum(), EPS, None)
    a_share = np.clip(a_counts / a_counts.sum(), EPS, None)
    return float(((a_share - e_share) * np.log(a_share / e_share)).sum())


def psi_verdict(value: float) -> str:
    if value < PSI_GREEN:
        return "stable"
    if value < PSI_AMBER:
        return "investigate"
    return "unstable"


def characteristic_stability(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], *, n_bins: int = 10
) -> pd.DataFrame:
    """Per-feature CSI - which inputs moved, not just the score."""
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref = reference[feature]
        cur = current[feature]
        if not np.issubdtype(ref.dtype, np.number):
            ref_share = ref.astype(str).value_counts(normalize=True)
            cur_share = cur.astype(str).value_counts(normalize=True)
            levels = ref_share.index.union(cur_share.index)
            e = np.clip(ref_share.reindex(levels).fillna(0).to_numpy(), EPS, None)
            a = np.clip(cur_share.reindex(levels).fillna(0).to_numpy(), EPS, None)
            value = float(((a - e) * np.log(a / e)).sum())
        else:
            value = psi(ref.to_numpy(), cur.to_numpy(), n_bins=n_bins)
        rows.append({"feature": feature, "csi": round(value, 4), "verdict": psi_verdict(value)})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Comparing two models
# --------------------------------------------------------------------------------------


@dataclass
class GiniDifference:
    """Paired bootstrap comparison of two models on the same sample."""

    gini_a: float
    gini_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_resamples: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def as_dict(self) -> dict:
        return {
            "gini_a": round(self.gini_a, 4),
            "gini_b": round(self.gini_b, 4),
            "difference": round(self.difference, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "p_two_sided": round(self.p_two_sided, 4),
            "significant_at_95": self.significant,
        }


def bootstrap_gini_difference(
    y_true, score_a, score_b, *, n_resamples: int = 2000, seed: int = 42
) -> GiniDifference:
    """Paired bootstrap on the Gini gap between two models.

    Two models scored on the same customers are not independent samples, so the
    resampling has to be paired. Without this, a headline like "the booster beats
    the scorecard by 0.005 Gini" is a statement about sampling noise dressed up
    as a modelling result.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed = gini(y_true, a) - gini(y_true, b)
    draws = np.empty(n_resamples, dtype=float)
    kept = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if y_sample.min() == y_sample.max():  # pragma: no cover - degenerate draw
            continue
        draws[kept] = gini(y_sample, a[idx]) - gini(y_sample, b[idx])
        kept += 1
    draws = draws[:kept]

    low, high = np.percentile(draws, [2.5, 97.5])
    # Proportion of resamples on the far side of zero, doubled.
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return GiniDifference(
        gini_a=float(gini(y_true, a)),
        gini_b=float(gini(y_true, b)),
        difference=float(observed),
        ci_low=float(low),
        ci_high=float(high),
        p_two_sided=float(min(p, 1.0)),
        n_resamples=int(kept),
    )


# --------------------------------------------------------------------------------------
# Bundled evaluation
# --------------------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Everything one would put in front of a model-approval meeting."""

    label: str
    discrimination: Discrimination
    calibration: Calibration
    rank_ordering: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_dict(self) -> dict:
        return {
            "split": self.label,
            **self.discrimination.as_dict(),
            **self.calibration.as_dict(),
            "rank_ordered": bool(is_rank_ordered(self.rank_ordering))
            if len(self.rank_ordering)
            else None,
        }


def evaluate(label: str, y_true, y_prob, *, n_bands: int = 10) -> Evaluation:
    return Evaluation(
        label=label,
        discrimination=discrimination(y_true, y_prob),
        calibration=calibration(y_true, y_prob),
        rank_ordering=rank_ordering_table(y_true, y_prob, n_bands=n_bands),
    )


def evaluation_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame([e.as_dict() for e in evaluations])
