"""The interpretable logistic scorecard.

This is the model a credit committee can read line by line, and it is the
benchmark the gradient booster has to beat. Publishing a champion without a
challenger of this kind is how teams end up deploying a 400-tree ensemble that
buys three points of Gini over logistic regression and costs them the ability to
explain a single decline.

The build follows standard scorecard practice rather than generic ML practice:

1. admit only features with Information Value above a floor;
2. prune correlated pairs, keeping the higher-IV member, because a scorecard with
   two near-duplicate characteristics double-counts the same behaviour;
3. fit on weight-of-evidence, not raw values;
4. **drop wrong-signed coefficients one at a time.** A positive coefficient on a
   WOE feature means the fitted model says better evidence implies more risk.
   That is multicollinearity talking, and no amount of Gini excuses shipping it;
5. report Wald p-values and variance inflation factors, because "is this
   coefficient real?" is the first question anyone competent will ask;
6. allocate points so the total is a score on the published scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mkopo.config import MODEL_DIR, SCORECARD_SCALE, ScorecardScale
from mkopo.data.schema import TARGET
from mkopo.evaluation.metrics import Evaluation, evaluate, evaluation_frame, rank_ordering_table
from mkopo.features.pipeline import BINNER_PATH
from mkopo.models.scaling import probability_to_score
from mkopo.reporting import markdown_table, write_metrics, write_report

log = logging.getLogger(__name__)

SCORECARD_PATH = MODEL_DIR / "scorecard.joblib"
PROCESSED = "data/processed"

MIN_IV = 0.02
MAX_ABS_CORRELATION = 0.75
EPS = 1e-12


@dataclass
class ScorecardModel:
    """A fitted WOE scorecard plus everything needed to audit and serve it."""

    features: list[str]
    coefficients: dict[str, float]
    intercept: float
    scale: ScorecardScale = field(default_factory=lambda: SCORECARD_SCALE)
    #: feature -> bin label -> points
    points: dict[str, dict[str, float]] = field(default_factory=dict)
    base_points: float = 0.0
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    dropped: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ scoring

    def log_odds(self, woe_frame: pd.DataFrame) -> np.ndarray:
        contrib = np.zeros(len(woe_frame), dtype=float)
        for feature, coef in self.coefficients.items():
            contrib += coef * woe_frame[f"woe_{feature}"].to_numpy(dtype=float)
        return contrib + self.intercept

    def predict_proba(self, woe_frame: pd.DataFrame) -> np.ndarray:
        z = self.log_odds(woe_frame)
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return probability_to_score(self.predict_proba(woe_frame), self.scale)

    def points_table(self) -> pd.DataFrame:
        rows = [
            {"feature": feature, "bin": label, "points": round(pts, 1)}
            for feature, mapping in self.points.items()
            for label, pts in mapping.items()
        ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def _prune_correlated(
    woe: pd.DataFrame, candidates: list[str], iv: pd.Series, threshold: float
) -> tuple[list[str], list[tuple[str, str]]]:
    """Greedily drop the lower-IV member of any highly correlated pair."""
    ordered = sorted(candidates, key=lambda f: -iv.get(f, 0.0))
    corr = woe[[f"woe_{f}" for f in ordered]].corr().abs()
    corr.index = ordered
    corr.columns = ordered

    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for feature in ordered:
        clash = next((k for k in kept if corr.loc[feature, k] > threshold), None)
        if clash is None:
            kept.append(feature)
        else:
            dropped.append((feature, f"|r| = {corr.loc[feature, clash]:.2f} with {clash}"))
    return kept, dropped


def _wald_statistics(X: np.ndarray, probability: np.ndarray, coefficients: np.ndarray) -> pd.DataFrame:
    """Standard errors and Wald p-values from the logistic information matrix."""
    from scipy import stats

    design = np.column_stack([np.ones(len(X)), X])
    w = probability * (1.0 - probability)
    fisher = design.T @ (design * w[:, None])
    try:
        covariance = np.linalg.inv(fisher + np.eye(len(fisher)) * 1e-8)
        se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
        se = np.full(design.shape[1], np.nan)

    full = np.concatenate([[coefficients[0]], coefficients[1:]])
    with np.errstate(divide="ignore", invalid="ignore"):
        wald = full / se
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(wald)))
    return pd.DataFrame({"std_error": se[1:], "wald_z": wald[1:], "p_value": p_values[1:]})


def _vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factors from the inverse correlation matrix."""
    corr = np.corrcoef(X, rowvar=False)
    try:
        return np.diag(np.linalg.inv(corr + np.eye(len(corr)) * 1e-8))
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.full(X.shape[1], np.nan)


def fit_scorecard(
    train_woe: pd.DataFrame,
    binner,
    *,
    min_iv: float = MIN_IV,
    max_correlation: float = MAX_ABS_CORRELATION,
    scale: ScorecardScale = SCORECARD_SCALE,
) -> ScorecardModel:
    """Fit the scorecard, eliminating wrong-signed coefficients iteratively."""
    iv_table = binner.iv_table().set_index("feature")["iv"]
    candidates = [f for f in binner.binnings_ if iv_table.get(f, 0.0) >= min_iv]
    dropped: list[tuple[str, str]] = [
        (f, f"IV {iv_table.get(f, 0.0):.4f} below {min_iv}")
        for f in binner.binnings_
        if f not in candidates
    ]

    features, corr_dropped = _prune_correlated(train_woe, candidates, iv_table, max_correlation)
    dropped.extend(corr_dropped)

    y = train_woe[TARGET].to_numpy(dtype=float)

    # WOE is signed so that a higher value means a better customer. Predicting
    # the bad outcome, every coefficient must therefore come out negative.
    while True:
        X = train_woe[[f"woe_{f}" for f in features]].to_numpy(dtype=float)
        model = LogisticRegression(max_iter=2000, C=1e6, solver="lbfgs")
        model.fit(X, y)
        coefs = model.coef_.ravel()

        wrong = np.flatnonzero(coefs > 0)
        if wrong.size == 0 or len(features) <= 2:
            break
        worst = int(wrong[np.argmax(coefs[wrong])])
        dropped.append((features[worst], f"wrong sign (coef {coefs[worst]:+.3f})"))
        log.info("dropping %s: wrong-signed coefficient", features[worst])
        features = [f for i, f in enumerate(features) if i != worst]

    probability = model.predict_proba(X)[:, 1]
    stats_frame = _wald_statistics(X, probability, np.concatenate([model.intercept_, coefs]))
    diagnostics = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefs,
            "iv": [float(iv_table.get(f, np.nan)) for f in features],
            "vif": _vif(X),
        }
    ).join(stats_frame)
    diagnostics = diagnostics.sort_values("iv", ascending=False).reset_index(drop=True)

    card = ScorecardModel(
        features=features,
        coefficients=dict(zip(features, coefs, strict=True)),
        intercept=float(model.intercept_[0]),
        scale=scale,
        diagnostics=diagnostics,
        dropped=dropped,
    )
    card.points, card.base_points = _allocate_points(card, binner)
    log.info(
        "scorecard fitted on %d features (%d dropped)", len(features), len(dropped)
    )
    return card


def _allocate_points(card: ScorecardModel, binner) -> tuple[dict[str, dict[str, float]], float]:
    """Distribute the score across features so the points sum to the total.

    ``points_ij = -(coef_i * woe_ij + intercept / k) * factor + offset / k``
    """
    k = max(len(card.features), 1)
    factor = card.scale.factor
    offset = card.scale.offset
    base = -factor * (card.intercept / k) + offset / k

    points: dict[str, dict[str, float]] = {}
    for feature in card.features:
        coef = card.coefficients[feature]
        mapping = {}
        for bin_stat in binner.binnings_[feature].bins:
            mapping[bin_stat.label] = float(-factor * coef * bin_stat.woe + base)
        points[feature] = mapping
    return points, float(base)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _diagnostics_section(card: ScorecardModel) -> str:
    frame = card.diagnostics.copy()
    frame["coefficient"] = frame["coefficient"].round(4)
    frame["iv"] = frame["iv"].round(4)
    frame["vif"] = frame["vif"].round(2)
    frame["p_value"] = frame["p_value"].round(4)
    frame["wald_z"] = frame["wald_z"].round(2)
    frame = frame[["feature", "coefficient", "iv", "wald_z", "p_value", "vif"]]
    return markdown_table(frame)


def _points_section(card: ScorecardModel, binner, features: list[str]) -> str:
    parts = []
    for feature in features:
        mapping = card.points[feature]
        rows = [
            {
                "bin": b.label,
                "share_%": round(100 * b.count / max(sum(x.count for x in binner.binnings_[feature].bins), 1), 2),
                "bad_rate_%": round(100 * b.event_rate, 2),
                "woe": round(b.woe, 4),
                "points": round(mapping[b.label], 1),
            }
            for b in binner.binnings_[feature].bins
        ]
        parts.append(f"### `{feature}`\n\n" + markdown_table(pd.DataFrame(rows)))
    return "\n\n".join(parts)


def _rank_ordering_section(y_true, scores) -> str:
    table = rank_ordering_table(y_true, -np.asarray(scores))  # low score = high risk
    out = pd.DataFrame(
        {
            "band": table["band"],
            "count": table["count"],
            "score_range": [
                f"{int(-b)}-{int(-a)}" for a, b in zip(table["min_score"], table["max_score"], strict=True)
            ],
            "bad_rate_%": (table["bad_rate"] * 100).round(2),
            "cum_bad_capture_%": (table["cum_bad_capture"] * 100).round(1),
            "lift": table["lift"].round(2),
        }
    )
    return markdown_table(out)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main(_argv: list[str] | None = None) -> int:
    binner = joblib.load(BINNER_PATH)
    frames = {
        name: pd.read_parquet(f"{PROCESSED}/{name}_woe.parquet")
        for name in ("train", "valid", "oot")
    }

    card = fit_scorecard(frames["train"], binner)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(card, SCORECARD_PATH)
    log.info("saved scorecard to %s", SCORECARD_PATH)

    evaluations: list[Evaluation] = []
    scores = {}
    for name, frame in frames.items():
        probability = card.predict_proba(frame)
        scores[name] = card.score(frame)
        evaluations.append(evaluate(name, frame[TARGET].to_numpy(dtype=float), probability))

    summary = evaluation_frame(evaluations)
    oot = next(e for e in evaluations if e.label == "oot")

    top_features = card.diagnostics.head(8)["feature"].tolist()
    write_report(
        "03-scorecard",
        "Interpretable logistic scorecard (the challenger)",
        [
            (
                "Why this model exists",
                "This is the benchmark the gradient booster has to beat. A champion that "
                "buys two points of Gini over a scorecard, at the cost of being unable to "
                "explain a decline, is not a good trade in a regulated lending book — so "
                "the trade has to be measured rather than assumed.",
            ),
            (
                "Performance",
                markdown_table(
                    summary[["split", "n", "bad_rate", "auc", "gini", "ks", "calibration_ratio"]]
                ),
            ),
            (
                "Rank ordering out of time",
                "Band 1 is the riskiest decile by score. The bad rate must fall "
                "monotonically; a scorecard that inverts anywhere in the middle is not "
                "usable as a decline strategy.\n\n"
                + _rank_ordering_section(
                    frames["oot"][TARGET].to_numpy(dtype=float), scores["oot"]
                ),
            ),
            (
                "Coefficients and diagnostics",
                f"{len(card.features)} characteristics survived selection. Every "
                "coefficient is negative by construction: weight of evidence is signed so "
                "higher means a better customer, and the model predicts default. Variance "
                "inflation factors are reported because a scorecard with two "
                "near-duplicate characteristics double-counts the same behaviour.\n\n"
                + _diagnostics_section(card),
            ),
            (
                "Characteristics rejected during selection",
                markdown_table(
                    pd.DataFrame(card.dropped, columns=["feature", "reason"])
                )
                if card.dropped
                else "None.",
            ),
            (
                "Points allocation (eight strongest characteristics)",
                f"Points are scaled so a score of {card.scale.base_score} means "
                f"{card.scale.base_odds_good_to_bad:.0f}:1 good-to-bad odds and every "
                f"{card.scale.pdo} points doubles those odds. Each customer's score is the "
                f"sum of their bin points across all {len(card.features)} characteristics.\n\n"
                + _points_section(card, binner, top_features),
            ),
        ],
    )

    write_metrics(
        "03-scorecard",
        {
            "n_features": len(card.features),
            "features": card.features,
            "intercept": card.intercept,
            "coefficients": card.coefficients,
            "splits": [e.as_dict() for e in evaluations],
            "oot_gini": oot.discrimination.gini,
            "oot_ks": oot.discrimination.ks,
        },
    )

    log.info(
        "scorecard OOT: AUC %.4f | Gini %.4f | KS %.4f",
        oot.discrimination.auc,
        oot.discrimination.gini,
        oot.discrimination.ks,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
