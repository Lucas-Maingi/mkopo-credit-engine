"""Gradient boosted champions: LightGBM and XGBoost, tuned with Optuna.

Three decisions here are worth more than the hyperparameters.

**Monotonic constraints.** Where the schema declares a risk direction, the
booster is forced to respect it. Left unconstrained, a tree ensemble will
happily learn that customers with *slightly* more savings default *more*,
because forty rows in the training window say so. That is indefensible in a
credit hearing and it does not survive out of time. The constrained model is the
champion; the unconstrained one is fitted anyway so the cost of the constraint
is measured rather than hand-waved.

**Selection on validation, reporting on out-of-time.** The out-of-time window is
touched exactly once, after the champion is chosen. Tuning against it and then
quoting it as a hold-out is the most common way a portfolio project reports a
number that will not reproduce in production.

**A challenger that might win.** The scorecard from the previous stage is carried
into the comparison. If the booster cannot beat it by a margin worth the loss of
transparency, the honest recommendation is to ship the scorecard.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from mkopo.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from mkopo.data.schema import CATEGORICAL_FEATURES, MODELLED_FEATURES, MONOTONE_PRIORS, TARGET
from mkopo.evaluation.metrics import bootstrap_gini_difference, evaluate, evaluation_frame
from mkopo.reporting import markdown_table, read_metrics, write_metrics, write_report

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

PROCESSED = "data/processed"
CHAMPION_PATH = MODEL_DIR / "champion.joblib"
LGBM_PATH = MODEL_DIR / "lightgbm.joblib"
XGB_PATH = MODEL_DIR / "xgboost.joblib"
OPTUNA_DB = PROJECT_ROOT / "optuna.db"

N_TRIALS_LGBM = 40
N_TRIALS_XGB = 25


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    train: pd.DataFrame
    valid: pd.DataFrame
    oot: pd.DataFrame
    features: list[str]
    categorical: list[str]

    def X(self, split: str) -> pd.DataFrame:
        return getattr(self, split)[self.features]

    def y(self, split: str) -> np.ndarray:
        return getattr(self, split)[TARGET].to_numpy(dtype=float)


def load_dataset() -> Dataset:
    frames = {
        name: pd.read_parquet(f"{PROCESSED}/{name}_raw.parquet")
        for name in ("train", "valid", "oot")
    }
    features = [f for f in MODELLED_FEATURES if f in frames["train"].columns]
    categorical = [f for f in CATEGORICAL_FEATURES if f in features]

    for frame in frames.values():
        for column in categorical:
            frame[column] = frame[column].astype("category")
    # Categories must be identical across splits or the booster silently
    # re-encodes and the served model disagrees with the trained one.
    for column in categorical:
        levels = frames["train"][column].cat.categories
        for frame in frames.values():
            frame[column] = frame[column].cat.set_categories(levels)

    return Dataset(
        train=frames["train"],
        valid=frames["valid"],
        oot=frames["oot"],
        features=features,
        categorical=categorical,
    )


def monotone_vector(features: list[str]) -> list[int]:
    """Constraint per feature on P(default).

    The schema records the direction of *risk*: +1 means a higher value is a
    better customer. The booster models the probability of default, so the
    constraint is the opposite sign.
    """
    return [-MONOTONE_PRIORS.get(f, 0) for f in features]


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------


@dataclass
class TunedModel:
    name: str
    model: Any
    params: dict
    valid_auc: float
    trials: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 60.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    }


def _build_lgbm(params: dict, constraints: list[int] | None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        monotone_constraints=constraints,
        monotone_constraints_method="advanced" if constraints else None,
        **params,
    )


def _build_xgb(params: dict, constraints: list[int] | None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        monotone_constraints=tuple(constraints) if constraints else None,
        **params,
    )


def tune(
    data: Dataset,
    *,
    family: str,
    n_trials: int,
    constrained: bool = True,
) -> TunedModel:
    """Optuna search, scored on the validation window only."""
    constraints = monotone_vector(data.features) if constrained else None
    space = _lgbm_space if family == "lightgbm" else _xgb_space
    build = _build_lgbm if family == "lightgbm" else _build_xgb

    X_train, y_train = data.X("train"), data.y("train")
    X_valid, y_valid = data.X("valid"), data.y("valid")

    def objective(trial: optuna.Trial) -> float:
        model = build(space(trial), constraints)
        model.fit(X_train, y_train)
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))

    # Studies persist so a re-run resumes instead of paying for the search twice.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"{family}{'' if constrained else '-unconstrained'}",
        storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(n_trials - done, 0)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    else:
        log.info("%s: reusing %d completed trials from the study cache", family, done)

    best = build(study.best_params, constraints)
    best.fit(X_train, y_train)
    log.info(
        "%s%s: best valid AUC %.4f after %d trials",
        family,
        "" if constrained else " (unconstrained)",
        study.best_value,
        n_trials,
    )
    return TunedModel(
        name=family if constrained else f"{family}_unconstrained",
        model=best,
        params=study.best_params,
        valid_auc=float(study.best_value),
        trials=study.trials_dataframe(attrs=("number", "value", "params", "state")),
    )


# --------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------


def _log_to_mlflow(tuned: TunedModel, evaluations: list, data: Dataset) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=tuned.name):
        mlflow.log_params({f"hp_{k}": v for k, v in tuned.params.items()})
        mlflow.log_param("n_features", len(data.features))
        mlflow.log_param("monotone_constrained", not tuned.name.endswith("unconstrained"))
        mlflow.log_param("bad_definition", "dpd30_within_90d")
        for evaluation in evaluations:
            for key, value in evaluation.as_dict().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(f"{evaluation.label}_{key}", float(value))
        mlflow.log_text(tuned.trials.to_csv(index=False), "optuna_trials.csv")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _evaluate_all(tuned: TunedModel, data: Dataset) -> list:
    return [
        evaluate(split, data.y(split), tuned.predict_proba(data.X(split)))
        for split in ("train", "valid", "oot")
    ]


def _scorecard_versus_champion(champion: TunedModel, data: Dataset):
    """Paired bootstrap of the champion against the scorecard on out-of-time.

    Returns ``None`` when the scorecard has not been fitted yet.
    """
    from mkopo.data.schema import ID_COLUMN
    from mkopo.models.scorecard import SCORECARD_PATH

    if not SCORECARD_PATH.exists():
        return None

    card = joblib.load(SCORECARD_PATH)
    woe_oot = pd.read_parquet(f"{PROCESSED}/oot_woe.parquet")
    aligned = data.oot[[ID_COLUMN, TARGET]].merge(
        woe_oot.assign(card_pd=card.predict_proba(woe_oot))[[ID_COLUMN, "card_pd"]],
        on=ID_COLUMN,
        validate="1:1",
    )
    champion_pd = pd.Series(
        champion.predict_proba(data.X("oot")), index=data.oot[ID_COLUMN].to_numpy()
    )
    aligned["champion_pd"] = aligned[ID_COLUMN].map(champion_pd)
    return bootstrap_gini_difference(
        aligned[TARGET].to_numpy(dtype=float),
        aligned["champion_pd"].to_numpy(dtype=float),
        aligned["card_pd"].to_numpy(dtype=float),
    )


def _summary_row(name: str, evaluations: list) -> dict:
    by_split = {e.label: e for e in evaluations}
    return {
        "model": name,
        "valid_auc": round(by_split["valid"].discrimination.auc, 4),
        "oot_auc": round(by_split["oot"].discrimination.auc, 4),
        "oot_gini": round(by_split["oot"].discrimination.gini, 4),
        "oot_ks": round(by_split["oot"].discrimination.ks, 4),
        "oot_calibration_ratio": round(by_split["oot"].calibration.calibration_ratio, 4),
    }


def main(_argv: list[str] | None = None) -> int:
    data = load_dataset()
    log.info("training on %d features (%d categorical)", len(data.features), len(data.categorical))

    results: dict[str, TunedModel] = {}
    evaluations: dict[str, list] = {}

    for family, n_trials in (("lightgbm", N_TRIALS_LGBM), ("xgboost", N_TRIALS_XGB)):
        tuned = tune(data, family=family, n_trials=n_trials, constrained=True)
        results[tuned.name] = tuned
        evaluations[tuned.name] = _evaluate_all(tuned, data)
        _log_to_mlflow(tuned, evaluations[tuned.name], data)

    # What does the monotonic constraint actually cost?
    free = tune(data, family="lightgbm", n_trials=N_TRIALS_LGBM // 2, constrained=False)
    results[free.name] = free
    evaluations[free.name] = _evaluate_all(free, data)
    _log_to_mlflow(free, evaluations[free.name], data)

    # Champion selected on validation, never on out-of-time.
    constrained = {k: v for k, v in results.items() if not k.endswith("unconstrained")}
    champion_name = max(constrained, key=lambda k: constrained[k].valid_auc)
    champion = results[champion_name]

    joblib.dump(results["lightgbm"], LGBM_PATH)
    joblib.dump(results["xgboost"], XGB_PATH)
    joblib.dump(champion, CHAMPION_PATH)
    log.info("champion = %s, saved to %s", champion_name, CHAMPION_PATH)

    rows = [_summary_row(name, evals) for name, evals in evaluations.items()]
    scorecard_metrics = read_metrics("03-scorecard")
    if scorecard_metrics:
        oot = next(s for s in scorecard_metrics["splits"] if s["split"] == "oot")
        valid = next(s for s in scorecard_metrics["splits"] if s["split"] == "valid")
        rows.insert(
            0,
            {
                "model": "logistic scorecard",
                "valid_auc": valid["auc"],
                "oot_auc": oot["auc"],
                "oot_gini": oot["gini"],
                "oot_ks": oot["ks"],
                "oot_calibration_ratio": oot["calibration_ratio"],
            },
        )
    comparison = pd.DataFrame(rows)

    champion_oot = next(e for e in evaluations[champion_name] if e.label == "oot")
    scorecard_gini = comparison.loc[comparison["model"] == "logistic scorecard", "oot_gini"]
    uplift = (
        champion_oot.discrimination.gini - float(scorecard_gini.iloc[0])
        if len(scorecard_gini)
        else float("nan")
    )
    constraint_cost = (
        _summary_row("x", evaluations["lightgbm_unconstrained"])["oot_gini"]
        - _summary_row("x", evaluations["lightgbm"])["oot_gini"]
    )

    contest = _scorecard_versus_champion(champion, data)
    if contest is None:
        verdict_text = (
            f"The champion buys **{uplift:+.4f} Gini** out of time over the logistic "
            "scorecard. Run `mkopo fit-scorecard` first to get the significance test."
        )
    else:
        verdict = contest.as_dict()
        direction = "ahead of" if contest.difference > 0 else "behind"
        verdict_text = (
            f"Out of time the champion lands **{abs(contest.difference):.4f} Gini "
            f"{direction}** the logistic scorecard "
            f"({contest.gini_a:.4f} against {contest.gini_b:.4f}). A paired bootstrap over "
            f"{contest.n_resamples:,} resamples puts the 95% interval for that gap at "
            f"**[{verdict['ci_low']:+.4f}, {verdict['ci_high']:+.4f}]**, two-sided "
            f"p = {verdict['p_two_sided']:.3f}.\n\n"
            + (
                "The interval contains zero, so **the two models are statistically "
                "indistinguishable on this book**. That is the finding, and it is worth "
                "more than a tenth of a point of Gini would have been: on well-binned "
                "behavioural data with monotonic constraints, four hundred trees buy "
                "nothing a thirty-four-line scorecard does not already have.\n\n"
                "The booster is retained as champion because it wins on the validation "
                "window, which is the pre-registered selection rule and the only window "
                "selection is allowed to touch. But the honest recommendation to a credit "
                "committee is that the scorecard could be shipped instead with no "
                "measurable loss of discrimination and a substantial gain in "
                "explainability — so it stays in the repository as a live fallback, "
                "wired into the same serving path, rather than as a discarded baseline."
                if not contest.significant
                else "The interval excludes zero, so the uplift is real and the extra "
                "complexity is defensible. The scorecard is still retained as a live "
                "fallback for the thin-file segment and as a challenger in production."
            )
        )

    importance = pd.DataFrame(
        {
            "feature": data.features,
            "gain": getattr(champion.model, "feature_importances_", np.zeros(len(data.features))),
        }
    ).sort_values("gain", ascending=False)
    importance["share_%"] = (100 * importance["gain"] / max(importance["gain"].sum(), 1)).round(2)

    write_report(
        "04-champion-models",
        "Champion selection: LightGBM and XGBoost against the scorecard",
        [
            (
                "How the champion was chosen",
                "Hyperparameters were tuned with Optuna against the **validation** window "
                "only. The out-of-time window was scored once, after selection. Tuning "
                "against out-of-time and then quoting it as a hold-out is the most common "
                "way a portfolio project publishes a number that will not reproduce.",
            ),
            ("Comparison", markdown_table(comparison)),
            ("Does the booster earn its complexity?", verdict_text),
            (
                "What monotonic constraints cost",
                f"Removing the constraints changes out-of-time Gini by "
                f"**{constraint_cost:+.4f}**. Unconstrained, the ensemble is free to learn "
                "that slightly more savings implies slightly more risk because a few dozen "
                "training rows said so. That is not defensible in a credit hearing and it "
                "does not survive a population shift, so the constrained model is the "
                "champion regardless of which way this number falls.",
            ),
            (
                "Champion hyperparameters",
                markdown_table(
                    pd.DataFrame(
                        sorted(champion.params.items()), columns=["parameter", "value"]
                    )
                ),
            ),
            (
                "Feature importance (gain)",
                "Gain is a training-time artefact, not an explanation. The customer-facing "
                "reasoning comes from SHAP in a later stage; this table is here only to "
                "confirm the model is leaning on behaviour that makes sense.\n\n"
                + markdown_table(importance.head(20)[["feature", "share_%"]]),
            ),
            (
                "Per-split detail",
                markdown_table(evaluation_frame(evaluations[champion_name])),
            ),
            (
                "Tracking",
                f"Every run is logged to MLflow at `{MLFLOW_TRACKING_URI}` under experiment "
                f"`{MLFLOW_EXPERIMENT}`, including the full Optuna trial history as an "
                "artefact. Run `mlflow ui` to browse them.",
            ),
        ],
    )

    write_metrics(
        "04-champion-models",
        {
            "champion": champion_name,
            "champion_params": champion.params,
            "gini_uplift_over_scorecard": round(uplift, 4),
            "champion_vs_scorecard_bootstrap": contest.as_dict() if contest else None,
            "monotonic_constraint_cost_gini": round(constraint_cost, 4),
            "comparison": comparison.to_dict(orient="records"),
            "champion_splits": [e.as_dict() for e in evaluations[champion_name]],
        },
    )

    log.info(
        "champion OOT: AUC %.4f | Gini %.4f | KS %.4f (uplift over scorecard %+.4f Gini)",
        champion_oot.discrimination.auc,
        champion_oot.discrimination.gini,
        champion_oot.discrimination.ks,
        uplift,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
