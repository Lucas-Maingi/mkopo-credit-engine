"""Build the modelling views and the committed feature-analysis report.

Two views come out of this stage, on purpose:

* a **raw matrix** for the gradient boosters, which handle missing values and
  non-linearity natively and would only be hurt by pre-binning;
* a **WOE matrix** for the logistic scorecard challenger.

Both are fitted on the training window only. The binner never sees validation or
out-of-time rows, because a binning fitted on the full book is itself a leak.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from mkopo.config import MODEL_DIR, PROCESSED_DIR
from mkopo.data.loaders import Split, load_applications, time_split
from mkopo.data.schema import (
    DATE_COLUMN,
    ID_COLUMN,
    MODELLED_FEATURES,
    PROTECTED_FEATURES,
    TARGET,
)
from mkopo.features.engineering import add_derived_features
from mkopo.features.woe import WoeBinner, iv_strength
from mkopo.reporting import markdown_table, write_metrics, write_report

log = logging.getLogger(__name__)

BINNER_PATH = MODEL_DIR / "woe_binner.joblib"
CARRY_COLUMNS = (ID_COLUMN, DATE_COLUMN, "application_month", "approved", TARGET)

#: Features below this Information Value carry no univariate signal worth the
#: model-risk overhead of documenting them.
MIN_IV = 0.02


def build_matrices(frame: pd.DataFrame | None = None) -> tuple[Split, WoeBinner]:
    """Return the derived-feature split and a binner fitted on the training window."""
    frame = frame if frame is not None else load_applications()
    frame = add_derived_features(frame)

    split = time_split(frame, approved_only=True)
    features = [f for f in MODELLED_FEATURES if f in split.train.columns]

    binner = WoeBinner(features=tuple(features))
    binner.fit(split.train[features], split.train[TARGET])
    return split, binner


def woe_view(binner: WoeBinner, frame: pd.DataFrame) -> pd.DataFrame:
    """WOE matrix plus the identifiers and outcome needed downstream."""
    woe = binner.transform(frame)
    carry = [c for c in CARRY_COLUMNS if c in frame.columns]
    return pd.concat([frame[carry].reset_index(drop=True), woe.reset_index(drop=True)], axis=1)


def _iv_section(binner: WoeBinner) -> str:
    table = binner.iv_table()
    top = table.head(20).copy()
    top["iv"] = top["iv"].round(4)
    top.insert(0, "rank", range(1, len(top) + 1))
    return markdown_table(top[["rank", "feature", "iv", "strength", "bins", "monotonic"]])


def _binning_section(binner: WoeBinner, features: list[str]) -> str:
    parts = []
    for feature in features:
        binning = binner.binnings_[feature]
        table = binning.table()
        table["share"] = (table["share"] * 100).round(2)
        table["event_rate"] = (table["event_rate"] * 100).round(2)
        table = table.rename(columns={"share": "share_%", "event_rate": "bad_rate_%"})
        parts.append(
            f"### `{feature}` — IV {binning.iv:.4f} ({iv_strength(binning.iv)})\n\n"
            + markdown_table(table[["bin", "count", "share_%", "bads", "bad_rate_%", "woe"]])
        )
    return "\n\n".join(parts)


def _coverage_section(split: Split) -> str:
    return markdown_table(split.describe(), floatfmt="{:,.4f}")


def _missingness_section(frame: pd.DataFrame) -> str:
    features = [f for f in MODELLED_FEATURES if f in frame.columns]
    miss = frame[features].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return "No modelled feature has missing values in the training window."
    table = pd.DataFrame({"feature": miss.index, "missing_%": (miss.to_numpy() * 100).round(2)})
    return (
        "Missingness here is *informative*, not a defect: a customer with no borrowing "
        "history genuinely has no repayment ratio. Every one of these features keeps a "
        "dedicated `Missing` bin with its own weight rather than being mean-imputed.\n\n"
        + markdown_table(table)
    )


def main(_argv: list[str] | None = None) -> int:
    split, binner = build_matrices()
    features = [f for f in MODELLED_FEATURES if f in split.train.columns]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, frame in (("train", split.train), ("valid", split.valid), ("oot", split.oot)):
        keep = [c for c in CARRY_COLUMNS if c in frame.columns]
        keep += [c for c in PROTECTED_FEATURES if c in frame.columns]
        raw = frame[keep + features]
        raw.to_parquet(PROCESSED_DIR / f"{name}_raw.parquet", index=False)
        woe_view(binner, frame).to_parquet(PROCESSED_DIR / f"{name}_woe.parquet", index=False)

    joblib.dump(binner, BINNER_PATH)
    log.info("saved binner to %s", BINNER_PATH)

    iv_table = binner.iv_table()
    selected = binner.selected_features(min_iv=MIN_IV)
    dropped = [f for f in features if f not in selected]
    non_monotone = iv_table.loc[~iv_table["monotonic"], "feature"].tolist()

    write_report(
        "02-feature-analysis",
        "Feature analysis: information value and weight of evidence",
        [
            (
                "Splits",
                "Splitting is by application date, never at random. The out-of-time window is "
                "the only honest read on deployment behaviour, and it deliberately sits inside "
                "the simulated macro squeeze.\n\n" + _coverage_section(split),
            ),
            (
                "Information value ranking",
                f"{len(features)} modelled features, of which **{len(selected)}** clear the "
                f"IV ≥ {MIN_IV} bar used to admit a feature to the scorecard. Top 20:\n\n"
                + _iv_section(binner),
            ),
            (
                "Features dropped for weak univariate signal",
                ", ".join(f"`{f}`" for f in dropped) if dropped else "None.",
            ),
            ("Missingness", _missingness_section(split.train)),
            (
                "Binning detail for the ten strongest features",
                "Bins are merged until the weight-of-evidence trend is monotone in the "
                "direction a credit reviewer would insist on. A scorecard whose risk ranking "
                "reverses mid-feature does not get signed off, whatever its Gini.\n\n"
                + _binning_section(binner, iv_table.head(10)["feature"].tolist()),
            ),
        ],
    )

    write_metrics(
        "02-feature-analysis",
        {
            "n_features_modelled": len(features),
            "n_features_selected": len(selected),
            "selected_features": selected,
            "dropped_features": dropped,
            "non_monotone_features": non_monotone,
            "median_iv": float(iv_table["iv"].median()),
            "max_iv": float(iv_table["iv"].max()),
            "splits": split.describe().to_dict(orient="records"),
        },
    )

    log.info("selected %d/%d features (IV >= %.2f)", len(selected), len(features), MIN_IV)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
