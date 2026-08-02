"""Data access and the time-based splitting policy.

Splitting is *always* by application date, never randomly. A random split on a
credit book leaks the future into the past and inflates every metric you then
publish; the out-of-time window is the only honest estimate of how a scorecard
will behave once it is deployed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from mkopo.config import RAW_DIR
from mkopo.data.generator import OOT_START_MONTH
from mkopo.data.schema import APPROVAL_COLUMN, ID_COLUMN, ORACLE_TARGET, TARGET

log = logging.getLogger(__name__)

APPLICATIONS_PATH = RAW_DIR / "applications.parquet"
ORACLE_PATH = RAW_DIR / "oracle.parquet"

#: Share of the in-time window held back for validation / early stopping.
VALIDATION_FRACTION = 0.20


def load_applications(path=None) -> pd.DataFrame:
    """Load the through-the-door application book (approved *and* rejected)."""
    path = path or APPLICATIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `mkopo generate-data` (or `make data`) first."
        )
    return pd.read_parquet(path)


def load_oracle(path=None) -> pd.DataFrame:
    """Load counterfactual outcomes.

    This file is for **evaluation only**. Nothing under :mod:`mkopo.models` may
    read it during fitting; it exists so that reject inference can be scored
    against ground truth instead of taken on faith.
    """
    path = path or ORACLE_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `mkopo generate-data` first.")
    return pd.read_parquet(path)


@dataclass
class Split:
    """A time-ordered train / validation / out-of-time partition."""

    train: pd.DataFrame
    valid: pd.DataFrame
    oot: pd.DataFrame

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, frame in (("train", self.train), ("valid", self.valid), ("oot", self.oot)):
            target = frame[TARGET].dropna()
            rows.append(
                {
                    "split": name,
                    "rows": len(frame),
                    "with_outcome": len(target),
                    "bad_rate": round(float(target.mean()), 4) if len(target) else None,
                    "from": frame["application_date"].min().date(),
                    "to": frame["application_date"].max().date(),
                }
            )
        return pd.DataFrame(rows)


def time_split(
    frame: pd.DataFrame,
    *,
    oot_start_month: int = OOT_START_MONTH,
    validation_fraction: float = VALIDATION_FRACTION,
    approved_only: bool = True,
    seed: int = 42,
) -> Split:
    """Partition the book into train / valid / out-of-time.

    ``approved_only`` keeps the default modelling view: you can only learn from
    loans that were actually disbursed. Pass ``False`` to get the full
    through-the-door population used by reject inference and fairness work.
    """
    data = frame[frame[APPROVAL_COLUMN] == 1].copy() if approved_only else frame.copy()

    oot = data[data["application_month"] >= oot_start_month]
    in_time = data[data["application_month"] < oot_start_month]

    valid = in_time.sample(frac=validation_fraction, random_state=seed)
    train = in_time.drop(valid.index)

    log.info(
        "split -> train=%s valid=%s oot=%s (approved_only=%s)",
        f"{len(train):,}",
        f"{len(valid):,}",
        f"{len(oot):,}",
        approved_only,
    )
    return Split(train=train, valid=valid, oot=oot)


def attach_oracle(frame: pd.DataFrame, oracle: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join counterfactual outcomes for evaluation-time analysis only."""
    oracle = oracle if oracle is not None else load_oracle()
    cols = [ID_COLUMN, ORACLE_TARGET, "pd_true", "latent_capacity"]
    return frame.merge(oracle[cols], on=ID_COLUMN, how="left")
