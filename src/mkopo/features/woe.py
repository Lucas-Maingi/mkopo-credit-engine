"""Weight-of-evidence binning with enforced monotonicity.

Why WOE at all, when the champion is a gradient booster? Three reasons that
matter in a regulated lending context:

1. It produces the **interpretable challenger**. A WOE scorecard is the model a
   credit committee can read line by line; the booster has to earn its extra
   complexity by beating it on out-of-time Gini, not merely on paper.
2. **Information Value** gives a defensible, univariate feature-selection story
   that a risk reviewer already knows how to challenge.
3. It makes **missingness a first-class bin**. A customer with no borrowing
   history is not a customer with an average borrowing history, and mean
   imputation destroys exactly the signal that matters for thin files.

Conventions used here
---------------------
``WOE = ln( (goods in bin / all goods) / (bads in bin / all bads) )`` so a
**positive WOE means lower risk**. Counts are Laplace-smoothed so a pure bin
cannot produce an infinite weight.

Monotonicity is enforced by merging adjacent bins that violate the trend
direction implied by the data, subject to the sign prior declared in
:mod:`mkopo.data.schema` where one exists. A scorecard whose risk ranking
reverses in the middle of a feature is not one anybody should sign off.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from mkopo.data.schema import CATEGORICAL_FEATURES, MODELLED_FEATURES, MONOTONE_PRIORS

log = logging.getLogger(__name__)

MISSING_LABEL = "Missing"
OTHER_LABEL = "__other__"
SMOOTHING = 0.5


@dataclass
class BinStat:
    """One bin of one feature, with everything needed to audit it."""

    label: str
    lower: float | None
    upper: float | None
    is_missing: bool
    count: int
    bads: int
    event_rate: float
    woe: float
    iv: float

    @property
    def goods(self) -> int:
        return self.count - self.bads


@dataclass
class FeatureBinning:
    """Fitted binning for a single feature."""

    feature: str
    kind: str  # "numeric" | "categorical"
    bins: list[BinStat] = field(default_factory=list)
    edges: list[float] = field(default_factory=list)  # numeric: inner cut points
    level_map: dict[str, str] = field(default_factory=dict)  # categorical: level -> bin label
    iv: float = 0.0
    monotonic: bool = True
    direction: int = 0  # +1 WOE rises with the feature, -1 falls, 0 flat/categorical

    def woe_lookup(self) -> dict[str, float]:
        return {b.label: b.woe for b in self.bins}

    def table(self) -> pd.DataFrame:
        rows = []
        total = sum(b.count for b in self.bins) or 1
        for b in self.bins:
            rows.append(
                {
                    "bin": b.label,
                    "count": b.count,
                    "share": b.count / total,
                    "bads": b.bads,
                    "event_rate": b.event_rate,
                    "woe": b.woe,
                    "iv": b.iv,
                }
            )
        return pd.DataFrame(rows)


def iv_strength(iv: float) -> str:
    """The conventional Siddiqi interpretation bands for Information Value."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspiciously strong"


def _woe_and_iv(bads: np.ndarray, goods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed WOE and per-bin IV contribution."""
    b = bads + SMOOTHING
    g = goods + SMOOTHING
    bad_share = b / b.sum()
    good_share = g / g.sum()
    woe = np.log(good_share / bad_share)
    iv = (good_share - bad_share) * woe
    return woe, iv


class WoeBinner:
    """Fit monotonic WOE bins for every modelled feature.

    Parameters
    ----------
    max_bins:
        Upper bound on bins per feature after merging.
    min_bin_fraction:
        A bin must hold at least this share of the fitting rows, otherwise it is
        merged into its neighbour. Small bins produce unstable weights that fall
        apart out of time.
    enforce_monotonic:
        Merge adjacent bins until the WOE trend is monotone.
    respect_priors:
        Where :mod:`mkopo.data.schema` declares an expected risk direction, force
        the trend to that sign rather than whichever way the sample happens to go.
    """

    def __init__(
        self,
        *,
        max_bins: int = 8,
        min_bin_fraction: float = 0.03,
        max_prebins: int = 40,
        enforce_monotonic: bool = True,
        respect_priors: bool = True,
        features: tuple[str, ...] = MODELLED_FEATURES,
    ) -> None:
        self.max_bins = max_bins
        self.min_bin_fraction = min_bin_fraction
        self.max_prebins = max_prebins
        self.enforce_monotonic = enforce_monotonic
        self.respect_priors = respect_priors
        self.features = features
        self.binnings_: dict[str, FeatureBinning] = {}

    # ---------------------------------------------------------------- fitting

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoeBinner:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        if y.isna().any():
            raise ValueError("WoeBinner.fit received rows without an observed outcome")

        for feature in self.features:
            if feature not in X.columns:
                log.warning("feature %s missing from frame, skipped", feature)
                continue
            if feature in CATEGORICAL_FEATURES:
                binning = self._fit_categorical(X[feature], y, feature)
            else:
                binning = self._fit_numeric(X[feature], y, feature)
            self.binnings_[feature] = binning

        n_non_monotone = sum(1 for b in self.binnings_.values() if not b.monotonic)
        log.info(
            "fitted %d binnings (%d non-monotone), median IV %.4f",
            len(self.binnings_),
            n_non_monotone,
            float(np.median([b.iv for b in self.binnings_.values()])) if self.binnings_ else 0.0,
        )
        return self

    def _fit_numeric(self, values: pd.Series, y: pd.Series, feature: str) -> FeatureBinning:
        missing = values.isna()
        obs = values[~missing]
        obs_y = y[~missing]

        edges = self._prebin_edges(obs)
        counts, bads = self._numeric_counts(obs, obs_y, edges)

        min_count = max(int(self.min_bin_fraction * len(values)), 25)
        edges, counts, bads = self._merge_small(edges, counts, bads, min_count)

        prior = MONOTONE_PRIORS.get(feature, 0) if self.respect_priors else 0
        if self.enforce_monotonic:
            edges, counts, bads, direction = self._merge_to_monotone(edges, counts, bads, prior)
        else:
            direction = 0

        while len(counts) > self.max_bins:
            edges, counts, bads = self._merge_weakest(edges, counts, bads)

        bin_stats = self._assemble_numeric(edges, counts, bads, missing, y)
        binning = FeatureBinning(
            feature=feature,
            kind="numeric",
            bins=bin_stats,
            edges=[float(e) for e in edges],
            iv=float(sum(b.iv for b in bin_stats)),
            direction=direction,
        )
        binning.monotonic = self._is_monotone(
            [b.woe for b in bin_stats if not b.is_missing], direction
        )
        return binning

    def _prebin_edges(self, obs: pd.Series) -> np.ndarray:
        if obs.empty:
            return np.array([])
        quantiles = np.linspace(0, 1, self.max_prebins + 1)[1:-1]
        edges = np.unique(np.quantile(obs.to_numpy(), quantiles))
        # Discrete features (flags, small counts) collapse to their own levels.
        distinct = np.unique(obs.to_numpy())
        if len(distinct) <= self.max_bins:
            edges = (distinct[:-1] + distinct[1:]) / 2.0 if len(distinct) > 1 else np.array([])
        return edges

    @staticmethod
    def _numeric_counts(
        obs: pd.Series, obs_y: pd.Series, edges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.digitize(obs.to_numpy(), edges, right=True)
        n_bins = len(edges) + 1
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        bads = np.bincount(idx, weights=obs_y.to_numpy(), minlength=n_bins).astype(float)
        return counts, bads

    @staticmethod
    def _drop_edge(
        edges: np.ndarray, counts: np.ndarray, bads: np.ndarray, i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge bin ``i`` into bin ``i + 1`` by removing the edge between them."""
        counts = counts.copy()
        bads = bads.copy()
        counts[i + 1] += counts[i]
        bads[i + 1] += bads[i]
        keep = np.ones(len(counts), dtype=bool)
        keep[i] = False
        return np.delete(edges, i), counts[keep], bads[keep]

    def _merge_small(
        self, edges: np.ndarray, counts: np.ndarray, bads: np.ndarray, min_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        while len(counts) > 2 and counts.min() < min_count:
            i = int(np.argmin(counts))
            j = i - 1 if i == len(counts) - 1 else i
            edges, counts, bads = self._drop_edge(edges, counts, bads, j)
        return edges, counts, bads

    def _merge_weakest(
        self, edges: np.ndarray, counts: np.ndarray, bads: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge the adjacent pair whose WOE differs least - the least informative split."""
        woe, _ = _woe_and_iv(bads, counts - bads)
        i = int(np.argmin(np.abs(np.diff(woe))))
        return self._drop_edge(edges, counts, bads, i)

    def _merge_to_monotone(
        self, edges: np.ndarray, counts: np.ndarray, bads: np.ndarray, prior: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        direction = prior
        if direction == 0:
            woe, _ = _woe_and_iv(bads, counts - bads)
            if len(woe) > 1:
                order = np.arange(len(woe))
                direction = 1 if np.corrcoef(order, woe)[0, 1] >= 0 else -1
            else:
                direction = 1

        while len(counts) > 2:
            woe, _ = _woe_and_iv(bads, counts - bads)
            diffs = np.diff(woe) * direction
            violations = np.flatnonzero(diffs < 0)
            if violations.size == 0:
                break
            # Merge across the worst violation.
            i = int(violations[np.argmin(diffs[violations])])
            edges, counts, bads = self._drop_edge(edges, counts, bads, i)
        return edges, counts, bads, direction

    @staticmethod
    def _is_monotone(woes: list[float], direction: int) -> bool:
        if len(woes) < 2 or direction == 0:
            return True
        diffs = np.diff(woes) * direction
        return bool((diffs >= -1e-9).all())

    def _assemble_numeric(
        self,
        edges: np.ndarray,
        counts: np.ndarray,
        bads: np.ndarray,
        missing: pd.Series,
        y: pd.Series,
    ) -> list[BinStat]:
        all_counts = list(counts)
        all_bads = list(bads)
        n_missing = int(missing.sum())
        if n_missing:
            all_counts.append(float(n_missing))
            all_bads.append(float(y[missing].sum()))

        woe, iv = _woe_and_iv(np.array(all_bads), np.array(all_counts) - np.array(all_bads))

        stats: list[BinStat] = []
        lowers = [-np.inf, *edges]
        uppers = [*edges, np.inf]
        for i in range(len(counts)):
            stats.append(
                BinStat(
                    label=_numeric_label(lowers[i], uppers[i]),
                    lower=float(lowers[i]),
                    upper=float(uppers[i]),
                    is_missing=False,
                    count=int(all_counts[i]),
                    bads=int(all_bads[i]),
                    event_rate=float(all_bads[i] / all_counts[i]) if all_counts[i] else 0.0,
                    woe=float(woe[i]),
                    iv=float(iv[i]),
                )
            )
        if n_missing:
            stats.append(
                BinStat(
                    label=MISSING_LABEL,
                    lower=None,
                    upper=None,
                    is_missing=True,
                    count=int(all_counts[-1]),
                    bads=int(all_bads[-1]),
                    event_rate=float(all_bads[-1] / all_counts[-1]),
                    woe=float(woe[-1]),
                    iv=float(iv[-1]),
                )
            )
        return stats

    def _fit_categorical(self, values: pd.Series, y: pd.Series, feature: str) -> FeatureBinning:
        filled = values.astype("object").where(values.notna(), MISSING_LABEL)
        counts = filled.value_counts()
        min_count = max(int(self.min_bin_fraction * len(values)), 25)
        rare = set(counts[counts < min_count].index)

        level_map = {
            str(level): (OTHER_LABEL if level in rare else str(level)) for level in counts.index
        }
        grouped = filled.map(lambda v: level_map.get(str(v), OTHER_LABEL))

        agg = pd.DataFrame({"g": grouped, "y": y}).groupby("g")["y"].agg(["count", "sum"])
        woe, iv = _woe_and_iv(agg["sum"].to_numpy(), (agg["count"] - agg["sum"]).to_numpy())

        stats = [
            BinStat(
                label=str(label),
                lower=None,
                upper=None,
                is_missing=str(label) == MISSING_LABEL,
                count=int(row["count"]),
                bads=int(row["sum"]),
                event_rate=float(row["sum"] / row["count"]),
                woe=float(w),
                iv=float(v),
            )
            for (label, row), w, v in zip(agg.iterrows(), woe, iv, strict=True)
        ]
        stats.sort(key=lambda b: b.woe)
        return FeatureBinning(
            feature=feature,
            kind="categorical",
            bins=stats,
            level_map=level_map,
            iv=float(sum(b.iv for b in stats)),
            monotonic=True,
        )

    # ------------------------------------------------------------- transform

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.binnings_:
            raise RuntimeError("WoeBinner is not fitted")
        out = {}
        for feature, binning in self.binnings_.items():
            out[f"woe_{feature}"] = self._transform_one(X[feature], binning)
        return pd.DataFrame(out, index=X.index)

    def assign_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the bin *label* per feature - used by the explanation layer."""
        out = {}
        for feature, binning in self.binnings_.items():
            out[feature] = self._label_one(X[feature], binning)
        return pd.DataFrame(out, index=X.index)

    def _label_one(self, values: pd.Series, binning: FeatureBinning) -> pd.Series:
        if binning.kind == "categorical":
            known = {b.label for b in binning.bins}
            mapped = values.astype("object").where(values.notna(), MISSING_LABEL).map(
                lambda v: binning.level_map.get(str(v), OTHER_LABEL)
            )
            return mapped.where(mapped.isin(known), OTHER_LABEL)

        edges = np.array(binning.edges, dtype=float)
        labels = [b.label for b in binning.bins if not b.is_missing]
        idx = np.digitize(values.to_numpy(dtype=float), edges, right=True)
        idx = np.clip(idx, 0, len(labels) - 1)
        assigned = pd.Series([labels[i] for i in idx], index=values.index, dtype="object")
        if values.isna().any():
            has_missing_bin = any(b.is_missing for b in binning.bins)
            assigned[values.isna()] = MISSING_LABEL if has_missing_bin else labels[0]
        return assigned

    def _transform_one(self, values: pd.Series, binning: FeatureBinning) -> pd.Series:
        lookup = binning.woe_lookup()
        labels = self._label_one(values, binning)
        # A level never seen in training carries no evidence either way.
        return labels.map(lookup).fillna(0.0).astype(float)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ---------------------------------------------------------------- reports

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {
                "feature": f,
                "iv": b.iv,
                "strength": iv_strength(b.iv),
                "bins": len(b.bins),
                "monotonic": b.monotonic,
                "kind": b.kind,
            }
            for f, b in self.binnings_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def selected_features(self, *, min_iv: float = 0.02) -> list[str]:
        table = self.iv_table()
        return table.loc[table["iv"] >= min_iv, "feature"].tolist()

    def to_dict(self) -> dict:
        return {
            f: {**asdict(b), "bins": [asdict(x) for x in b.bins]}
            for f, b in self.binnings_.items()
        }


def _numeric_label(lower: float, upper: float) -> str:
    lo = "-inf" if np.isneginf(lower) else f"{lower:,.4g}"
    hi = "inf" if np.isposinf(upper) else f"{upper:,.4g}"
    return f"({lo}, {hi}]"
