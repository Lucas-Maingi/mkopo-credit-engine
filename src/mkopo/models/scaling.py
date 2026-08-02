"""Turning a probability into a credit score.

A probability of 0.043 means nothing to a branch agent, a customer or a
regulator. Every real scorecard therefore publishes an integer score on a fixed
scale, and the mapping is not arbitrary: it is a log-odds transform with two
parameters a credit committee chooses and then never changes, because changing
them silently re-prices the whole book.

    score = offset - factor * ln(odds_of_default)
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

With the defaults in :mod:`mkopo.config` a score of 600 means 30:1 good-to-bad,
and every additional 40 points doubles the odds of repaying.
"""

from __future__ import annotations

import numpy as np

from mkopo.config import RISK_BANDS, SCORECARD_SCALE, ScorecardScale

EPS = 1e-9


def probability_to_score(probability, scale: ScorecardScale = SCORECARD_SCALE) -> np.ndarray:
    """Map probability of default to an integer score on the configured scale."""
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    odds_bad = p / (1.0 - p)
    raw = scale.offset - scale.factor * np.log(odds_bad)
    return np.clip(np.round(raw), scale.min_score, scale.max_score).astype(int)


def score_to_probability(score, scale: ScorecardScale = SCORECARD_SCALE) -> np.ndarray:
    """Inverse of :func:`probability_to_score`, up to the rounding and clipping."""
    s = np.asarray(score, dtype=float)
    log_odds_bad = (scale.offset - s) / scale.factor
    odds = np.exp(log_odds_bad)
    return odds / (1.0 + odds)


def risk_band(score) -> np.ndarray:
    """Map scores to the A-E policy bands."""
    s = np.asarray(score, dtype=float)
    out = np.full(s.shape, RISK_BANDS[-1][0], dtype=object)
    for name, low, high in RISK_BANDS:
        out[(s >= low) & (s <= high)] = name
    return out


def band_table() -> list[dict]:
    """The published band definitions, for the API and the dashboard."""
    return [
        {
            "band": name,
            "min_score": low,
            "max_score": high,
            "indicative_pd": round(float(score_to_probability((low + high) / 2)), 4),
        }
        for name, low, high in RISK_BANDS
    ]
