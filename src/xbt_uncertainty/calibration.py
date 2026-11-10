"""Proper scores and reliability tables for multiclass probabilities."""

import numpy as np
import pandas as pd


def one_hot(y_index, n_classes):
    out = np.zeros((len(y_index), n_classes))
    out[np.arange(len(y_index)), y_index] = 1.0
    return out


def brier_score(y_index, proba):
    """Mean over casts of the squared distance between proba and the one-hot truth.

    Ranges from 0 (certain and right) to 2 (certain and wrong).
    """
    return float(np.mean(np.sum((proba - one_hot(y_index, proba.shape[1])) ** 2, axis=1)))


def log_loss(y_index, proba, floor=1e-3):
    """Mean negative log probability of the true class.

    Vote fractions can be exactly 0, which would make the loss infinite, so
    probabilities are raised to at least floor and renormalised first.
    """
    clipped = np.maximum(proba, floor)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(y_index)), y_index])))


def reliability_table(confidence, correct, n_bins=10):
    """Mean confidence and observed accuracy in equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "confidence": confidence, "correct": correct})
    table = frame.groupby("bin").agg(
        count=("correct", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean")
    )
    table["lower"] = edges[table.index]
    table["upper"] = edges[table.index + 1]
    return table.reset_index(drop=True)


def expected_calibration_error(confidence, correct, n_bins=10):
    """Weighted mean gap between confidence and accuracy over the bins."""
    table = reliability_table(confidence, correct, n_bins)
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * np.abs(table["accuracy"] - table["confidence"])))


def top_label(y_index, proba):
    """Confidence in the predicted class, and whether that class is right."""
    predicted = proba.argmax(axis=1)
    return proba.max(axis=1), predicted == y_index


def cluster_bootstrap(metric, groups, n_resamples=200, seed=0):
    """Percentile interval for metric(rows) when whole groups are resampled.

    Casts from one cruise are not independent, so resampling casts would make
    the interval too narrow.
    """
    groups = pd.Series(np.asarray(groups))
    rows_by_group = list(groups.groupby(groups).indices.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(rows_by_group), size=len(rows_by_group))
        values.append(metric(np.concatenate([rows_by_group[i] for i in drawn])))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
