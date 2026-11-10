import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from xbt_uncertainty.calibration import (
    brier_score,
    cluster_bootstrap,
    expected_calibration_error,
    log_loss,
    reliability_table,
    top_label,
)
from xbt_uncertainty.dataset import YearFrequency
from xbt_uncertainty.ensemble import CruiseBaggedTrees, cruise_bootstrap


def test_brier_limits():
    y = np.array([0, 1, 2])
    assert brier_score(y, np.eye(3)) == 0.0
    assert brier_score(y, np.eye(3)[[1, 2, 0]]) == 2.0
    uniform = np.full((3, 3), 1 / 3)
    assert brier_score(y, uniform) == pytest.approx(2 / 3)


def test_log_loss_is_finite_when_the_true_class_gets_zero():
    y = np.array([0])
    loss = log_loss(y, np.array([[0.0, 1.0]]), floor=1e-3)
    assert np.isfinite(loss)
    assert loss == pytest.approx(-np.log(1e-3 / (1 + 1e-3)))


def test_log_loss_of_a_fair_coin():
    assert log_loss(np.array([0, 1]), np.full((2, 2), 0.5)) == pytest.approx(np.log(2))


def test_calibrated_forecasts_have_small_ece():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.3, 1.0, 200_000)
    correct = rng.uniform(size=confidence.size) < confidence
    assert expected_calibration_error(confidence, correct) < 0.01


def test_overconfident_forecasts_have_large_ece():
    rng = np.random.default_rng(0)
    confidence = np.full(10_000, 0.95)
    correct = rng.uniform(size=confidence.size) < 0.6
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.35, abs=0.02)


def test_reliability_table_puts_confidence_one_in_the_last_bin():
    table = reliability_table(np.array([0.05, 0.5, 1.0]), np.array([0, 1, 1]), n_bins=10)
    assert table["count"].sum() == 3
    assert table["upper"].iloc[-1] == 1.0


def test_top_label():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    confidence, correct = top_label(np.array([1, 1]), proba)
    assert confidence.tolist() == [0.8, 0.6]
    assert correct.tolist() == [True, False]


def test_cruise_bootstrap_keeps_cruises_whole():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    rows = cruise_bootstrap(groups, np.random.default_rng(0))
    drawn = pd.Series(groups[rows]).value_counts()
    sizes = pd.Series(groups).value_counts()
    for cruise, n in drawn.items():
        assert n % sizes[cruise] == 0


def test_cluster_bootstrap_interval_contains_the_mean():
    groups = np.repeat(np.arange(100), 5)
    values = np.repeat(np.random.default_rng(1).normal(size=100), 5)
    low, high = cluster_bootstrap(lambda rows: values[rows].mean(), groups, n_resamples=300)
    assert low < values.mean() < high


def two_class_data(n_cruises=40, per_cruise=20, seed=0):
    rng = np.random.default_rng(seed)
    cruise = np.repeat(np.arange(n_cruises), per_cruise)
    label = np.where(cruise % 2, "T7", "T4")
    depth = np.where(label == "T4", 460.0, 760.0) + rng.normal(0, 150, cruise.size)
    X = pd.DataFrame({"max_depth": depth})
    return X, pd.Series(label), cruise


def test_ensemble_probabilities_are_vote_fractions():
    X, y, cruise = two_class_data()
    model = CruiseBaggedTrees(DecisionTreeClassifier(max_depth=3), n_members=10, n_jobs=1)
    model.fit(X, y, groups=cruise)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba * 10, np.round(proba * 10))
    assert list(model.classes_) == ["T4", "T7"]
    assert set(model.predict(X)) <= {"T4", "T7"}


def test_ensemble_is_repeatable_with_a_seed():
    X, y, cruise = two_class_data()
    a = CruiseBaggedTrees(DecisionTreeClassifier(), n_members=5, seed=3, n_jobs=1).fit(X, y, cruise)
    b = CruiseBaggedTrees(DecisionTreeClassifier(), n_members=5, seed=3, n_jobs=1).fit(X, y, cruise)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_ensemble_is_less_certain_than_one_deep_tree_on_overlapping_classes():
    X, y, cruise = two_class_data()
    X_new, _, _ = two_class_data(seed=1)
    single = DecisionTreeClassifier(random_state=0).fit(X, y)
    ensemble = CruiseBaggedTrees(DecisionTreeClassifier(), n_members=20, n_jobs=1).fit(X, y, cruise)
    certain_single = (single.predict_proba(X_new).max(axis=1) == 1).mean()
    certain_ensemble = (ensemble.predict_proba(X_new).max(axis=1) == 1).mean()
    assert certain_ensemble < certain_single


def test_year_frequency_matches_training_shares():
    X = pd.DataFrame({"year": [1990, 1990, 1990, 1991]})
    y = ["T4", "T4", "T7", "T7"]
    model = YearFrequency().fit(X, y)
    proba = model.predict_proba(pd.DataFrame({"year": [1990, 1991, 1999]}))
    np.testing.assert_allclose(proba, [[2 / 3, 1 / 3], [0, 1], [0.5, 0.5]])


def test_average_proba_sums_to_one_and_spreads_overlap():
    X, y, cruise = two_class_data()
    model = CruiseBaggedTrees(DecisionTreeClassifier(max_depth=2), n_members=10, n_jobs=1)
    model.fit(X, y, groups=cruise)
    votes = model.predict_proba(X)
    average = model.average_proba(X)
    np.testing.assert_allclose(average.sum(axis=1), 1.0)
    assert (average.max(axis=1) == 1).mean() < (votes.max(axis=1) == 1).mean()
