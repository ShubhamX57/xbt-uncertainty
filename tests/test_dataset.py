import numpy as np
import pandas as pd
import pytest

from xbt_uncertainty.dataset import (
    FEATURES,
    OTHER,
    YearMajority,
    feature_encoder,
    labelled_ship_casts,
    merge_rare_models,
    split_by_cruise,
)


def synthetic_casts(n_cruises=60, casts_per_cruise=10, seed=0):
    """Cruises that each use one probe model, with a small rare class."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cruises):
        if c < 3:
            instrument = "XBT: XBT-5 (SPARTON)"
        elif c % 2:
            instrument = "XBT: T7 (SIPPICAN)"
        else:
            instrument = "XBT: T4 (UNKNOWN BRAND)"
        for _ in range(casts_per_cruise):
            rows.append(
                {
                    "cast_id": len(rows),
                    "cruise": f"US{c:06d}",
                    "instrument": instrument,
                    "year": int(rng.integers(1990, 2000)),
                    "month": int(rng.integers(1, 13)),
                    "lat": rng.uniform(-60, 60),
                    "lon": rng.uniform(-180, 180),
                    "max_depth": 460.0 if "T4" in instrument else 760.0,
                    "country": "UNITED STATES",
                }
            )
    casts = pd.DataFrame(rows)
    casts["labelled"] = True
    casts["has_cruise"] = True
    return casts


def test_selection_drops_unlabelled_axbt_and_no_cruise():
    casts = synthetic_casts().head(4).copy()
    casts.loc[0, "instrument"] = "XBT: TYPE UNKNOWN"
    casts.loc[0, "labelled"] = False
    casts.loc[1, "instrument"] = "XBT: AXBT A/3 (HERMES)"
    casts.loc[2, "has_cruise"] = False
    selected = labelled_ship_casts(casts)
    assert selected["cast_id"].tolist() == [3]


def test_rare_models_are_merged():
    table, rare = merge_rare_models(labelled_ship_casts(synthetic_casts()), min_cruises=5)
    assert rare == ["XBT-5"]
    assert set(table["target"]) == {"T7", "T4", OTHER}
    assert (table.loc[table["model"] == "XBT-5", "target"] == OTHER).all()


def test_no_cruise_is_in_both_train_and_test():
    table, _ = merge_rare_models(labelled_ship_casts(synthetic_casts()), min_cruises=1)
    is_test = split_by_cruise(table, seed=1)
    train_cruises = set(table.loc[~is_test, "cruise"])
    test_cruises = set(table.loc[is_test, "cruise"])
    assert train_cruises and test_cruises
    assert not train_cruises & test_cruises


def test_split_is_about_the_requested_size_and_repeatable():
    table, _ = merge_rare_models(labelled_ship_casts(synthetic_casts()), min_cruises=1)
    first = split_by_cruise(table, seed=3)
    second = split_by_cruise(table, seed=3)
    np.testing.assert_array_equal(first, second)
    assert 0.1 < first.mean() < 0.3


def test_every_class_with_enough_cruises_reaches_the_test_set():
    table, _ = merge_rare_models(labelled_ship_casts(synthetic_casts()), min_cruises=1)
    is_test = split_by_cruise(table, seed=0)
    assert set(table.loc[is_test, "target"]) == set(table["target"])


def test_encoder_handles_a_country_unseen_in_training():
    table = synthetic_casts().head(5)
    encoder = feature_encoder().fit(table[FEATURES])
    unseen = table[FEATURES].copy()
    unseen["country"] = "ATLANTIS"
    encoded = encoder.transform(unseen)
    assert encoded.shape == (5, len(FEATURES))
    assert (encoded[:, 0] == -1).all()


def test_year_majority_baseline():
    X = pd.DataFrame({"year": [1990, 1990, 1990, 1991, 1991]})
    y = ["T4", "T4", "T7", "T7", "T7"]
    model = YearMajority().fit(X, y)
    test = pd.DataFrame({"year": [1990, 1991, 1995]})
    assert model.predict(test).tolist() == ["T4", "T7", "T7"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_beats_baseline_when_depth_separates_classes(seed):
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    table, _ = merge_rare_models(labelled_ship_casts(synthetic_casts(seed=seed)), min_cruises=5)
    is_test = split_by_cruise(table, seed=seed)
    train, test = table[~is_test], table[is_test]
    tree = make_pipeline(feature_encoder(), DecisionTreeClassifier(max_depth=3, random_state=0))
    tree.fit(train[FEATURES], train["target"])
    baseline = YearMajority().fit(train[FEATURES], train["target"])
    tree_f1 = f1_score(test["target"], tree.predict(test[FEATURES]), average="macro")
    base_f1 = f1_score(test["target"], baseline.predict(test[FEATURES]), average="macro")
    assert tree_f1 > base_f1
