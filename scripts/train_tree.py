"""Stage 3: decision tree for probe model, scored on cruises it has not seen.

Usage: python scripts/train_tree.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from xbt_uncertainty.dataset import (
    FEATURES,
    YearMajority,
    feature_encoder,
    labelled_ship_casts,
    merge_rare_models,
    split_by_cruise,
)
from xbt_uncertainty.wod import load_cast_table

CASTS = Path("data/processed/casts.csv")
RESULTS = Path("results")
MIN_CRUISES = 50
SEED = 0
DEPTHS = [5, 10, 15, 20, 30]


def tree_pipeline(max_depth=None):
    return make_pipeline(
        feature_encoder(), DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)
    )


def report(name, y_true, y_pred, labels):
    print(f"\n{name}")
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))
    return classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )


def main():
    table, rare = merge_rare_models(labelled_ship_casts(load_cast_table(CASTS)), MIN_CRUISES)
    labels = table["target"].value_counts().index.tolist()
    print(f"Casts: {len(table)}, cruises: {table['cruise'].nunique()}")
    print(f"Merged into {'OTHER'} (fewer than {MIN_CRUISES} cruises): {', '.join(rare)}")

    is_test = split_by_cruise(table, seed=SEED)
    train, test = table[~is_test], table[is_test]
    assert not set(train["cruise"]) & set(test["cruise"])
    print(
        f"Train: {len(train)} casts, {train['cruise'].nunique()} cruises. "
        f"Test: {len(test)} casts, {test['cruise'].nunique()} cruises "
        f"({len(test) / len(table):.1%} of casts)"
    )
    print("\nClass counts")
    print(
        pd.DataFrame(
            {
                "train": train["target"].value_counts(),
                "test": test["target"].value_counts(),
                "test cruises": test.groupby("target")["cruise"].nunique(),
            }
        )
        .reindex(labels)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    # Tree depth is chosen by cross-validation on the training cruises only.
    search = GridSearchCV(
        tree_pipeline(),
        {"decisiontreeclassifier__max_depth": DEPTHS},
        scoring="f1_macro",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
    )
    search.fit(train[FEATURES], train["target"], groups=train["cruise"])
    print("\nCross-validated macro F1 on training cruises, by max_depth")
    for depth, mean, std in zip(
        DEPTHS, search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]
    ):
        print(f"  {depth:>3}: {mean:.3f} +/- {std:.3f}")
    best_depth = search.best_params_["decisiontreeclassifier__max_depth"]
    print(f"  chosen: {best_depth}")

    tree = search.best_estimator_
    tree_pred = tree.predict(test[FEATURES])
    baseline_pred = YearMajority().fit(train[FEATURES], train["target"]).predict(test[FEATURES])

    tree_report = report("Decision tree, test cruises", test["target"], tree_pred, labels)
    baseline_report = report(
        "Baseline, most common model per year, test cruises", test["target"], baseline_pred, labels
    )

    # Same tree trained on a random split of casts, to measure the leakage
    # that a cruise split avoids.
    r_train, r_test = train_test_split(table, test_size=len(test), random_state=SEED)
    random_tree = tree_pipeline(best_depth).fit(r_train[FEATURES], r_train["target"])
    random_pred = random_tree.predict(r_test[FEATURES])

    summary = {
        "tree_by_cruise": f1_score(test["target"], tree_pred, average="macro"),
        "baseline_by_cruise": f1_score(test["target"], baseline_pred, average="macro"),
        "tree_random_split": f1_score(r_test["target"], random_pred, average="macro"),
    }
    print("Macro F1")
    for name, value in summary.items():
        print(f"  {name}: {value:.3f}")

    print("\nConfusion matrix, decision tree (rows true, columns predicted)")
    matrix = pd.DataFrame(
        confusion_matrix(test["target"], tree_pred, labels=labels), index=labels, columns=labels
    )
    print(matrix.to_string())

    RESULTS.mkdir(exist_ok=True)
    out = {
        "min_cruises": MIN_CRUISES,
        "merged_models": rare,
        "seed": SEED,
        "max_depth": best_depth,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "macro_f1": summary,
        "tree": tree_report,
        "baseline": baseline_report,
    }
    (RESULTS / "stage3_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    matrix.to_csv(RESULTS / "stage3_confusion.csv")
    print(f"\nWrote {RESULTS / 'stage3_metrics.json'} and {RESULTS / 'stage3_confusion.csv'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    np.set_printoptions(linewidth=200)
    main()
