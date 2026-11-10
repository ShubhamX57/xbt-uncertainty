"""Stage 4: how trustworthy are the probabilities from a cruise-bagged tree ensemble?

Usage: python scripts/evaluate_uncertainty.py
Needs results/stage3_metrics.json from scripts/train_tree.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from xbt_uncertainty.calibration import (
    brier_score,
    cluster_bootstrap,
    expected_calibration_error,
    log_loss,
    reliability_table,
    top_label,
)
from xbt_uncertainty.dataset import (
    FEATURES,
    YearFrequency,
    feature_encoder,
    labelled_ship_casts,
    merge_rare_models,
    split_by_cruise,
)
from xbt_uncertainty.ensemble import CruiseBaggedTrees
from xbt_uncertainty.wod import load_cast_table

CASTS = Path("data/processed/casts.csv")
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
N_MEMBERS = 50
N_BOOTSTRAP = 200


def scores(y_index, proba, rows=slice(None)):
    confidence, correct = top_label(y_index[rows], proba[rows])
    return {
        "accuracy": float(correct.mean()),
        "brier": brier_score(y_index[rows], proba[rows]),
        "log_loss": log_loss(y_index[rows], proba[rows]),
        "ece": expected_calibration_error(confidence, correct),
    }


def main():
    stage3 = json.loads((RESULTS / "stage3_metrics.json").read_text())
    table, _ = merge_rare_models(
        labelled_ship_casts(load_cast_table(CASTS)), stage3["min_cruises"]
    )
    is_test = split_by_cruise(table, seed=stage3["seed"])
    train, test = table[~is_test], table[is_test]
    assert len(test) == stage3["n_test"], "split differs from Stage 3"

    def tree():
        return make_pipeline(
            feature_encoder(),
            DecisionTreeClassifier(max_depth=stage3["max_depth"], random_state=stage3["seed"]),
        )

    single = tree().fit(train[FEATURES], train["target"])
    ensemble = CruiseBaggedTrees(tree(), n_members=N_MEMBERS, seed=stage3["seed"])
    ensemble.fit(train[FEATURES], train["target"], groups=train["cruise"])
    reference = YearFrequency().fit(train[FEATURES], train["target"])

    classes = ensemble.classes_
    assert list(single.classes_) == list(classes) == list(reference.classes_)
    y_index = np.searchsorted(classes, test["target"].to_numpy())

    proba = {
        "year frequency": reference.predict_proba(test[FEATURES]),
        "single tree": single.predict_proba(test[FEATURES]),
        "ensemble votes": ensemble.predict_proba(test[FEATURES]),
        "ensemble average": ensemble.average_proba(test[FEATURES]),
    }

    print(f"Test casts: {len(test)}, cruises: {test['cruise'].nunique()}")
    print(f"Interval: 2.5 to 97.5 percentile over {N_BOOTSTRAP} resamples of test cruises\n")
    groups = test["cruise"].to_numpy()
    summary = {}
    for name, p in proba.items():
        point = scores(y_index, p)
        intervals = {
            key: cluster_bootstrap(
                lambda rows, key=key: scores(y_index, p, rows)[key], groups, N_BOOTSTRAP
            )
            for key in ["brier", "ece"]
        }
        summary[name] = {**point, "brier_interval": intervals["brier"], "ece_interval": intervals["ece"]}
        print(name)
        print(f"  accuracy  {point['accuracy']:.3f}")
        print(f"  brier     {point['brier']:.3f}  [{intervals['brier'][0]:.3f}, {intervals['brier'][1]:.3f}]")
        print(f"  log loss  {point['log_loss']:.3f}")
        print(f"  ece       {point['ece']:.3f}  [{intervals['ece'][0]:.3f}, {intervals['ece'][1]:.3f}]")

    reference_brier = summary["year frequency"]["brier"]
    print("\nBrier skill score against year frequency (1 is perfect, 0 is no better)")
    for name in list(proba)[1:]:
        print(f"  {name}: {1 - summary[name]['brier'] / reference_brier:.3f}")

    ens = proba["ensemble votes"]
    votes = ens * N_MEMBERS
    true_votes = votes[np.arange(len(test)), y_index]
    print("\nEnsemble votes for the true class")
    print(f"  casts where no member voted for the true class: {(true_votes == 0).mean():.1%}")
    print(f"  casts where every member voted for the true class: {(true_votes == N_MEMBERS).mean():.1%}")
    single_conf, _ = top_label(y_index, proba["single tree"])
    ens_conf, _ = top_label(y_index, ens)
    print(f"  single tree gives probability 1 to its top class: {(single_conf == 1).mean():.1%}")
    print(f"  ensemble gives probability 1 to its top class: {(ens_conf == 1).mean():.1%}")

    reliability = {}
    for name in ["ensemble votes", "ensemble average"]:
        print(f"\nReliability of the top class, {name}")
        confidence, correct = top_label(y_index, proba[name])
        reliability[name] = reliability_table(confidence, correct)
        print(reliability[name].round(3).to_string(index=False))

    print("\nMean ensemble vote share against observed share, per class")
    per_class = pd.DataFrame(
        {
            "mean probability": ens.mean(axis=0),
            "observed share": np.bincount(y_index, minlength=len(classes)) / len(test),
        },
        index=classes,
    )
    print(per_class.round(4).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect calibration")
    for (name, p), marker in zip(list(proba.items())[1:], ["s", "o", "^"]):
        conf, corr = top_label(y_index, p)
        rel = reliability_table(conf, corr)
        rel = rel[rel["count"] >= 50]
        top.plot(rel["confidence"], rel["accuracy"], marker=marker, label=name)
        bottom.hist(conf, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.5, label=name)
    top.set_ylabel("observed accuracy")
    top.set_title("Probe model, test cruises 1990 to 1999")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.legend(loc="upper left", frameon=False)
    bottom.set_xlabel("probability given to the predicted model")
    bottom.set_ylabel("casts")
    bottom.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability.png", dpi=150)

    out = {
        "n_members": N_MEMBERS,
        "n_bootstrap": N_BOOTSTRAP,
        "log_loss_floor": 1e-3,
        "classes": list(classes),
        "scores": summary,
        "reliability": {name: t.to_dict(orient="list") for name, t in reliability.items()},
    }
    (RESULTS / "stage4_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'stage4_metrics.json'} and {FIGURES / 'reliability.png'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
