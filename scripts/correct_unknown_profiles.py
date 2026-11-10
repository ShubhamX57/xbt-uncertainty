"""Stage 5: spread in corrected profiles caused by not knowing the probe model.

For unlabelled casts from 1990 to 1995 that WOD flags as needing the
Hanawa et al. (1994) correction, draw probe models from the ensemble's
probabilities, correct the depths of each draw, and measure the spread in
temperature at fixed depths.

Usage: python scripts/correct_unknown_profiles.py
Needs results/stage3_metrics.json and the raw files in data/raw.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from xbt_uncertainty.calibration import brier_score, expected_calibration_error, reliability_table
from xbt_uncertainty.dataset import (
    FEATURES,
    feature_encoder,
    labelled_ship_casts,
    merge_rare_models,
    split_by_cruise,
)
from xbt_uncertainty.ensemble import CruiseBaggedTrees
from xbt_uncertainty.fallrate import HANAWA_GROUP, NO_CORRECTION_GROUP, hanawa_from_manufacturer
from xbt_uncertainty.labels import add_label_columns
from xbt_uncertainty.profiles import ProfileReader, on_grid, sample_profiles
from xbt_uncertainty.wod import load_cast_table, open_wod

CASTS = Path("data/processed/casts.csv")
RAW = Path("data/raw")
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
NEEDS_HANAWA = "needs Hanawa et al.; 1994 applied (XBT)"
YEARS = range(1990, 1996)
N_MEMBERS = 50
N_CASTS = 500
N_SAMPLES = 200
MIN_LEVELS = 10
GRID = np.arange(0.0, 1000.0, 5.0)
SEED = 0

# Grid depths below a profile are NaN, and NumPy warns about all-NaN columns.
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom", category=RuntimeWarning)


def check_groups(casts):
    """Confirm the correction groups against WOD's own flags on labelled casts."""
    labelled = add_label_columns(casts)
    labelled = labelled[labelled["labelled"] & ~labelled["is_axbt"]]
    flagged = labelled.groupby("model")["needs_z_fix"].agg(
        casts="size", needs_hanawa=lambda s: (s == NEEDS_HANAWA).sum()
    )
    flagged["share"] = flagged["needs_hanawa"] / flagged["casts"]
    print("Labelled casts flagged as needing Hanawa, by model")
    print(flagged.round(3).to_string())
    for model in HANAWA_GROUP:
        assert flagged.loc[model, "share"] > 0.1, f"WOD rarely flags {model} for Hanawa"
    for model in NO_CORRECTION_GROUP - {"OTHER"}:
        assert flagged.loc[model, "share"] < 0.01, f"WOD flags {model} for Hanawa"


def check_flag_meanings(path):
    with open_wod(path) as ds:
        for name in ["z_WODflag", "Temperature_WODflag"]:
            meanings = ds[name].attrs.get("flag_meanings", "")
            print(f"{name}: values {ds[name].attrs.get('flag_values')}, meanings: {meanings}")
            assert meanings.split()[0].lower().startswith("accepted"), (
                f"{name}: expected flag 0 to mean accepted"
            )


def main():
    stage3 = json.loads((RESULTS / "stage3_metrics.json").read_text())
    rng = np.random.default_rng(SEED)
    casts = load_cast_table(CASTS)
    check_groups(casts)
    check_flag_meanings(RAW / "wod_xbt_1990.nc")

    table, _ = merge_rare_models(labelled_ship_casts(casts), stage3["min_cruises"])
    is_test = split_by_cruise(table, seed=stage3["seed"])
    train, test = table[~is_test], table[is_test]
    tree = make_pipeline(
        feature_encoder(),
        DecisionTreeClassifier(max_depth=stage3["max_depth"], random_state=stage3["seed"]),
    )
    ensemble = CruiseBaggedTrees(tree, n_members=N_MEMBERS, seed=stage3["seed"])
    ensemble.fit(train[FEATURES], train["target"], groups=train["cruise"])
    hanawa_columns = np.isin(ensemble.classes_, sorted(HANAWA_GROUP))
    assert set(ensemble.classes_) <= HANAWA_GROUP | NO_CORRECTION_GROUP

    # The quantity that decides the correction is the group, not the model,
    # so check that its probability is calibrated on the test cruises.
    p_test = ensemble.average_proba(test[FEATURES])[:, hanawa_columns].sum(axis=1)
    in_group = test["target"].isin(HANAWA_GROUP).to_numpy()
    two_class = np.column_stack([1 - p_test, p_test])
    confidence = np.maximum(p_test, 1 - p_test)
    correct = (p_test >= 0.5) == in_group
    print("\nProbability of the Hanawa group, test cruises")
    print(f"  casts: {len(test)}, share in the group: {in_group.mean():.3f}")
    print(f"  brier (two classes, 0 to 2): {brier_score(in_group.astype(int), two_class):.4f}")
    print(f"  ece: {expected_calibration_error(confidence, correct):.4f}")
    group_reliability = reliability_table(p_test, in_group)
    print(group_reliability.round(3).to_string(index=False))

    all_casts = add_label_columns(casts)
    all_casts["file_index"] = all_casts.groupby("source_file").cumcount()
    eligible = all_casts[
        ~all_casts["labelled"]
        & all_casts["year"].isin(YEARS)
        & (all_casts["needs_z_fix"] == NEEDS_HANAWA)
        & (all_casts["n_levels"] >= MIN_LEVELS)
    ].reset_index(drop=True)
    eligible["p_hanawa"] = ensemble.average_proba(eligible[FEATURES])[:, hanawa_columns].sum(axis=1)
    p = eligible["p_hanawa"]
    print(f"\nUnlabelled casts 1990 to 1995 flagged as needing Hanawa, {MIN_LEVELS}+ levels: {len(eligible)}")
    print(f"  with a cruise id: {eligible['has_cruise'].mean():.1%}")
    print(f"  mean probability of the Hanawa group: {p.mean():.3f}")
    for low, high in [(0.99, 1.01), (0.9, 0.99), (0.5, 0.9), (0.0, 0.5)]:
        print(f"  share with probability in [{low}, {high}): {((p >= low) & (p < high)).mean():.1%}")

    chosen = eligible.iloc[rng.choice(len(eligible), size=min(N_CASTS, len(eligible)), replace=False)]
    spreads, full_shift, examples = [], [], {}
    for source_file, rows in chosen.groupby("source_file"):
        with open_wod(RAW / source_file) as ds:
            reader = ProfileReader(ds)
            for _, cast in rows.iterrows():
                assert reader.cast_ids[cast["file_index"]] == cast["cast_id"]
                z, temperature = reader.read(cast["file_index"])
                if len(z) < MIN_LEVELS:
                    continue
                samples, _ = sample_profiles(
                    z, temperature, cast["p_hanawa"], GRID, N_SAMPLES, rng
                )
                spreads.append(np.nanstd(samples, axis=0))
                corrected = on_grid(hanawa_from_manufacturer(z), temperature, GRID)
                full_shift.append(corrected - on_grid(z, temperature, GRID))
                examples[cast["cast_id"]] = (z, temperature, cast["p_hanawa"], samples)
    spreads = np.array(spreads)
    full_shift = np.array(full_shift)
    print(f"\nProfiles used: {len(spreads)} of {len(chosen)} sampled, {N_SAMPLES} draws each")

    report_depths = [50, 100, 200, 300, 400, 500, 700]
    rows = []
    for depth in report_depths:
        i = int(np.searchsorted(GRID, depth))
        s = spreads[:, i]
        f = full_shift[:, i]
        rows.append(
            {
                "depth_m": depth,
                "profiles": int(np.isfinite(s).sum()),
                "spread_median": np.nanmedian(s),
                "spread_p95": np.nanpercentile(s, 95),
                "spread_max": np.nanmax(s),
                "hanawa_shift_median": np.nanmedian(f),
            }
        )
    summary = pd.DataFrame(rows)
    print("\nStandard deviation of temperature across draws (degrees C), and the")
    print("full Hanawa correction at that depth for comparison")
    print(summary.round(4).to_string(index=False))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(p, bins=np.linspace(0, 1, 41), color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("probability of a Hanawa-group model")
    axes[0].set_ylabel("unlabelled casts")
    axes[0].set_title("Unlabelled casts, 1990 to 1995")
    axes[1].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    axes[1].plot(group_reliability["confidence"], group_reliability["accuracy"], marker="o")
    axes[1].set_xlabel("predicted probability of the Hanawa group")
    axes[1].set_ylabel("observed share in the group")
    axes[1].set_title("Labelled test cruises")
    fig.tight_layout()
    fig.savefig(FIGURES / "hanawa_group_probability.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.fill_betweenx(GRID, np.nanpercentile(spreads, 5, axis=0), np.nanpercentile(spreads, 95, axis=0), alpha=0.3, label="5 to 95% across profiles")
    ax.plot(np.nanmedian(spreads, axis=0), GRID, label="median across profiles")
    ax.plot(np.nanmedian(np.abs(full_shift), axis=0), GRID, color="0.4", linestyle="--", label="full Hanawa shift, median")
    ax.invert_yaxis()
    ax.set_xlabel("temperature spread across draws (degrees C)")
    ax.set_ylabel("depth (m)")
    ax.set_ylim(800, 0)
    # Below the axes, so it cannot cover the curves.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIGURES / "profile_spread.png", dpi=150, bbox_inches="tight")

    cast_id = min(examples, key=lambda k: abs(examples[k][2] - 0.5))
    z, temperature, p_cast, samples = examples[cast_id]
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(on_grid(z, temperature, GRID), GRID, color="0.4", label="depths as stored")
    ax.plot(on_grid(hanawa_from_manufacturer(z), temperature, GRID), GRID, color="tab:red", label="Hanawa et al. (1994)")
    ax.fill_betweenx(GRID, np.nanpercentile(samples, 5, axis=0), np.nanpercentile(samples, 95, axis=0), alpha=0.3, label="5 to 95% of draws")
    ax.invert_yaxis()
    ax.set_xlabel("temperature (degrees C)")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"Cast {cast_id}, P(Hanawa group) = {p_cast:.2f}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIGURES / "example_cast.png", dpi=150, bbox_inches="tight")

    out = {
        "years": [min(YEARS), max(YEARS)],
        "n_eligible": int(len(eligible)),
        "n_profiles": int(len(spreads)),
        "n_samples": N_SAMPLES,
        "mean_p_hanawa": float(p.mean()),
        "test_group_brier": brier_score(in_group.astype(int), two_class),
        "test_group_ece": expected_calibration_error(confidence, correct),
        "spread_by_depth": summary.to_dict(orient="list"),
        "example_cast": int(cast_id),
    }
    (RESULTS / "stage5_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'stage5_metrics.json'} and three figures in {FIGURES}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
