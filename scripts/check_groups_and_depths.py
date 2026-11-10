"""Check cruise grouping and profile depths in one WOD yearly XBT file.

Usage: python scripts/check_groups_and_depths.py data/raw/wod_xbt_1999.nc
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

UNKNOWN = "XBT: TYPE UNKNOWN"


def text(ds, name):
    values = ds[name].values
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8", errors="replace")
    return pd.Series(values.astype(str)).str.strip()


def purity(labelled, key):
    grouped = labelled[labelled[key] != ""].groupby(key)["instrument"]
    types_per_group = grouped.nunique()
    majority = grouped.agg(lambda s: s.value_counts().iloc[0])
    sizes = grouped.size()
    print(f"\n  grouped by {key}")
    print(f"    groups: {len(sizes)}, labelled casts in them: {sizes.sum()}")
    print(f"    groups with exactly one probe type: {(types_per_group == 1).mean():.1%}")
    print(f"    casts matching their group's most common type: {majority.sum() / sizes.sum():.1%}")
    print("    probe types per group:")
    for n_types, n_groups in types_per_group.value_counts().sort_index().items():
        print(f"      {n_types} types: {n_groups} groups")
    q = sizes.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    print("    group size min, 25%, median, 75%, max: " + ", ".join(f"{v:.0f}" for v in q))


def main(path):
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)

    casts = pd.DataFrame(
        {
            "instrument": text(ds, "Temperature_Instrument"),
            "cruise": text(ds, "WOD_cruise_identifier"),
            "orig_cruise": text(ds, "originators_cruise_identifier"),
            "platform": text(ds, "Platform"),
            "country": text(ds, "country"),
            "rows": ds["z_row_size"].values.astype(np.int64),
        }
    )
    casts["label"] = "labelled"
    casts.loc[casts["instrument"] == UNKNOWN, "label"] = "type unknown"
    casts.loc[casts["instrument"] == "", "label"] = "empty"
    for key in ["cruise", "orig_cruise", "platform"]:
        casts[f"has_{key}"] = casts[key] != ""

    print("Label status")
    print(casts["label"].value_counts().to_string())

    print("\nLabel status against grouping fields")
    for key in ["has_cruise", "has_orig_cruise", "has_platform"]:
        print()
        print(pd.crosstab(casts["label"], casts[key]).to_string())

    no_cruise = casts[~casts["has_cruise"]]
    print(f"\nCasts without a WOD cruise identifier: {len(no_cruise)}")
    print(pd.crosstab(no_cruise["has_platform"], no_cruise["has_orig_cruise"]).to_string())
    print("  top countries:")
    print(no_cruise["country"].value_counts().head(5).to_string())

    print("\nDo casts in one group share a probe type? (labelled casts only)")
    labelled = casts[casts["label"] == "labelled"]
    purity(labelled, "cruise")
    purity(labelled, "platform")

    rows = casts["rows"].to_numpy()
    if (rows == 0).any():
        raise SystemExit("some casts have no depth levels; this script assumes none do")
    z = ds["z"].values
    ends = np.cumsum(rows)
    starts = ends - rows
    casts["first_z"] = z[starts]
    casts["last_z"] = z[ends - 1]
    casts["max_z"] = np.maximum.reduceat(z, starts)

    print("\nDepth levels per cast")
    q = casts["rows"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
    print("  quantiles 0, 5, 25, 50, 75, 95, 100: " + ", ".join(f"{v:.0f}" for v in q))
    print(f"  casts with a single level: {(casts['rows'] == 1).sum()}")
    print(f"  casts where the last level is not the deepest: {(casts['last_z'] != casts['max_z']).sum()}")
    print(f"  casts whose first level is not 0 m: {(casts['first_z'] != 0).sum()}")
    print(f"  depth values below 0 m: {(z < 0).sum()}")
    temperature = ds["Temperature"].values
    print(f"  temperature fill values: {(temperature < -1e9).sum()}")

    for name, mask in [
        ("Deepest level exactly 0 m", casts["max_z"] == 0),
        ("Deepest level between 0 and 10 m", (casts["max_z"] > 0) & (casts["max_z"] < 10)),
        ("Deepest level above 2000 m", casts["max_z"] > 2000),
    ]:
        subset = casts[mask]
        print(f"\n{name}: {len(subset)} casts")
        if subset.empty:
            continue
        print("  levels per cast:")
        print(subset["rows"].value_counts().head(5).to_string())
        print("  label status:")
        print(subset["label"].value_counts().to_string())
        print("  instrument:")
        print(subset["instrument"].value_counts().head(5).to_string())
        print("  country:")
        print(subset["country"].value_counts().head(5).to_string())

    print("\nDeepest level by probe type (m): 5%, median, 95%, count")
    by_type = casts.groupby("instrument")["max_z"]
    summary = by_type.quantile([0.05, 0.5, 0.95]).unstack()
    summary["count"] = by_type.size()
    print(summary.sort_values("count", ascending=False).round(0).to_string())

    ds.close()


if __name__ == "__main__":
    main(sys.argv[1])
