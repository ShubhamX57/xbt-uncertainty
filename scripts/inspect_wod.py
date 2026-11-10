"""Print the structure and key metadata of one WOD yearly XBT file.

Usage: python scripts/inspect_wod.py data/raw/wod_xbt_1999.nc
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

TEXT_VARIABLES = [
    "Temperature_Instrument",
    "country",
    "dataset",
    "depth_eq",
    "needs_z_fix",
    "Recorder",
    "Institute",
    "Platform",
    "WOD_cruise_identifier",
]
SHOW_ALL = {"Temperature_Instrument", "depth_eq", "needs_z_fix", "dataset"}


def as_text(values):
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8", errors="replace")
    return pd.Series(values.astype(str)).str.strip()


def describe_text(ds, name):
    if name not in ds.variables:
        print(f"\n{name}: not in file")
        return
    text = as_text(ds[name].values)
    counts = text.value_counts()
    empty = (text == "").mean()
    print(f"\n{name}: {len(counts)} distinct values, {empty:.1%} empty")
    limit = None if name in SHOW_ALL else 10
    for value, n in counts.head(limit).items():
        print(f"  {n:>8}  {value!r}")


def main(path):
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)

    print("File:", path)
    for key in ["title", "date_created", "id", "Conventions"]:
        print(f"  {key}: {ds.attrs.get(key)}")

    print("\nDimensions")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print("\nVariables")
    for name, var in ds.variables.items():
        print(f"  {name}  {var.dims}  {var.dtype}")

    for name in TEXT_VARIABLES:
        describe_text(ds, name)

    print("\nPosition and date")
    for name in ["lat", "lon", "date", "time"]:
        values = ds[name].values
        print(f"  {name}: min {values.min()}, max {values.max()}")
    print(f"  time units: {ds['time'].attrs.get('units')}")

    print("\nRagged array checks")
    z_rows = ds["z_row_size"].values.astype(np.int64)
    t_rows = ds["Temperature_row_size"].values.astype(np.int64)
    print(f"  casts: {len(z_rows)}")
    print(f"  sum of z_row_size: {z_rows.sum()}, z_obs: {ds.sizes['z_obs']}")
    print(
        f"  sum of Temperature_row_size: {t_rows.sum()}, "
        f"Temperature_obs: {ds.sizes['Temperature_obs']}"
    )
    print(f"  casts where the two row sizes differ: {(z_rows != t_rows).sum()}")
    print(f"  casts with zero depth levels: {(z_rows == 0).sum()}")
    print(f"  unique cast ids: {len(np.unique(ds['wod_unique_cast'].values))}")

    z = ds["z"].values
    ends = np.cumsum(z_rows)
    has_levels = z_rows > 0
    max_depth = z[ends[has_levels] - 1]
    quantiles = np.quantile(max_depth, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    print("\nDeepest level per cast (m)")
    print("  quantiles 0, 5, 25, 50, 75, 95, 100:")
    print("  " + ", ".join(f"{q:.1f}" for q in quantiles))

    ds.close()


if __name__ == "__main__":
    main(sys.argv[1])
