"""Read World Ocean Database yearly XBT files into a table with one row per cast."""

import numpy as np
import pandas as pd
import xarray as xr

UNKNOWN_TYPE = "XBT: TYPE UNKNOWN"

TEXT_COLUMNS = {
    "country": "country",
    "WOD_cruise_identifier": "cruise",
    "Platform": "platform",
    "Institute": "institute",
    "Temperature_Instrument": "instrument",
    "Recorder": "recorder",
    "depth_eq": "depth_eq",
    "needs_z_fix": "needs_z_fix",
}


def open_wod(path):
    # Decoding is off so that fill values and the 1770 time origin stay as stored.
    return xr.open_dataset(path, decode_times=False, mask_and_scale=False)


def cast_bounds(row_size):
    """Start and stop index of each cast in a contiguous ragged array."""
    stops = np.cumsum(row_size)
    return stops - row_size, stops


def _text(ds, name):
    if name not in ds.variables:
        return pd.Series([""] * ds.sizes["casts"])
    values = ds[name].values
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8", errors="replace")
    return pd.Series(values.astype(str)).str.strip()


def read_cast_table(path):
    with open_wod(path) as ds:
        z_rows = ds["z_row_size"].values.astype(np.int64)
        t_rows = ds["Temperature_row_size"].values.astype(np.int64)
        if z_rows.sum() != ds.sizes["z_obs"]:
            raise ValueError(f"{path}: z_row_size does not add up to the length of z")
        if t_rows.sum() != ds.sizes["Temperature_obs"]:
            raise ValueError(
                f"{path}: Temperature_row_size does not add up to the length of Temperature"
            )

        # The last level is not always the deepest, so take the maximum.
        max_depth = np.full(len(z_rows), np.nan)
        has_levels = z_rows > 0
        if has_levels.any():
            starts, _ = cast_bounds(z_rows)
            max_depth[has_levels] = np.maximum.reduceat(ds["z"].values, starts[has_levels])

        table = pd.DataFrame(
            {
                "cast_id": ds["wod_unique_cast"].values,
                "date": ds["date"].values,
                "lat": ds["lat"].values,
                "lon": ds["lon"].values,
                "n_levels": z_rows,
                "n_temperature": t_rows,
                "max_depth": max_depth,
            }
        )
        for source, column in TEXT_COLUMNS.items():
            table[column] = _text(ds, source)
        table.attrs["missing_variables"] = [
            name for name in TEXT_COLUMNS if name not in ds.variables
        ]

    table["year"] = table["date"] // 10000
    table["month"] = table["date"] // 100 % 100
    table["labelled"] = ~table["instrument"].isin(["", UNKNOWN_TYPE])
    table["has_cruise"] = table["cruise"] != ""
    table["surface_only"] = (table["n_levels"] == 1) & (table["max_depth"] == 0)
    return table


def read_profile(ds, index):
    """Depth and temperature arrays for the cast at position index."""
    z_start, z_stop = cast_bounds(ds["z_row_size"].values.astype(np.int64))
    t_start, t_stop = cast_bounds(ds["Temperature_row_size"].values.astype(np.int64))
    z = ds["z"][z_start[index] : z_stop[index]].values
    temperature = ds["Temperature"][t_start[index] : t_stop[index]].values
    return z, temperature


def check_cast_table(table):
    """Raise if the table breaks a rule that every cast must satisfy."""
    problems = []
    if table["cast_id"].duplicated().any():
        problems.append("duplicate cast ids")
    if not table["lat"].between(-90, 90).all():
        problems.append("latitude outside -90 to 90")
    if not table["lon"].between(-180, 180).all():
        problems.append("longitude outside -180 to 180")
    if not table["month"].between(1, 12).all():
        problems.append("month outside 1 to 12")
    if (table["n_levels"] < 0).any():
        problems.append("negative number of levels")
    if problems:
        raise ValueError("; ".join(problems))


def summarise(table):
    """Counts that describe data quality but do not make a file unusable."""
    labelled = table["labelled"]
    return {
        "casts": len(table),
        "labelled": int(labelled.sum()),
        "type_unknown": int((table["instrument"] == UNKNOWN_TYPE).sum()),
        "empty_type": int((table["instrument"] == "").sum()),
        "labelled_no_cruise": int((labelled & ~table["has_cruise"]).sum()),
        "surface_only": int(table["surface_only"].sum()),
        "levels_mismatch": int((table["n_levels"] != table["n_temperature"]).sum()),
        "deeper_than_2000m": int((table["max_depth"] > 2000).sum()),
    }


def save_cast_table(table, path):
    table.to_csv(path, index=False)


def load_cast_table(path):
    # Empty strings mean "not recorded" here and must not turn into NaN.
    text_columns = list(TEXT_COLUMNS.values())
    return pd.read_csv(
        path,
        dtype={column: str for column in text_columns},
        keep_default_na=False,
        na_values={"max_depth": [""]},
    )
