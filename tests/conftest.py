import netCDF4
import numpy as np
import pytest

TEXT_VARIABLES = {
    "country": "strnlensmall",
    "WOD_cruise_identifier": "strnlensmall",
    "Platform": "strnlen",
    "Institute": "strnlen",
    "Temperature_Instrument": "strnlen",
    "Recorder": "strnlen",
    "depth_eq": "strnlen",
    "needs_z_fix": "strnlen",
}

# Five casts laid out like a WOD yearly file. Cast 1 has its deepest level
# in the middle, cast 2 is a single surface reading.
CASTS = {
    "wod_unique_cast": [101, 102, 103, 104, 105],
    "date": [19990105, 19990105, 19990220, 19990711, 19991231],
    "lat": [10.0, 10.5, -33.25, 48.0, 0.0],
    "lon": [-150.0, -150.5, 18.5, -20.0, 180.0],
    "z": [[0.0, 50.0, 20.0], [0.0], [0.6, 400.0], [0.0, 250.0, 500.0, 760.0], [4.0]],
    "country": ["UNITED STATES", "UNITED STATES", "UNKNOWN", "JAPAN", "JAPAN"],
    "WOD_cruise_identifier": ["US000001", "US000001", "", "JP000007", ""],
    "Platform": ["SHIP A", "SHIP A", "", "SHIP B", ""],
    "Institute": ["", "", "", "AGENCY", ""],
    "Temperature_Instrument": [
        "XBT: T4 (SIPPICAN)",
        "XBT: TYPE UNKNOWN",
        "",
        "XBT: T7 (SIPPICAN)",
        "XBT: T7 (SIPPICAN)",
    ],
    "Recorder": ["", "", "", "SIPPICAN MK-12", ""],
    "depth_eq": ["original", "", "", "corrected", ""],
    "needs_z_fix": ["no fix necessary"] * 5,
}


def write_wod_file(path, casts=CASTS, drop=(), z_row_size=None):
    rows = [len(levels) for levels in casts["z"]]
    z = np.concatenate(casts["z"]).astype("float32")
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("casts", len(rows))
        nc.createDimension("z_obs", len(z))
        nc.createDimension("Temperature_obs", len(z))
        nc.createDimension("strnlen", 170)
        nc.createDimension("strnlensmall", 40)

        for name, dim in TEXT_VARIABLES.items():
            if name in drop:
                continue
            width = nc.dimensions[dim].size
            var = nc.createVariable(name, "S1", ("casts", dim))
            var[:] = netCDF4.stringtochar(np.array(casts[name], dtype=f"S{width}"))

        nc.createVariable("wod_unique_cast", "i4", ("casts",))[:] = casts["wod_unique_cast"]
        nc.createVariable("date", "i4", ("casts",))[:] = casts["date"]
        nc.createVariable("lat", "f4", ("casts",))[:] = casts["lat"]
        nc.createVariable("lon", "f4", ("casts",))[:] = casts["lon"]
        nc.createVariable("z_row_size", "i4", ("casts",))[:] = z_row_size or rows
        nc.createVariable("Temperature_row_size", "i4", ("casts",))[:] = rows
        nc.createVariable("z", "f4", ("z_obs",))[:] = z
        nc.createVariable("Temperature", "f4", ("Temperature_obs",))[:] = 20.0 - z / 50.0
    return path


@pytest.fixture
def wod_file(tmp_path):
    return write_wod_file(tmp_path / "wod_xbt_1999.nc")
