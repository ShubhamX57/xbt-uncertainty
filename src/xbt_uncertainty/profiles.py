"""Ensembles of depth-corrected temperature profiles."""

import numpy as np

from xbt_uncertainty.fallrate import hanawa_from_manufacturer
from xbt_uncertainty.wod import cast_bounds


class ProfileReader:
    """Read single casts from an open WOD yearly file, keeping good levels only."""

    def __init__(self, ds):
        self.ds = ds
        self.z_start, self.z_stop = cast_bounds(ds["z_row_size"].values.astype(np.int64))
        self.t_start, self.t_stop = cast_bounds(
            ds["Temperature_row_size"].values.astype(np.int64)
        )
        self.cast_ids = ds["wod_unique_cast"].values

    def read(self, index):
        z_slice = slice(self.z_start[index], self.z_stop[index])
        t_slice = slice(self.t_start[index], self.t_stop[index])
        z = self.ds["z"][z_slice].values.astype(float)
        temperature = self.ds["Temperature"][t_slice].values.astype(float)
        good = (
            (self.ds["z_WODflag"][z_slice].values == 0)
            & (self.ds["Temperature_WODflag"][t_slice].values == 0)
            & (temperature > -1e9)
        )
        return clean_profile(z[good], temperature[good])


def clean_profile(z, temperature):
    """Sort by depth and drop repeated depths, as interpolation needs."""
    order = np.argsort(z, kind="stable")
    z, temperature = z[order], temperature[order]
    keep = np.concatenate([[True], np.diff(z) > 0])
    return z[keep], temperature[keep]


def on_grid(z, temperature, grid):
    """Temperature at the grid depths, NaN outside the measured range."""
    out = np.interp(grid, z, temperature)
    out[(grid < z[0]) | (grid > z[-1])] = np.nan
    return out


def sample_profiles(z, temperature, p_hanawa, grid, n_samples, rng):
    """Profiles on the grid for n_samples probe types drawn from the classifier.

    Each draw is in the Hanawa group with probability p_hanawa, in which case
    the stored depths are converted to Hanawa et al. (1994); otherwise they
    are left as stored.
    """
    corrected = on_grid(hanawa_from_manufacturer(z), temperature, grid)
    stored = on_grid(z, temperature, grid)
    in_hanawa = rng.random(n_samples) < p_hanawa
    return np.where(in_hanawa[:, None], corrected, stored), in_hanawa
