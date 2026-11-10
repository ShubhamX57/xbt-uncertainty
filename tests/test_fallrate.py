import numpy as np
import pytest

from xbt_uncertainty.fallrate import (
    HANAWA_1994,
    HANAWA_GROUP,
    MANUFACTURER_T4_T6_T7,
    NO_CORRECTION_GROUP,
    depth_from_time,
    hanawa_eq12,
    hanawa_from_manufacturer,
    time_from_depth,
)
from xbt_uncertainty.profiles import clean_profile, on_grid, sample_profiles


def test_time_and_depth_are_inverse():
    t = np.linspace(0, 120, 50)
    for a, b in [MANUFACTURER_T4_T6_T7, HANAWA_1994]:
        np.testing.assert_allclose(time_from_depth(depth_from_time(t, a, b), a, b), t, atol=1e-9)


def test_conversion_matches_the_papers_own_formula():
    # Eq. 12 of Hanawa et al. (1994) was derived from the same two equations,
    # so this also checks the coefficients were transcribed correctly.
    z = np.linspace(0, 900, 91)
    np.testing.assert_allclose(hanawa_from_manufacturer(z), hanawa_eq12(z), atol=0.05)


def test_conversion_is_about_three_percent_deeper():
    # The paper's linear approximation is Z = 1.0336 z.
    z = np.array([100.0, 400.0, 800.0])
    np.testing.assert_allclose(hanawa_from_manufacturer(z) / z, 1.0336, atol=0.002)
    assert hanawa_from_manufacturer(0.0) == 0.0


def test_groups_do_not_overlap():
    assert not HANAWA_GROUP & NO_CORRECTION_GROUP


def test_clean_profile_sorts_and_drops_repeats():
    z, t = clean_profile(np.array([10.0, 0.0, 10.0, 5.0]), np.array([3.0, 1.0, 4.0, 2.0]))
    assert z.tolist() == [0.0, 5.0, 10.0]
    assert t.tolist() == [1.0, 2.0, 3.0]


def test_on_grid_is_nan_outside_the_profile():
    out = on_grid(np.array([0.0, 100.0]), np.array([20.0, 10.0]), np.array([50.0, 150.0]))
    assert out[0] == 15.0
    assert np.isnan(out[1])


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_certain_probabilities_give_no_spread(p):
    z = np.linspace(0, 700, 71)
    temperature = 25 - z / 40
    grid = np.arange(0, 600, 10.0)
    samples, _ = sample_profiles(z, temperature, p, grid, 50, np.random.default_rng(0))
    assert np.nanstd(samples, axis=0).max() < 1e-10


def test_spread_matches_a_two_point_mixture():
    z = np.linspace(0, 700, 141)
    temperature = 25 - z / 40
    grid = np.arange(0, 600, 10.0)
    p = 0.3
    samples, _ = sample_profiles(z, temperature, p, grid, 20_000, np.random.default_rng(1))
    corrected = on_grid(hanawa_from_manufacturer(z), temperature, grid)
    stored = on_grid(z, temperature, grid)
    expected = np.sqrt(p * (1 - p)) * np.abs(corrected - stored)
    np.testing.assert_allclose(np.nanstd(samples, axis=0), expected, rtol=0.03, atol=1e-6)


def test_corrected_profile_is_warmer_at_a_fixed_depth_when_temperature_falls():
    # Readings move deeper, so the reading now placed at a given depth was
    # stored higher up, where the water is warmer.
    z = np.linspace(0, 700, 141)
    temperature = 25 - z / 40
    grid = np.array([200.0, 500.0])
    corrected = on_grid(hanawa_from_manufacturer(z), temperature, grid)
    stored = on_grid(z, temperature, grid)
    assert (corrected > stored).all()
