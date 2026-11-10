"""Reader logic, tested on a small file with the same layout as a WOD yearly file."""

import numpy as np
import pytest
from conftest import CASTS, write_wod_file

from xbt_uncertainty.wod import (
    cast_bounds,
    check_cast_table,
    load_cast_table,
    open_wod,
    read_cast_table,
    read_profile,
    save_cast_table,
    summarise,
)


def test_cast_bounds_cover_the_ragged_array():
    starts, stops = cast_bounds(np.array([3, 1, 2]))
    assert starts.tolist() == [0, 3, 4]
    assert stops.tolist() == [3, 4, 6]


def test_one_row_per_cast(wod_file):
    table = read_cast_table(wod_file)
    assert table["cast_id"].tolist() == [101, 102, 103, 104, 105]
    assert table["n_levels"].tolist() == [3, 1, 2, 4, 1]


def test_max_depth_is_the_deepest_level_not_the_last(wod_file):
    table = read_cast_table(wod_file)
    assert table["max_depth"].tolist() == [50.0, 0.0, 400.0, 760.0, 4.0]


def test_year_and_month_come_from_the_date(wod_file):
    table = read_cast_table(wod_file)
    assert table["year"].unique().tolist() == [1999]
    assert table["month"].tolist() == [1, 1, 2, 7, 12]


def test_text_is_decoded_and_stripped(wod_file):
    table = read_cast_table(wod_file)
    assert table["instrument"][0] == "XBT: T4 (SIPPICAN)"
    assert table["country"][3] == "JAPAN"
    assert table["cruise"][2] == ""


def test_unknown_and_empty_types_are_unlabelled(wod_file):
    table = read_cast_table(wod_file)
    assert table["labelled"].tolist() == [True, False, False, True, True]


def test_cruise_and_surface_flags(wod_file):
    table = read_cast_table(wod_file)
    assert table["has_cruise"].tolist() == [True, True, False, True, False]
    assert table["surface_only"].tolist() == [False, True, False, False, False]


def test_summary_counts(wod_file):
    summary = summarise(read_cast_table(wod_file))
    assert summary == {
        "casts": 5,
        "labelled": 3,
        "type_unknown": 1,
        "empty_type": 1,
        "labelled_no_cruise": 1,
        "surface_only": 1,
        "levels_mismatch": 0,
        "deeper_than_2000m": 0,
    }


def test_missing_text_variable_gives_empty_column(tmp_path):
    path = write_wod_file(tmp_path / "no_institute.nc", drop=("Institute",))
    table = read_cast_table(path)
    assert (table["institute"] == "").all()
    assert table.attrs["missing_variables"] == ["Institute"]


def test_wrong_row_sizes_are_rejected(tmp_path):
    path = write_wod_file(tmp_path / "broken.nc", z_row_size=[3, 1, 2, 4, 2])
    with pytest.raises(ValueError, match="z_row_size"):
        read_cast_table(path)


def test_read_profile_returns_the_levels_of_one_cast(wod_file):
    with open_wod(wod_file) as ds:
        z, temperature = read_profile(ds, 3)
    assert z.tolist() == CASTS["z"][3]
    assert len(temperature) == len(z)
    np.testing.assert_allclose(temperature, 20.0 - z / 50.0)


def test_check_accepts_a_valid_table(wod_file):
    check_cast_table(read_cast_table(wod_file))


def test_check_rejects_duplicate_cast_ids(wod_file):
    table = read_cast_table(wod_file)
    table.loc[1, "cast_id"] = table.loc[0, "cast_id"]
    with pytest.raises(ValueError, match="duplicate cast ids"):
        check_cast_table(table)


def test_check_rejects_impossible_latitude(wod_file):
    table = read_cast_table(wod_file)
    table.loc[0, "lat"] = 91.0
    with pytest.raises(ValueError, match="latitude"):
        check_cast_table(table)


def test_csv_round_trip_keeps_empty_strings(wod_file, tmp_path):
    table = read_cast_table(wod_file)
    path = tmp_path / "casts.csv"
    save_cast_table(table, path)
    loaded = load_cast_table(path)

    assert loaded["cruise"].tolist() == table["cruise"].tolist()
    assert loaded["instrument"].tolist() == table["instrument"].tolist()
    assert loaded["labelled"].tolist() == table["labelled"].tolist()
    assert not loaded[["cruise", "instrument", "institute"]].isna().any().any()
    np.testing.assert_array_equal(loaded["max_depth"], table["max_depth"])
