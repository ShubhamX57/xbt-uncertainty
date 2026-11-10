"""Checks against the downloaded 1999 file. Skipped when the data is not present."""

import hashlib
from pathlib import Path

import pytest

from xbt_uncertainty.wod import check_cast_table, read_cast_table, summarise

ROOT = Path(__file__).resolve().parents[1]
FILE_1999 = ROOT / "data" / "raw" / "wod_xbt_1999.nc"
CHECKSUMS = ROOT / "data" / "checksums.txt"

pytestmark = pytest.mark.skipif(not FILE_1999.exists(), reason="WOD data not downloaded")


@pytest.fixture(scope="module")
def table_1999():
    return read_cast_table(FILE_1999)


def test_1999_file_matches_recorded_checksum():
    recorded = dict(
        reversed(line.split()) for line in CHECKSUMS.read_text().splitlines() if line.strip()
    )
    digest = hashlib.sha256()
    with open(FILE_1999, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == recorded[FILE_1999.name]


def test_1999_passes_checks(table_1999):
    check_cast_table(table_1999)
    assert table_1999.attrs["missing_variables"] == []
    assert (table_1999["year"] == 1999).all()


def test_1999_counts_match_first_inspection(table_1999):
    # Numbers recorded from the file created by NCEI on 2025-06-25.
    assert summarise(table_1999) == {
        "casts": 56034,
        "labelled": 36561,
        "type_unknown": 18667,
        "empty_type": 806,
        "labelled_no_cruise": 1166,
        "surface_only": 4301,
        "levels_mismatch": 0,
        "deeper_than_2000m": 3,
    }
