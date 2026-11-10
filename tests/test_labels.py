import pandas as pd
import pytest

from xbt_uncertainty.labels import add_label_columns, parse_instrument

# Every instrument string found in the 1990 to 1999 files, with the expected parse.
VOCABULARY = {
    "XBT: T7 (SIPPICAN)": ("T7", "SIPPICAN", False),
    "XBT: DEEP BLUE (SIPPICAN)": ("DEEP BLUE", "SIPPICAN", False),
    "XBT: T7 (UNKNOWN BRAND)": ("T7", "", False),
    "XBT: T4 (SIPPICAN)": ("T4", "SIPPICAN", False),
    "XBT: T4 (UNKNOWN BRAND)": ("T4", "", False),
    "XBT: T7 (TSK - TSURUMI SEIKI Co.)": ("T7", "TSK", False),
    "XBT: T4 (TSK - TSURUMI SEIKI Co.)": ("T4", "TSK", False),
    "XBT: T6 (TSK - TSURUMI SEIKI Co.)": ("T6", "TSK", False),
    "XBT: T5 (SIPPICAN)": ("T5", "SIPPICAN", False),
    "XBT: T10 (SIPPICAN)": ("T10", "SIPPICAN", False),
    "XBT: T6 (SIPPICAN)": ("T6", "SIPPICAN", False),
    "XBT: XBT-7 (SPARTON)": ("XBT-7", "SPARTON", False),
    "XBT: DEEP BLUE, UNKNOWN BRAND": ("DEEP BLUE", "", False),
    "XBT: T5 (UNKNOWN BRAND)": ("T5", "", False),
    "XBT: T6 (UNKNOWN BRAND)": ("T6", "", False),
    "XBT: T10 (UNKNOWN BRAND)": ("T10", "", False),
    "XBT: FAST DEEP (SIPPICAN)": ("FAST DEEP", "SIPPICAN", False),
    "XBT: FAST DEEP, UNKNOWN BRAND": ("FAST DEEP", "", False),
    "XBT: DEEP BLUE (TSK - TSURUMI SEIKI Co.)": ("DEEP BLUE", "TSK", False),
    "XBT: T5 (TSK - TSURUMI SEIKI Co.)": ("T5", "TSK", False),
    "XBT: AXBT A-A/3 (SPARTON)": ("AXBT", "SPARTON", True),
    "XBT: XBT-10 (SPARTON)": ("XBT-10", "SPARTON", False),
    "XBT: AXBT (UNKNOWN BRAND AND TYPE)": ("AXBT", "", True),
    "XBT: T11 (SIPPICAN)": ("T11", "SIPPICAN", False),
    "XBT: AXBT A/3 (HERMES)": ("AXBT", "HERMES", True),
    "XBT: XBT-6 (SPARTON)": ("XBT-6", "SPARTON", False),
    "XBT: XBT-3 (SPARTON)": ("XBT-3", "SPARTON", False),
    "XBT: XBT-20 (SPARTON)": ("XBT-20", "SPARTON", False),
    "XBT: XBT-5 (SPARTON)": ("XBT-5", "SPARTON", False),
    "XBT: XBT-4 (SPARTON)": ("XBT-4", "SPARTON", False),
    "XBT: AXBT A/3 (SPARTON-HERMES)": ("AXBT", "SPARTON-HERMES", True),
    "XBT: T11 (UNKNOWN BRAND)": ("T11", "", False),
    "XBT: T10 (TSK - TSURUMI SEIKI Co.)": ("T10", "TSK", False),
    "XBT: XBT-1 (SPARTON)": ("XBT-1", "SPARTON", False),
    "XBT: XBT-5DB (SPARTON)": ("XBT-5DB", "SPARTON", False),
}


@pytest.mark.parametrize("text, expected", VOCABULARY.items())
def test_every_string_in_the_decade_parses(text, expected):
    assert parse_instrument(text) == expected


@pytest.mark.parametrize("text", ["", "XBT: TYPE UNKNOWN"])
def test_unlabelled_strings_give_empty_labels(text):
    assert parse_instrument(text) == ("", "", False)


@pytest.mark.parametrize(
    "text",
    [
        "XBT T7 (SIPPICAN)",
        "XBT: T7",
        "XBT: T7 (ACME)",
        "XBT: T7, SIPPICAN",
        "CTD: SBE 911",
    ],
)
def test_unexpected_strings_are_rejected(text):
    with pytest.raises(ValueError):
        parse_instrument(text)


def test_add_label_columns_keeps_rows_and_order():
    table = pd.DataFrame(
        {
            "cast_id": [1, 2, 3, 4],
            "instrument": [
                "XBT: T7 (SIPPICAN)",
                "XBT: TYPE UNKNOWN",
                "XBT: T7 (SIPPICAN)",
                "XBT: DEEP BLUE, UNKNOWN BRAND",
            ],
        }
    )
    out = add_label_columns(table)
    assert out["cast_id"].tolist() == [1, 2, 3, 4]
    assert out["model"].tolist() == ["T7", "", "T7", "DEEP BLUE"]
    assert out["manufacturer"].tolist() == ["SIPPICAN", "", "SIPPICAN", ""]
    assert out["is_axbt"].tolist() == [False, False, False, False]
    assert "model" not in table.columns
