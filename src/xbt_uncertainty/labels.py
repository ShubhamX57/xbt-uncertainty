"""Split the WOD Temperature_Instrument string into probe model and manufacturer."""

import re

import pandas as pd

from xbt_uncertainty.wod import UNKNOWN_TYPE

# Two layouts occur in the 1990s files:
#   "XBT: T7 (SIPPICAN)"
#   "XBT: DEEP BLUE, UNKNOWN BRAND"
PATTERN = re.compile(r"^XBT: (?P<model>.+?)(?: \((?P<paren>[^()]*)\)|, (?P<comma>UNKNOWN BRAND))$")

MANUFACTURERS = {
    "SIPPICAN": "SIPPICAN",
    "TSK - TSURUMI SEIKI Co.": "TSK",
    "SPARTON": "SPARTON",
    "HERMES": "HERMES",
    "SPARTON-HERMES": "SPARTON-HERMES",
    "UNKNOWN BRAND": "",
    "UNKNOWN BRAND AND TYPE": "",
}


def parse_instrument(text):
    """Return (model, manufacturer, is_axbt) for one instrument string.

    Unlabelled casts give ("", "", False). An empty manufacturer means the
    brand was not recorded. Strings in an unexpected layout raise ValueError,
    so a change in the WOD vocabulary cannot pass unnoticed.
    """
    if text in ("", UNKNOWN_TYPE):
        return "", "", False
    match = PATTERN.match(text)
    if match is None:
        raise ValueError(f"unrecognised instrument string: {text!r}")
    raw_model = match["model"]
    raw_maker = match["paren"] if match["paren"] is not None else match["comma"]
    if raw_maker not in MANUFACTURERS:
        raise ValueError(f"unrecognised manufacturer in {text!r}")

    # Air-launched probes carry a variant after the name, e.g. "AXBT A-A/3".
    is_axbt = raw_model.split()[0] == "AXBT"
    model = "AXBT" if is_axbt else raw_model
    return model, MANUFACTURERS[raw_maker], is_axbt


def add_label_columns(table):
    """Add model, manufacturer and is_axbt columns, parsing each distinct string once."""
    unique = table["instrument"].unique()
    parsed = pd.DataFrame(
        [parse_instrument(text) for text in unique],
        index=unique,
        columns=["model", "manufacturer", "is_axbt"],
    )
    out = table.copy()
    for column in parsed.columns:
        out[column] = out["instrument"].map(parsed[column])
    return out
