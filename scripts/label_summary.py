"""Count casts and cruises per probe model, to choose how to handle rare classes.

Usage: python scripts/label_summary.py
"""

import pandas as pd

from xbt_uncertainty.labels import add_label_columns
from xbt_uncertainty.wod import load_cast_table

CASTS = "data/processed/casts.csv"


def main():
    casts = add_label_columns(load_cast_table(CASTS))
    labelled = casts[casts["labelled"]]

    print("Labelled casts")
    print(f"  all: {len(labelled)}")
    print(f"  air-launched (AXBT): {labelled['is_axbt'].sum()}")
    ship = labelled[~labelled["is_axbt"]]
    print(f"  ship-launched without a cruise id: {(~ship['has_cruise']).sum()}")
    usable = ship[ship["has_cruise"]]
    print(f"  usable for training and testing: {len(usable)}")
    print(f"  cruises: {usable['cruise'].nunique()}")

    print("\nPer model (usable casts)")
    per_model = usable.groupby("model").agg(
        casts=("cast_id", "size"),
        cruises=("cruise", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        no_brand=("manufacturer", lambda s: (s == "").mean()),
    )
    per_model["share"] = per_model["casts"] / per_model["casts"].sum()
    per_model = per_model.sort_values("casts", ascending=False)
    print(per_model.to_string(formatters={"share": "{:.2%}".format, "no_brand": "{:.0%}".format}))

    print("\nPer manufacturer (usable casts)")
    print(usable["manufacturer"].replace("", "(not recorded)").value_counts().to_string())

    print("\nCruises with more than one model")
    models_per_cruise = usable.groupby("cruise")["model"].nunique()
    print(f"  {(models_per_cruise > 1).mean():.1%} of cruises")
    majority = usable.groupby("cruise")["model"].agg(lambda s: s.value_counts().iloc[0])
    print(f"  casts matching their cruise's most common model: {majority.sum() / len(usable):.1%}")

    print("\nUsable casts per year")
    print(usable["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
