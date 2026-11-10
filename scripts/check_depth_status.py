"""Which depth equation produced the stored depths, by probe model and label status.

Usage: python scripts/check_depth_status.py
"""

import pandas as pd

from xbt_uncertainty.labels import add_label_columns
from xbt_uncertainty.wod import load_cast_table

CASTS = "data/processed/casts.csv"


def show(title, rows, columns):
    print(f"\n{title}")
    print(pd.crosstab(rows, columns, margins=True).to_string())


def main():
    casts = add_label_columns(load_cast_table(CASTS))
    casts = casts[~casts["is_axbt"]]
    casts["status"] = casts["model"].where(casts["labelled"], "(unlabelled)")
    fix = casts["needs_z_fix"].replace("", "(empty)")
    eq = casts["depth_eq"].replace("", "(empty)")

    show("needs_z_fix by model", casts["status"], fix)
    show("depth_eq by model", casts["status"], eq)
    show("needs_z_fix against depth_eq, all casts", fix, eq)

    unlabelled = ~casts["labelled"]
    show("Unlabelled casts: needs_z_fix by year", casts.loc[unlabelled, "year"], fix[unlabelled])
    show("Unlabelled casts: depth_eq by year", casts.loc[unlabelled, "year"], eq[unlabelled])

    deep = unlabelled & ~casts["surface_only"]
    print("\nUnlabelled casts with more than a surface reading")
    print(f"  casts: {deep.sum()}, with a cruise id: {(deep & casts['has_cruise']).sum()}")
    print("  deepest level quantiles 5, 25, 50, 75, 95 (m):")
    q = casts.loc[deep, "max_depth"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print("  " + ", ".join(f"{v:.0f}" for v in q))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    main()
