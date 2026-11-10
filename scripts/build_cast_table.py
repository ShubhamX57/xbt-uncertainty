"""Build one table of cast metadata from all yearly files in data/raw.

Usage: python scripts/build_cast_table.py
"""

from pathlib import Path

import pandas as pd

from xbt_uncertainty.wod import check_cast_table, read_cast_table, save_cast_table, summarise

RAW = Path("data/raw")
OUTPUT = Path("data/processed/casts.csv")


def main():
    files = sorted(RAW.glob("wod_xbt_*.nc"))
    if not files:
        raise SystemExit(f"no files found in {RAW}")

    tables = []
    rows = []
    for path in files:
        table = read_cast_table(path)
        check_cast_table(table)
        file_year = int(path.stem.split("_")[-1])
        summary = summarise(table)
        summary["wrong_year"] = int((table["year"] != file_year).sum())
        summary["missing_variables"] = ",".join(table.attrs["missing_variables"]) or "-"
        rows.append({"file": path.name, **summary})
        table["source_file"] = path.name
        tables.append(table)

    casts = pd.concat(tables, ignore_index=True)
    check_cast_table(casts)

    print("Per file")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nAll files")
    for key, value in summarise(casts).items():
        print(f"  {key}: {value}")

    labelled = casts[casts["labelled"]]
    print(f"\nCruises with labelled casts: {labelled.loc[labelled['has_cruise'], 'cruise'].nunique()}")

    print("\nProbe types, all files")
    print(casts["instrument"].value_counts().to_string())

    print("\nLabelled casts per year for the six most common types")
    top = labelled["instrument"].value_counts().head(6).index
    print(pd.crosstab(labelled["year"], labelled["instrument"])[top].to_string())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_cast_table(casts, OUTPUT)
    print(f"\nWrote {len(casts)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
