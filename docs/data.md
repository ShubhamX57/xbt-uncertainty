# Data

## Source

All profiles come from the World Ocean Database (WOD), maintained by the NOAA
National Centers for Environmental Information (NCEI). The project uses the
yearly XBT files for 1990 to 1999, one NetCDF file per year, as served by the
NCEI THREDDS server:

    https://www.ncei.noaa.gov/thredds-ocean/catalog/ncei/wod/catalog.html

Each file was downloaded from the HTTPServer link of that catalogue, for example

    https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/wod/1999/wod_xbt_1999.nc

with `scripts/download_wod.sh 1990 1999`. The files were downloaded on
10 November 2026. Their SHA-256 checksums are in `data/checksums.txt`. The 1999
file reports `date_created: 2025-06-25` in its global attributes.

NCEI updates WOD quarterly, so a later download may give different files and
different counts. Compare checksums before comparing results.

The raw files are 6.24 GB in total and are not stored in this repository.

## Licence and attribution

NOAA data distributed through the NOAA Open Data Dissemination programme is open
for any use. NOAA asks for attribution and does not allow modified data to be
presented as original, unaltered NOAA data. The cast table built here is derived
data, not a copy of WOD.

## File layout

A yearly file holds every XBT cast for that year as a contiguous ragged array.
Per-cast variables have the dimension `casts`. Depth `z` and `Temperature` are
stored end to end, and `z_row_size` and `Temperature_row_size` give the number
of levels in each cast. The variable names used here were read from the 1999
file's header through the THREDDS NcML service and confirmed on all ten files
by `scripts/build_cast_table.py`.

| WOD variable | Column in casts.csv | Use |
| --- | --- | --- |
| `wod_unique_cast` | `cast_id` | unique id of each cast |
| `date` | `date`, `year`, `month` | stored as YYYYMMDD |
| `lat`, `lon` | `lat`, `lon` | position |
| `z_row_size` | `n_levels` | levels per cast |
| `z` | `max_depth` | deepest level of the cast |
| `Temperature_Instrument` | `instrument` | probe type, the label |
| `WOD_cruise_identifier` | `cruise` | grouping key for the train and test split |
| `country`, `Platform`, `Institute`, `Recorder` | same names, lower case | candidate features |
| `depth_eq`, `needs_z_fix` | same names | depth equation status, for the correction stage |

## What the ten files contain

From `scripts/build_cast_table.py` run on the files above:

| | casts |
| --- | ---: |
| all casts | 662,844 |
| named probe type | 318,695 |
| `XBT: TYPE UNKNOWN` | 324,438 |
| empty instrument field | 19,711 |
| named type but no cruise id | 13,055 |
| single reading at 0 m | 13,883 |
| deepest level below 2000 m | 465 |

No file failed the structural checks: row sizes match array lengths, cast ids
are unique, coordinates and months are in range, and every cast is in the year
of its file.

## Known issues

- About 52% of casts have no usable probe type. The share falls from 75% in
  1990 to 20% in 1997, so the labelled set is not a random sample of the decade.
- The labelled and unlabelled casts differ. In 1999 the median deepest level is
  400 m for unknown types and 836 m for Sippican Deep Blue, and most unknown
  casts have no cruise id.
- The label vocabulary changes over time. `XBT: T7 (UNKNOWN BRAND)` and
  `XBT: T4 (UNKNOWN BRAND)` are common up to 1995 and absent from 1996.
- The instrument field also contains air-launched probes (`XBT: AXBT ...`).
- Some T4 and T7 casts report levels deeper than 2000 m, below the depth these
  probes are expected to reach. They are counted, not removed.
