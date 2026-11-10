#!/bin/sh
# Download WOD yearly XBT files from the NCEI THREDDS server.
# Usage: sh scripts/download_wod.sh 1990 1999
# Run from the project root. Files that are already present are skipped,
# and an interrupted download resumes when the script is run again.

set -eu

first_year=$1
last_year=$2
base=https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/wod

mkdir -p data/raw

year=$first_year
while [ "$year" -le "$last_year" ]; do
    file=wod_xbt_$year.nc
    if [ -f "data/raw/$file" ]; then
        echo "$file already present, skipping"
    else
        echo "downloading $file"
        curl -L --fail -C - -o "data/raw/$file.part" "$base/$year/$file"
        mv "data/raw/$file.part" "data/raw/$file"
    fi
    year=$((year + 1))
done

cd data/raw
shasum -a 256 wod_xbt_*.nc > ../checksums.txt
ls -l wod_xbt_*.nc
cat ../checksums.txt
