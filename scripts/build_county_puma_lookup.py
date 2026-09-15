#!/usr/bin/env python
"""Build ``resources/location/county_puma_tracts.npz`` to locate a project's Site Origin offline.

For every 2010 Census tract the table stores its population center and its ResStock ``County and PUMA``.

Inputs are public domain from the U.S. Census Bureau (URLs are downloaded, paths are read):
* ``CenPop2010_Mean_TR.txt``: 2010 Census tract centers of population.
* ``2010_Census_Tract_to_2010_PUMA.txt``: the 2010 tract to 2010 PUMA relationship file.

ResStock labels ``County and PUMA`` with 2010 geography GISJOIN codes (``G2405100, G24000804``). Tracts whose
combination is not an option in options_lookup.tsv are dropped, and their neighbours cover the area.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from usg.resources import county_puma_lookup_path, options_lookup_path

TRACT_CENTERS_URL = "https://www2.census.gov/geo/docs/reference/cenpop2010/tract/CenPop2010_Mean_TR.txt"
TRACT_PUMA_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt"
# county FIPS codes changed after 2010 -> the code ResStock uses
COUNTY_RENAMES = {"46113": "46102", "51515": "51019"}  # Shannon -> Oglala Lakota County, SD; Bedford city -> County, VA
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def _read(source: str) -> tuple[str, str]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=120) as response:
            data = response.read()
    else:
        data = Path(source).read_bytes()
    return data.decode("utf-8-sig"), hashlib.sha256(data).hexdigest()


def _county_puma_options(lookup: Path) -> set[str]:
    with open(lookup, newline="", encoding="utf-8") as f:
        return {row[1].strip() for row in csv.reader(f, delimiter="\t") if len(row) > 1 and row[0] == "County and PUMA"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tract-centers", default=TRACT_CENTERS_URL, help="CenPop2010_Mean_TR.txt (path or URL)")
    parser.add_argument("--tract-puma", default=TRACT_PUMA_URL, help="2010_Census_Tract_to_2010_PUMA.txt (path or URL)")
    parser.add_argument("--options-lookup", type=Path, default=None, help="default: the packaged options_lookup.tsv")
    parser.add_argument("--out", type=Path, default=None, help="default: resources/location/county_puma_tracts.npz")
    args = parser.parse_args(argv)

    centers_text, centers_sha = _read(args.tract_centers)
    relation_text, relation_sha = _read(args.tract_puma)
    valid = _county_puma_options(args.options_lookup or options_lookup_path())
    puma_of = {(r["STATEFP"], r["COUNTYFP"], r["TRACTCE"]): r["PUMA5CE"] for r in csv.DictReader(io.StringIO(relation_text))}

    lat, lon, keys = [], [], []
    no_puma = not_in_lookup = 0
    for r in csv.DictReader(io.StringIO(centers_text)):
        state_fips = r["STATEFP"]
        puma = puma_of.get((state_fips, r["COUNTYFP"], r["TRACTCE"]))
        if puma is None:
            no_puma += 1
            continue
        county = COUNTY_RENAMES.get(state_fips + r["COUNTYFP"], state_fips + r["COUNTYFP"])
        key = f"G{county[:2]}0{county[2:]}0, G{state_fips}0{puma}"
        if key not in valid:
            not_in_lookup += 1
            continue
        lat.append(float(r["LATITUDE"]))
        lon.append(float(r["LONGITUDE"]))
        keys.append(key)

    combos = sorted(set(keys))
    index = {k: i for i, k in enumerate(combos)}
    states = [STATE_FIPS[k[1:3]] for k in combos]
    meta = {
        "format": "usg-county-puma-tracts/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "geography": "2010 Census tracts and 2010 PUMAs (the vintage ResStock uses)",
        "license": "public domain (U.S. Census Bureau)",
        "sources": {"tract_centers": {"source": args.tract_centers, "sha256": centers_sha},
                    "tract_puma": {"source": args.tract_puma, "sha256": relation_sha}},
        "n_tracts": len(keys), "n_county_puma": len(combos), "county_puma_options": len(valid),
        "dropped": {"no_puma": no_puma, "not_in_options_lookup": not_in_lookup},
    }
    out = args.out or county_puma_lookup_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, latitude=np.asarray(lat, dtype=np.float32), longitude=np.asarray(lon, dtype=np.float32),
        combo=np.asarray([index[k] for k in keys], dtype=np.uint16), combos=np.asarray(combos),
        states=np.asarray(states), meta=np.asarray(json.dumps(meta)),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {len(keys):,} tracts, {len(combos):,} of "
          f"{len(valid):,} County and PUMA options; dropped {no_puma} without PUMA, {not_in_lookup} not in options_lookup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
