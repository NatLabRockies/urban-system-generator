"""A project's ResStock ``County and PUMA`` from its Site Origin coordinates, offline.

The packaged ``resources/location/county_puma_tracts.npz`` stores, for every 2010 Census tract, its population center
and its ``County and PUMA``; ResStock uses 2010 geography. The project gets the combination of the nearest tract
center. Tracts are small (about 4,000 people), so the nearest center almost always lies in the site's own county and
PUMA. Post-processing then derives County, PUMA, City, AHS Region, ISO/RTO and the other location columns from it.
The file is rebuilt with ``scripts/build_county_puma_lookup.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from usg.location import haversine_km
from usg.resources import county_puma_lookup_path

COUNTY_AND_PUMA = "County and PUMA"
MAX_DISTANCE_KM = 25.0  # farther from every tract center of the state: not used


@dataclass(frozen=True)
class TractMatch:
    county_and_puma: str
    state: str
    distance_km: float


class CountyPumaLookup:
    def __init__(self, latitude: np.ndarray, longitude: np.ndarray, combo: np.ndarray, combos: np.ndarray,
                 states: np.ndarray, meta: dict):
        self.latitude, self.longitude, self.combo = latitude.astype(np.float64), longitude.astype(np.float64), combo
        self.combos, self.states, self.meta = combos, states, meta
        self._tract_state = states[combo]

    @classmethod
    def load(cls, path: str | Path | None = None) -> CountyPumaLookup:
        return _load(str(path or county_puma_lookup_path()))

    def nearest(self, latitude: float, longitude: float, *, state: str | None = None,
                max_km: float = MAX_DISTANCE_KM) -> TractMatch | None:
        """County and PUMA of the nearest tract center (within ``state`` when given) at most ``max_km`` away."""
        dlat = max_km / 111.0
        dlon = dlat / max(math.cos(math.radians(latitude)), 0.2)
        near = (np.abs(self.latitude - latitude) <= dlat) & (np.abs(self.longitude - longitude) <= dlon)
        if state:
            near &= self._tract_state == state
        idx = np.flatnonzero(near)
        if idx.size == 0:
            return None
        distances = haversine_km(latitude, longitude, self.latitude[idx], self.longitude[idx])
        best = int(np.argmin(distances))
        if distances[best] > max_km:
            return None
        combo = self.combo[idx[best]]
        return TractMatch(str(self.combos[combo]), str(self.states[combo]), float(distances[best]))


@lru_cache(maxsize=2)
def _load(path: str) -> CountyPumaLookup:
    with np.load(path, allow_pickle=False) as data:
        return CountyPumaLookup(data["latitude"], data["longitude"], data["combo"], data["combos"], data["states"],
                                json.loads(str(data["meta"])))
