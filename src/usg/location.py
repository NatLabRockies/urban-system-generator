"""Location conditioning inputs: State, climate zones, Census Division and PUMA Metro Status.

The models condition on five location columns. :func:`resolve_location` fills them from whatever is known — explicit
values, an EnergyPlus weather filename, coordinates (for a GeoJSON project, only its Site Origin point) — using a
packaged lookup of ResStock 2024.2 weather stations and states (``resources/location/location_lookup.json``). Only
ResStock labels are ever produced; anything uncertain stays unknown and the model masks it. See ``docs/location.md``.

Precedence per field (first available wins):

* State: explicit > weather filename ``USA_XX_…`` > nearest weather station (share >= 0.6)
* ASHRAE IECC Climate Zone 2004: explicit > nearest station (>= 0.6, one of the state's zones) > from state (>= 0.9)
* Building America Climate Zone: explicit > implied by the ASHRAE zone (one BA zone for >= 99% of its homes) >
  nearest station (>= 0.6) > from state (>= 0.9); derived values must co-occur with the ASHRAE zone
* Census Division: from the state
* PUMA Metro Status: explicit only
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from usg.constants import CLIMATE_ZONE_COLUMN, STATE_COLUMN, STATES
from usg.errors import InputValidationError, USGError

LOCATION_LOOKUP_FORMAT = "usg-location-lookup/v1"

BUILDING_AMERICA_COLUMN = "Building America Climate Zone"
CENSUS_DIVISION_COLUMN = "Census Division"
PUMA_METRO_COLUMN = "PUMA Metro Status"
LOCATION_COLUMNS = (
    STATE_COLUMN, CLIMATE_ZONE_COLUMN, BUILDING_AMERICA_COLUMN, CENSUS_DIVISION_COLUMN, PUMA_METRO_COLUMN,
)  # fmt: skip

STATION_MIN_SHARE = 0.6  # nearest-station value used when it covers >= 60% of the station's homes
STATE_MIN_SHARE = 0.9  # state-wide value used when it covers >= 90% of the state's homes
ASHRAE_TO_BA_MIN_SHARE = 0.99  # an ASHRAE zone implies a Building America zone covering >= 99% of its homes
MAX_STATION_KM = 150.0  # coordinates farther than this from every station are not used
COLOCATED_STATION_KM = 2.0  # stations this close to the nearest one are pooled (duplicate records of one airport)
CONTIGUOUS_US_BOUNDS = (24.3, 49.5, -125.0, -66.8)  # latitude min, max; longitude min, max
EARTH_RADIUS_KM = 6371.0088

SOURCE_EXPLICIT = "explicit"
SOURCE_FROM_STATE = "from state"
SOURCE_INPUT = "input"
SOURCE_STATION = "nearest station"

_WEATHER_STATE_RE = re.compile(r"^USA_([A-Z]{2})_")
_NAMES = {
    STATE_COLUMN: "state",
    CLIMATE_ZONE_COLUMN: "climate zone",
    BUILDING_AMERICA_COLUMN: "Building America climate zone",
    CENSUS_DIVISION_COLUMN: "census division",
    PUMA_METRO_COLUMN: "PUMA metro status",
}
_SHORT_KEYS = {"state": STATE_COLUMN, "ashrae": CLIMATE_ZONE_COLUMN, "ba": BUILDING_AMERICA_COLUMN,
               "division": CENSUS_DIVISION_COLUMN, "metro": PUMA_METRO_COLUMN}


# ==================================================================================================
# Helpers
# ==================================================================================================


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees (numpy broadcasting)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def state_from_weather_filename(weather_filename: str | None) -> str | None:
    """``.../USA_NY_Buffalo-Greater.Buffalo.Intl.AP.725280_TMY3.epw`` -> ``"NY"``."""
    if not weather_filename:
        return None
    m = _WEATHER_STATE_RE.match(_basename(weather_filename))
    return m.group(1) if m and m.group(1) in STATES else None


def in_contiguous_us(latitude: float, longitude: float) -> bool:
    """Coarse bounding box of the 48 contiguous states + DC (ocean and border areas are caught by the distance check)."""
    lat_min, lat_max, lon_min, lon_max = CONTIGUOUS_US_BOUNDS
    return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


def source_label(source: str | None) -> str | None:
    """``"weather file (USA_MD_….epw)"`` -> ``"weather file"`` (sources are ``label`` or ``label (detail)``)."""
    return None if source is None else source.split(" (", 1)[0]


def _basename(path) -> str:
    return re.split(r"[\\/]", str(path).strip())[-1]


def _km(distance: float) -> str:
    return f"{distance:.1f}" if distance < 10 else f"{distance:.0f}"


def _shares_text(shares: Mapping[str, float], limit: int = 3) -> str:
    return ", ".join(f"{label} {share:.0%}" for label, share in list(shares.items())[:limit])


def _dominant(shares: Mapping[str, float] | None) -> tuple[str, float] | None:
    return max(shares.items(), key=lambda kv: kv[1]) if shares else None


def _present(value: Any) -> bool:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(str(value).strip())


def _number(value: Any) -> float | None:
    if not _present(value) or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _coordinates(latitude, longitude) -> tuple[float, float] | tuple[None, None]:
    if latitude is None and longitude is None:
        return None, None
    if latitude is None or longitude is None:
        raise InputValidationError("latitude and longitude must be given together")
    lat, lon = _number(latitude), _number(longitude)
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise InputValidationError(
            f"invalid coordinates ({latitude}, {longitude}): latitude must be within [-90, 90] and longitude within "
            "[-180, 180] degrees"
        )
    return lat, lon


def _outside_hint(latitude: float, longitude: float) -> str:
    if 18.5 <= latitude <= 22.5 and -160.5 <= longitude <= -154.5:
        return " (this looks like Hawaii)"
    if 51.0 <= latitude <= 72.0 and (longitude <= -129.9 or longitude >= 172.0):
        return " (this looks like Alaska)"
    return ""


@cache
def _resstock_options(column: str) -> tuple[str, ...]:
    from usg.geojson import _allowed_options  # shared, cached options_lookup.tsv parse

    return tuple(_allowed_options().get(column, ()))


def _census_division_of(state: str) -> str | None:
    from usg.postprocessor import STATE_TO_CENSUS_DIVISION

    return STATE_TO_CENSUS_DIVISION.get(state)


# ==================================================================================================
# Lookup data
# ==================================================================================================


@dataclass(frozen=True)
class Station:
    """A ResStock weather station and the building-weighted label shares of the homes simulated with its weather."""

    name: str
    latitude: float
    longitude: float
    n_buildings: int
    shares: Mapping[str, Mapping[str, float]]  # column -> {label: share}, most common first

    def dominant(self, column: str) -> tuple[str, float] | None:
        return _dominant(self.shares.get(column))


@dataclass(frozen=True)
class StateStats:
    """Building-weighted label shares of one state's homes."""

    code: str
    census_division: str
    n_buildings: int
    shares: Mapping[str, Mapping[str, float]]

    def dominant(self, column: str) -> tuple[str, float] | None:
        return _dominant(self.shares.get(column))


class LocationLookup:
    """ResStock weather-station and state statistics (format ``usg-location-lookup/v1``)."""

    def __init__(self, data: Mapping[str, Any]):
        if data.get("format") != LOCATION_LOOKUP_FORMAT:
            raise USGError(f"unsupported location lookup format {data.get('format')!r} (expected {LOCATION_LOOKUP_FORMAT!r})")
        try:
            keys = {short: column for short, column in data["columns"].items()}

            def shares(record: Mapping) -> dict[str, dict[str, float]]:
                return {keys[k]: {str(label): float(s) for label, s in v.items()}
                        for k, v in record.items() if k in keys and isinstance(v, Mapping)}

            self.data_release: str | None = data.get("data_release")
            self.license: str | None = data.get("license")
            self.attribution: str | None = data.get("attribution")
            self.source: dict = dict(data.get("source") or {})
            self.vocab: dict[str, tuple[str, ...]] = {c: tuple(v) for c, v in data["vocab"].items()}
            self.stations: tuple[Station, ...] = tuple(
                Station(s["name"], float(s["lat"]), float(s["lon"]), int(s["n"]), shares(s)) for s in data["stations"]
            )
            self.states: dict[str, StateStats] = {
                code: StateStats(code, s["division"], int(s["n"]), shares(s)) for code, s in data["states"].items()
            }
            self.ashrae_to_ba: dict[str, dict[str, float]] = {
                zone: {str(k): float(v) for k, v in m.items()} for zone, m in data["ashrae_to_ba"].items()
            }
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise USGError(f"malformed location lookup: {type(exc).__name__}: {exc}") from None
        self._validate(keys)
        self._lat = np.array([s.latitude for s in self.stations], dtype=float)
        self._lon = np.array([s.longitude for s in self.stations], dtype=float)

    def _validate(self, keys: Mapping[str, str]) -> None:
        if set(keys.values()) != set(LOCATION_COLUMNS) or not self.stations:
            raise USGError("malformed location lookup: unexpected columns or no stations")
        bad = []
        records = [*self.stations, *self.states.values()]
        for record in records:
            for column, labels in record.shares.items():
                bad += [f"{column}={label!r}" for label in labels if label not in self.vocab.get(column, ())]
        bad += [f"State={s.code!r}" for s in self.states.values() if s.code not in self.vocab.get(STATE_COLUMN, ())]
        bad += [f"Census Division={s.census_division!r}" for s in self.states.values()
                if s.census_division not in self.vocab.get(CENSUS_DIVISION_COLUMN, ())]
        for zone, bas in self.ashrae_to_ba.items():
            bad += [f"{CLIMATE_ZONE_COLUMN}={zone!r}"] if zone not in self.vocab.get(CLIMATE_ZONE_COLUMN, ()) else []
            bad += [f"{BUILDING_AMERICA_COLUMN}={b!r}" for b in bas if b not in self.vocab.get(BUILDING_AMERICA_COLUMN, ())]
        if bad:
            raise USGError(f"malformed location lookup: labels outside its vocab: {sorted(set(bad))[:5]}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> LocationLookup:
        """The packaged lookup (loaded once and cached), or the lookup file at ``path``."""
        if path is None:
            return _packaged_lookup()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise USGError(f"cannot read location lookup {path}: {exc}") from None
        return cls(data)

    def labels(self, column: str) -> tuple[str, ...]:
        """Labels of ``column`` present in the lookup data (a subset of the bundle vocab)."""
        return self.vocab.get(column, ())

    def state(self, code: str | None) -> StateStats | None:
        return self.states.get(code) if code else None

    def nearest_stations(self, latitude: float, longitude: float, k: int = 3) -> list[tuple[Station, float]]:
        """The ``k`` stations closest to a point (degrees) with their great-circle distances in km, nearest first."""
        lat, lon = _coordinates(latitude, longitude)
        distances = haversine_km(lat, lon, self._lat, self._lon)
        order = np.argsort(distances, kind="stable")[: max(0, min(int(k), len(self.stations)))]
        return [(self.stations[i], float(distances[i])) for i in order]

    def building_america_for(self, ashrae_zone: str | None) -> tuple[str, float] | None:
        """``(BA zone, share)`` when the ASHRAE zone implies one BA zone for >= 99% of its homes, else None."""
        dominant = _dominant(self.ashrae_to_ba.get(ashrae_zone)) if ashrae_zone else None
        return dominant if dominant and dominant[1] >= ASHRAE_TO_BA_MIN_SHARE else None

    def zones_compatible(self, ashrae_zone: str | None, building_america_zone: str | None) -> bool:
        """False only when both zones are known and never co-occur in the data."""
        if not ashrae_zone or not building_america_zone or ashrae_zone not in self.ashrae_to_ba:
            return True
        return building_america_zone in self.ashrae_to_ba[ashrae_zone]


@lru_cache(maxsize=1)
def _packaged_lookup() -> LocationLookup:
    from usg.resources import location_lookup_path

    return LocationLookup.load(location_lookup_path())


def canonical_location_label(column: str, value: Any, lookup: LocationLookup | None = None) -> str | None:
    """The ResStock label for ``value`` in a location column (case-insensitive), or None if it is not a valid option.

    Valid labels are the lookup's (ResStock 2024.2 data) plus the column's other ``options_lookup.tsv`` options, such as
    ``AK``, ``7AK`` or ``Subarctic``, which the models have no data for.
    """
    if column not in LOCATION_COLUMNS:
        raise ValueError(f"not a location column: {column!r}")
    if not _present(value) or isinstance(value, bool):
        return None
    text = str(value).strip().strip('"').strip()
    lookup = lookup if lookup is not None else LocationLookup.load()
    low = text.lower()
    for labels in (lookup.labels(column), _resstock_options(column)):
        if text in labels:
            return text
        for label in labels:
            if label.lower() == low:
                return label
    return None


# ==================================================================================================
# Resolution
# ==================================================================================================


def format_location_summary(
    values: Mapping[str, str],
    sources: Mapping[str, str | None],
    nearest_station: str | None = None,
    station_distance_km: float | None = None,
) -> str:
    """``Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state)``."""
    parts = []
    for column in LOCATION_COLUMNS:
        value = values.get(column)
        if value is None:
            if column in (STATE_COLUMN, CLIMATE_ZONE_COLUMN):
                parts.append(f"{_NAMES[column]} unknown")
            continue
        label = source_label(sources.get(column))
        parts.append(f"{value} ({label})" if label else str(value))
    line = "Location: " + " · ".join(parts)
    if nearest_station and station_distance_km is not None and any(
        source_label(s) == SOURCE_STATION for s in sources.values()
    ):
        line += f" · nearest station {nearest_station} ({_km(station_distance_km)} km)"
    return line


@dataclass(frozen=True)
class LocationInfo:
    """Resolved location columns, where each value came from, and the warnings raised while resolving them.

    Sources are ``"<label>"`` or ``"<label> (<detail>)"``: ``explicit``, ``input``, ``weather file (USA_MD_….epw)``,
    ``nearest station (<name>, <km> km)``, ``from state (KY 100%)``, ``from 4A``, ``from state``.
    """

    state: str | None = None
    ashrae_zone: str | None = None
    building_america_zone: str | None = None
    census_division: str | None = None
    puma_metro_status: str | None = None
    state_source: str | None = None
    ashrae_zone_source: str | None = None
    building_america_zone_source: str | None = None
    census_division_source: str | None = None
    puma_metro_status_source: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    nearest_station: str | None = None
    station_distance_km: float | None = None
    warnings: tuple[str, ...] = ()

    FIELDS: ClassVar[tuple[tuple[str, str], ...]] = (
        (STATE_COLUMN, "state"),
        (CLIMATE_ZONE_COLUMN, "ashrae_zone"),
        (BUILDING_AMERICA_COLUMN, "building_america_zone"),
        (CENSUS_DIVISION_COLUMN, "census_division"),
        (PUMA_METRO_COLUMN, "puma_metro_status"),
    )

    def values(self) -> dict[str, str]:
        """Known location columns: ``{column: label}``."""
        return {column: getattr(self, attr) for column, attr in self.FIELDS if getattr(self, attr)}

    def sources(self) -> dict[str, str]:
        """``{column: source}`` for the known columns."""
        return {column: getattr(self, f"{attr}_source") for column, attr in self.FIELDS if getattr(self, attr)}

    def summary(self) -> str:
        return format_location_summary(self.values(), self.sources(), self.nearest_station, self.station_distance_km)

    def to_json(self) -> dict:
        return {
            "summary": self.summary(),
            "values": self.values(),
            "sources": self.sources(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "nearest_station": self.nearest_station,
            "nearest_station_km": None if self.station_distance_km is None else round(self.station_distance_km, 1),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LocationOverrides:
    """Location values the user gave explicitly (CLI options or API); they win over values derived from the input."""

    state: str | None = None
    climate_zone: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @property
    def given(self) -> bool:
        return any(v is not None for v in (self.state, self.climate_zone, self.latitude, self.longitude))

    def kwargs(self) -> dict[str, Any]:
        return {"state": self.state, "climate_zone": self.climate_zone, "latitude": self.latitude,
                "longitude": self.longitude}

    def to_json(self) -> dict:
        return {k: v for k, v in self.kwargs().items() if v is not None}


def _explicit_label(column: str, value: Any, lookup: LocationLookup) -> str | None:
    if not _present(value):
        return None
    label = canonical_location_label(column, value, lookup)
    if label is None:
        if column == STATE_COLUMN:
            hint = "use a two-letter code such as MD"
        elif column == CLIMATE_ZONE_COLUMN:
            hint = "use an ASHRAE IECC 2004 climate zone such as 4A"
        else:
            hint = "use one of " + ", ".join(repr(v) for v in _resstock_options(column) or lookup.labels(column))
        raise InputValidationError(f"invalid {_NAMES[column]} {value!r}: {hint}")
    return label


def _pool(stations: list[Station]) -> Station:
    """Merge co-located station records (weighted by buildings) under the nearest one's name and position."""
    if len(stations) == 1:
        return stations[0]
    total = sum(s.n_buildings for s in stations)
    shares: dict[str, dict[str, float]] = {}
    for column in dict.fromkeys(c for s in stations for c in s.shares):
        acc: Counter[str] = Counter()
        for s in stations:
            for label, share in s.shares.get(column, {}).items():
                acc[label] += share * s.n_buildings / total
        shares[column] = dict(sorted(acc.items(), key=lambda kv: (-kv[1], kv[0])))
    first = stations[0]
    return Station(first.name, first.latitude, first.longitude, total, shares)


def _derived(
    column: str,
    site: Station | None,
    site_source: str | None,
    state: StateStats | None,
    compatible: Callable[[str], bool],
) -> tuple[str | None, str | None, str | None]:
    """``(value, source, reason_if_unknown)`` from the nearest station (>= 0.6), else the state majority (>= 0.9)."""
    reason = None
    dominant = site.dominant(column) if site is not None else None
    if dominant is not None:
        if dominant[1] >= STATION_MIN_SHARE and compatible(dominant[0]):
            return dominant[0], site_source, None
        if dominant[1] < STATION_MIN_SHARE:
            reason = (f"the nearest weather station ({site.name}) is ambiguous "
                      f"({_shares_text(site.shares[column])})")
        else:
            reason = (f"the nearest weather station's value {dominant[0]} ({site.name}) is inconsistent with the "
                      "other location values")
    dominant = state.dominant(column) if state is not None else None
    if dominant is not None and dominant[1] >= STATE_MIN_SHARE and compatible(dominant[0]):
        return dominant[0], f"{SOURCE_FROM_STATE} ({state.code} {dominant[1]:.0%})", None
    return None, None, reason


def resolve_location(
    *,
    state: str | None = None,
    climate_zone: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    weather_filename: str | None = None,
    building_america_zone: str | None = None,
    puma_metro_status: str | None = None,
    lookup: LocationLookup | None = None,
) -> LocationInfo:
    """Resolve the five location conditioning columns (see the module docstring for the precedence).

    ``state``, ``climate_zone``, ``building_america_zone`` and ``puma_metro_status`` are explicit values (invalid labels
    raise :class:`InputValidationError`). ``weather_filename`` names the state (``USA_<ST>_…``). Coordinates outside
    the contiguous US or more than 150 km from every ResStock weather station are not used.
    """
    lookup = lookup if lookup is not None else LocationLookup.load()
    warnings: list[str] = []

    def warn(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    explicit_state = _explicit_label(STATE_COLUMN, state, lookup)
    explicit_zone = _explicit_label(CLIMATE_ZONE_COLUMN, climate_zone, lookup)
    explicit_ba = _explicit_label(BUILDING_AMERICA_COLUMN, building_america_zone, lookup)
    explicit_metro = _explicit_label(PUMA_METRO_COLUMN, puma_metro_status, lookup)
    lat, lon = _coordinates(latitude, longitude)
    weather_state = state_from_weather_filename(weather_filename)

    # ---- coordinates -> nearest weather station (co-located duplicate records pooled) ----------------------------
    site: Station | None = None
    site_source = nearest_name = nearest_km = None
    if lat is not None:
        if not in_contiguous_us(lat, lon):
            warn(f"coordinates ({lat:.4f}, {lon:.4f}) are outside the contiguous United States{_outside_hint(lat, lon)}; "
                 "ResStock 2024.2 only covers the 48 contiguous states and DC, so they are not used")
        else:
            nearest = lookup.nearest_stations(lat, lon, k=8)
            station, nearest_km = nearest[0]
            nearest_name = station.name
            if nearest_km > MAX_STATION_KM:
                warn(f"coordinates ({lat:.4f}, {lon:.4f}) are {nearest_km:.0f} km from the nearest ResStock weather "
                     f"station ({station.name}), more than {MAX_STATION_KM:.0f} km; they are not used")
            else:
                colocated = [s for s, _ in nearest[1:] if float(haversine_km(
                    station.latitude, station.longitude, s.latitude, s.longitude)) <= COLOCATED_STATION_KM]
                site = _pool([station, *colocated])
                site_source = f"{SOURCE_STATION} ({station.name}, {_km(nearest_km)} km)"

    # ---- State -----------------------------------------------------------------------------------------------------
    site_state = site.dominant(STATE_COLUMN) if site is not None else None
    confident_site_state = site_state[0] if site_state and site_state[1] >= STATION_MIN_SHARE else None
    state_source = None
    if explicit_state:
        resolved_state, state_source = explicit_state, SOURCE_EXPLICIT
    elif weather_state:
        resolved_state, state_source = weather_state, f"weather file ({_basename(weather_filename)})"
        if confident_site_state and confident_site_state != weather_state:
            warn(f"the weather file state {weather_state} disagrees with the coordinates: the nearest weather station "
                 f"({nearest_name}, {_km(nearest_km)} km) serves {_shares_text(site.shares[STATE_COLUMN])}; "
                 f"using {weather_state}")
    elif confident_site_state:
        resolved_state, state_source = confident_site_state, site_source
    else:
        resolved_state = None
        if site_state is not None:
            warn(f"state unknown: the nearest weather station ({nearest_name}) serves homes in several states "
                 f"({_shares_text(site.shares[STATE_COLUMN])}); give the state explicitly")
    state_stats = lookup.state(resolved_state)
    if resolved_state and state_stats is None:
        warn(f"the models have no data for {resolved_state} (ResStock 2024.2 covers the 48 contiguous states and DC); "
             f"State {resolved_state} is kept but the models cannot use it")

    # ---- Census Division -------------------------------------------------------------------------------------------
    division = None
    if resolved_state:
        division = state_stats.census_division if state_stats is not None else _census_division_of(resolved_state)

    # ---- ASHRAE IECC Climate Zone 2004 -----------------------------------------------------------------------------
    def fits_state(column: str) -> Callable[[str], bool]:
        return lambda label: state_stats is None or label in state_stats.shares.get(column, {label: 1.0})

    zone, zone_source = explicit_zone, SOURCE_EXPLICIT
    if not zone:
        zone, zone_source, reason = _derived(CLIMATE_ZONE_COLUMN, site, site_source, state_stats,
                                             fits_state(CLIMATE_ZONE_COLUMN))
        if zone is None and reason:
            warn(f"climate zone unknown: {reason}")

    # ---- Building America Climate Zone -----------------------------------------------------------------------------
    ba, ba_source = explicit_ba, SOURCE_EXPLICIT if explicit_ba else None
    if ba:
        if not lookup.zones_compatible(zone, ba):
            warn(f"Building America climate zone {ba} never occurs with climate zone {zone} in ResStock 2024.2")
    elif (implied := lookup.building_america_for(zone)) is not None:
        ba, ba_source = implied[0], f"from {zone}"
    else:
        fits_state_ba = fits_state(BUILDING_AMERICA_COLUMN)
        ba, ba_source, reason = _derived(
            BUILDING_AMERICA_COLUMN, site, site_source, state_stats,
            lambda label: lookup.zones_compatible(zone, label) and fits_state_ba(label),
        )
        if ba is None and reason:
            warn(f"Building America climate zone unknown: {reason}")

    return LocationInfo(
        state=resolved_state,
        ashrae_zone=zone,
        building_america_zone=ba,
        census_division=division,
        puma_metro_status=explicit_metro,  # never derived: see docs/location.md
        state_source=state_source if resolved_state else None,
        ashrae_zone_source=zone_source if zone else None,
        building_america_zone_source=ba_source if ba else None,
        census_division_source=SOURCE_FROM_STATE if division else None,
        puma_metro_status_source=SOURCE_EXPLICIT if explicit_metro else None,
        latitude=lat,
        longitude=lon,
        nearest_station=nearest_name,
        station_distance_km=nearest_km,
        warnings=tuple(warnings),
    )


# ==================================================================================================
# Building tables (CSV rows, single buildings)
# ==================================================================================================


@dataclass
class LocationFill:
    """Result of :func:`fill_location_columns`."""

    data: pd.DataFrame
    mode: str | None = None  # "coordinates" (per-row latitude/longitude), "explicit" (options only), None (untouched)
    overrides: LocationOverrides = field(default_factory=LocationOverrides)
    coordinate_columns: tuple[str, str] | None = None
    filled: dict[str, int] = field(default_factory=dict)  # column -> cells filled
    conflicts: dict[str, int] = field(default_factory=dict)  # column -> input cells kept despite a different value
    summaries: dict[str, int] = field(default_factory=dict)  # location summary line -> rows (10 most common)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        """Lines to print: one summary when every row resolved the same way, else a per-row overview."""
        if self.mode is None:
            return []
        if len(self.summaries) == 1:
            return [next(iter(self.summaries))]
        if self.mode == "coordinates":
            head = f"Location: derived per row from {self.coordinate_columns[0]}/{self.coordinate_columns[1]}"
        else:
            head = "Location: explicit options applied to every row"
        filled = ", ".join(f"{column} {n}" for column, n in self.filled.items()) or "nothing"
        lines = [f"{head}; filled {filled}"]
        for line, count in list(self.summaries.items())[:3]:
            lines.append(f"  {count} row(s): {line.removeprefix('Location: ')}")
        return lines

    def to_json(self) -> dict | None:
        if self.mode is None and not self.warnings:
            return None
        return {
            "mode": self.mode,
            "explicit": self.overrides.to_json(),
            "coordinate_columns": list(self.coordinate_columns) if self.coordinate_columns else None,
            "filled": self.filled,
            "conflicts": self.conflicts,
            "summaries": self.summaries,
            "warnings": list(self.warnings),
        }


def fill_location_columns(
    df: pd.DataFrame,
    *,
    state: str | None = None,
    climate_zone: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    lookup: LocationLookup | None = None,
    max_row_warnings: int = 5,
) -> LocationFill:
    """Fill empty location cells of a building table; cells that already hold a value are never overwritten.

    * With ``latitude``/``longitude`` columns (case-insensitive) and no ``State`` values, the location is resolved per
      row from its coordinates. The coordinate columns are consumed (dropped from ``LocationFill.data``).
    * Explicit ``state``/``climate_zone``/``latitude``/``longitude`` apply to every row (a row's own coordinates win).
    * Without either, the table is returned unchanged.

    An input value that differs from the one the location gives is kept and reported in a warning.
    """
    lookup = lookup if lookup is not None else LocationLookup.load()
    overrides = LocationOverrides(state, climate_zone, latitude, longitude)
    resolve_location(**overrides.kwargs(), lookup=lookup)  # validate explicit values up front

    by_lower: dict[str, str] = {}
    for column in df.columns:
        by_lower.setdefault(str(column).strip().lower(), column)
    lat_col, lon_col = by_lower.get("latitude"), by_lower.get("longitude")
    coordinate_columns = (lat_col, lon_col) if lat_col is not None and lon_col is not None else None
    has_state = STATE_COLUMN in df.columns and bool(df[STATE_COLUMN].map(_present).any())
    warnings: list[str] = []
    if (lat_col is None) != (lon_col is None):
        warnings.append(f"column {lat_col or lon_col!r} ignored: locating rows needs both latitude and longitude columns")
    elif coordinate_columns and has_state:
        warnings.append(f"columns {lat_col!r}/{lon_col!r} not used: rows are located from coordinates only when the "
                        "input has no State values")
    consumed = [c for c in (lat_col, lon_col) if c is not None]
    data = df.drop(columns=consumed) if consumed else df
    per_row = coordinate_columns is not None and not has_state
    result = LocationFill(data, overrides=overrides, coordinate_columns=coordinate_columns, warnings=warnings)
    if not per_row and not overrides.given:
        return result
    result.mode = "coordinates" if per_row else "explicit"

    n = len(data)
    present: dict[str, list[bool]] = {}
    given: dict[str, list[str | None]] = {}
    for column in LOCATION_COLUMNS:
        if column in data.columns:
            raw = data[column].tolist()
            labels = {v: canonical_location_label(column, v, lookup) for v in set(raw) if _present(v)}
            present[column] = [_present(v) for v in raw]
            given[column] = [labels.get(v) if p else None for v, p in zip(raw, present[column])]
        else:
            present[column], given[column] = [False] * n, [None] * n
    lat_raw = df[lat_col].tolist() if per_row else None
    lon_raw = df[lon_col].tolist() if per_row else None

    new: dict[str, list[str | None]] = {column: [None] * n for column in LOCATION_COLUMNS}
    filled: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    examples: dict[str, tuple[int, str, str]] = {}
    summaries: Counter[str] = Counter()
    row_warnings: dict[str, list[int]] = {}
    invalid_rows: list[int] = []
    cache: dict[tuple, tuple[dict[str, str], dict[str, str], str, tuple[str, ...]]] = {}
    for i in range(n):
        lat, lon = overrides.latitude, overrides.longitude
        if per_row and (_present(lat_raw[i]) or _present(lon_raw[i])):
            try:
                lat, lon = _coordinates(lat_raw[i], lon_raw[i])
            except InputValidationError:
                invalid_rows.append(i)
                lat, lon = overrides.latitude, overrides.longitude
        cells = tuple(given[column][i] for column in LOCATION_COLUMNS)
        key = (lat, lon, cells)
        if key not in cache:
            # derived: what the options and coordinates alone give (compared with input cells; source of warnings)
            # final: consistent with the row's own values (e.g. Building America zone from its climate zone)
            derived = resolve_location(**{**overrides.kwargs(), "latitude": lat, "longitude": lon}, lookup=lookup)
            final = derived
            if any(cell is not None for cell in cells):
                final = resolve_location(
                    state=cells[0] or overrides.state, climate_zone=cells[1] or overrides.climate_zone,
                    building_america_zone=cells[2], puma_metro_status=cells[4], latitude=lat, longitude=lon,
                    lookup=lookup,
                )
                final = replace(final, **{f"{attr}_source": SOURCE_INPUT for (_, attr), cell
                                          in zip(LocationInfo.FIELDS, cells) if cell is not None and attr != "census_division"})
            cache[key] = (derived.values(), final.values(), final.summary(), derived.warnings)
        derived_values, final_values, summary, messages = cache[key]
        for j, column in enumerate(LOCATION_COLUMNS):
            if present[column][i]:
                other = derived_values.get(column)
                if cells[j] is not None and other is not None and other != cells[j]:
                    conflicts[column] += 1
                    examples.setdefault(column, (i, cells[j], other))
            elif column in final_values:
                new[column][i] = final_values[column]
                filled[column] += 1
        summaries[summary] += 1
        for message in messages:
            row_warnings.setdefault(message, []).append(i)

    if filled:
        data = data.copy()
        for column in LOCATION_COLUMNS:
            if not filled[column]:
                continue
            old = data[column].tolist() if column in data.columns else [None] * n
            data[column] = pd.Series([v if v is not None else o for v, o in zip(new[column], old)],
                                     index=data.index, dtype=object)
    if invalid_rows:
        warnings.append(f"{len(invalid_rows)} row(s) have invalid latitude/longitude values, treated as missing "
                        f"(e.g. row {invalid_rows[0]})")
    for column, count in conflicts.items():
        row, cell, other = examples[column]
        warnings.append(f"{column}: {count} row(s) keep their input value although the location gives another one "
                        f"(e.g. row {row}: {cell!r} vs {other!r})")
    for message, rows in list(row_warnings.items())[:max_row_warnings]:
        more = f" and {len(rows) - 1} more row(s)" if len(rows) > 1 else ""
        warnings.append(f"row {rows[0]}{more}: {message}")
    if len(row_warnings) > max_row_warnings:
        warnings.append(f"{len(row_warnings) - max_row_warnings} more distinct location warning(s) not shown")

    result.data = data
    result.filled = dict(filled)
    result.conflicts = dict(conflicts)
    result.summaries = dict(summaries.most_common(10))
    return result
