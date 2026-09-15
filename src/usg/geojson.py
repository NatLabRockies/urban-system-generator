"""URBANopt GeoJSON -> ResStock parameter columns, plus the project's location (see ``usg.location``).

Mapper outputs are always valid option names in ``options_lookup.tsv`` for their parameter, or ``None`` when the
GeoJSON value is missing or unrecognized (the model then infers the column). Vintage uses the ResStock 2024 labels
(``<1940``, ``1940s`` ... ``2010s``); ``options_lookup.tsv`` also lists ``<1950`` but the 2024 data never uses it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import pandas as pd

from usg.constants import ID_COLUMN
from usg.errors import InputValidationError
from usg.location import (  # noqa: F401 - CENSUS_DIVISION_COLUMN, state_from_weather_filename re-exported
    CENSUS_DIVISION_COLUMN,
    SOURCE_FROM_STATE,
    LocationInfo,
    LocationOverrides,
    resolve_location,
    state_from_weather_filename,
)


@lru_cache(maxsize=1)
def _allowed_options() -> dict[str, list[str]]:
    from usg.postprocessor import load_allowed_options
    from usg.resources import options_lookup_path

    return load_allowed_options(options_lookup_path())


def _valid_option(parameter: str, value: str | None) -> str | None:
    if value is None:
        return None
    return value if value in _allowed_options().get(parameter, ()) else None


def _to_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ==================================================================================================
# Project context
# ==================================================================================================


@dataclass(frozen=True)
class ProjectContext:
    """Project location: the location conditioning columns, where each came from, coordinates and warnings.

    Built by :func:`project_context_from_geojson` with :func:`usg.location.resolve_location`; every building
    of the project shares it. Sources are ``"<label>"`` or ``"<label> (<detail>)"``, e.g.
    ``"weather file (USA_MD_….epw)"``, ``"nearest station (Baltimore Blt Washngtn IntL, 16 km)"``,
    ``"from state (KY 100%)"``, ``"from 4A"``.
    """

    state: str | None = None
    state_source: str | None = None
    climate_zone: str | None = None  # ASHRAE IECC Climate Zone 2004
    weather_filename: str | None = None
    climate_zone_source: str | None = None
    building_america_zone: str | None = None
    building_america_zone_source: str | None = None
    census_division: str | None = None  # None: follows the state
    census_division_source: str | None = None
    puma_metro_status: str | None = None
    puma_metro_status_source: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    coordinates_source: str | None = None  # "Site Origin" or "explicit"
    nearest_station: str | None = None
    nearest_station_km: float | None = None
    county_and_puma: str | None = None  # ResStock "County and PUMA" of the Census tract nearest to the coordinates
    county_and_puma_source: str | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_location(
        cls,
        info: LocationInfo,
        *,
        weather_filename: str | None = None,
        coordinates_source: str | None = None,
        warnings: Iterable[str] = (),
    ) -> ProjectContext:
        return cls(
            state=info.state,
            state_source=info.state_source,
            climate_zone=info.ashrae_zone,
            weather_filename=weather_filename,
            climate_zone_source=info.ashrae_zone_source,
            building_america_zone=info.building_america_zone,
            building_america_zone_source=info.building_america_zone_source,
            census_division=info.census_division,
            census_division_source=info.census_division_source,
            puma_metro_status=info.puma_metro_status,
            puma_metro_status_source=info.puma_metro_status_source,
            latitude=info.latitude,
            longitude=info.longitude,
            coordinates_source=coordinates_source if info.latitude is not None else None,
            nearest_station=info.nearest_station,
            nearest_station_km=info.station_distance_km,
            warnings=tuple(dict.fromkeys([*warnings, *info.warnings])),
        )

    def location(self) -> LocationInfo:
        """The resolved location; Census Division follows the state when it was not set."""
        division, division_source = self.census_division, self.census_division_source
        if division is None and self.state:
            from usg.postprocessor import STATE_TO_CENSUS_DIVISION

            division = STATE_TO_CENSUS_DIVISION.get(self.state)
            division_source = SOURCE_FROM_STATE if division else None
        return LocationInfo(
            state=self.state,
            ashrae_zone=self.climate_zone,
            building_america_zone=self.building_america_zone,
            census_division=division,
            puma_metro_status=self.puma_metro_status,
            state_source=self.state_source,
            ashrae_zone_source=self.climate_zone_source,
            building_america_zone_source=self.building_america_zone_source,
            census_division_source=division_source,
            puma_metro_status_source=self.puma_metro_status_source,
            latitude=self.latitude,
            longitude=self.longitude,
            nearest_station=self.nearest_station,
            station_distance_km=self.nearest_station_km,
            warnings=self.warnings,
        )

    def conditioning_values(self) -> dict[str, str]:
        """Known location columns among State, ASHRAE IECC Climate Zone 2004, Building America Climate Zone,
        Census Division and PUMA Metro Status."""
        return self.location().values()

    def summary(self) -> str:
        """One line: ``Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · …``."""
        line = self.location().summary()
        if self.county_and_puma:
            line += f" · County and PUMA {self.county_and_puma} ({self.county_and_puma_source})"
        return line

    def location_json(self) -> dict:
        """The location record written to ``run_manifest.json``."""
        return {**self.location().to_json(), "coordinates_source": self.coordinates_source,
                "weather_filename": self.weather_filename, "county_and_puma": self.county_and_puma,
                "county_and_puma_source": self.county_and_puma_source}


def _is_site_origin(feature: dict) -> bool:
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    return str(props.get("type") or props.get("name") or "").strip().lower() == "site origin"


def site_origin_point(data: dict, warnings: list[str] | None = None) -> tuple[float, float] | None:
    """``(latitude, longitude)`` of the Site Origin feature's Point geometry, or None.

    This is the only place coordinates are read from a GeoJSON file (no project fields, no building footprints), so
    every building of the project shares one location. GeoJSON positions are ``[longitude, latitude]``; a position
    that is not in degrees is ignored with a message appended to ``warnings``.
    """
    features = data.get("features") if isinstance(data, dict) else None
    for feature in features or []:
        if not isinstance(feature, dict) or not _is_site_origin(feature):
            continue
        geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
        position = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        lon, lat = _to_number(position[0]), _to_number(position[1])
        if lat is None or lon is None:
            return None
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
        if warnings is not None:
            warnings.append(f"Site Origin coordinates ({position[0]}, {position[1]}) are not longitude/latitude "
                            "degrees; ignored")
        return None
    return None


def project_context_from_geojson(data: dict, overrides: LocationOverrides | None = None) -> ProjectContext:
    """The location shared by every building of an URBANopt GeoJSON project.

    Only two things in the file locate the project: the weather filename ``USA_<ST>_…`` (``project.weather_filename``,
    else the Site Origin's ``weather_filename``) gives the state, and the Site Origin Point gives the coordinates, whose
    nearest ResStock weather station gives the climate zones (and the state when there is no weather filename).
    ``project.climate_zone``, other project fields and building footprints are not used. ``overrides`` (explicit user
    values) win. See :func:`usg.location.resolve_location`.
    """
    overrides = overrides or LocationOverrides()
    data = data if isinstance(data, dict) else {}
    project = data.get("project") if isinstance(data.get("project"), dict) else {}

    weather, site_origin_weather = project.get("weather_filename") or None, False
    if not weather:
        for feature in data.get("features") or []:
            props = feature.get("properties") if isinstance(feature, dict) else None
            if isinstance(props, dict) and _is_site_origin(feature) and props.get("weather_filename"):
                weather, site_origin_weather = props["weather_filename"], True
                break

    warnings: list[str] = []
    if overrides.latitude is not None or overrides.longitude is not None:
        point, coordinates_source = (overrides.latitude, overrides.longitude), "explicit"
    else:
        point, coordinates_source = site_origin_point(data, warnings), "Site Origin"
    latitude, longitude = point if point else (None, None)
    info = resolve_location(
        state=overrides.state, climate_zone=overrides.climate_zone, latitude=latitude, longitude=longitude,
        weather_filename=weather,
    )
    if site_origin_weather and info.state_source and info.state_source.startswith("weather file"):
        info = replace(info, state_source=f"Site Origin {info.state_source}")
    context = ProjectContext.from_location(
        info, weather_filename=weather, coordinates_source=coordinates_source, warnings=warnings
    )
    return with_county_and_puma(context)


def with_county_and_puma(context: ProjectContext) -> ProjectContext:
    """Add the ResStock ``County and PUMA`` of the Census tract nearest to the project's coordinates.

    Only tracts of the project's state are considered, so the result can never contradict the weather file. Without
    coordinates (or with no tract of that state within 25 km), post-processing keeps choosing it from the weather
    station instead.
    """
    if context.latitude is None or context.longitude is None or not context.state:
        return context
    from usg.county_puma import MAX_DISTANCE_KM, CountyPumaLookup

    lookup = CountyPumaLookup.load()
    where = context.coordinates_source or "site"
    match = lookup.nearest(context.latitude, context.longitude, state=context.state)
    warnings = list(context.warnings)
    if context.state_source == "explicit":  # the user chose the state: no location warnings (as for the station)
        return context if match is None else replace(
            context, county_and_puma=match.county_and_puma,
            county_and_puma_source=f"{where}, nearest Census tract {match.distance_km:.1f} km",
        )
    if match is None:
        warnings.append(f"no {context.state} Census tract within {MAX_DISTANCE_KM:.0f} km of the {where} coordinates; "
                        "County and PUMA follow the nearest weather station instead")
        return replace(context, warnings=tuple(dict.fromkeys(warnings)))
    overall = lookup.nearest(context.latitude, context.longitude)
    if overall is not None and overall.state != context.state and overall.distance_km + 1.0 < match.distance_km:
        warnings.append(f"the {where} lies in {overall.state} but the project state is {context.state}; "
                        f"County and PUMA use the nearest {context.state} Census tract ({match.distance_km:.1f} km)")
    return replace(
        context, county_and_puma=match.county_and_puma, warnings=tuple(dict.fromkeys(warnings)),
        county_and_puma_source=f"{where}, nearest Census tract {match.distance_km:.1f} km",
    )


def _load_geojson(path: str | Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise InputValidationError(f"GeoJSON file not found: {path}") from None
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot read GeoJSON {path}: {exc}") from None


def read_project_context(path: str | Path, overrides: LocationOverrides | None = None) -> ProjectContext:
    return project_context_from_geojson(_load_geojson(path), overrides)


def read_project_state(geojson_path: str | Path) -> str | None:
    """State abbreviation from the project (or Site Origin) weather filename ``USA_XX_...``; None if absent."""
    return read_project_context(geojson_path).state if Path(geojson_path).exists() else None


def read_project_climate_zone(geojson_path: str | Path) -> str | None:
    """Resolved ASHRAE climate zone (Site Origin's nearest weather station, else the state's); None if unknown."""
    return read_project_context(geojson_path).climate_zone if Path(geojson_path).exists() else None


# ==================================================================================================
# Building properties
# ==================================================================================================


class GeoJSONProcessor:
    """Reads URBANopt GeoJSON buildings and maps their properties to ResStock parameter columns."""

    PROPERTY_MAPPER: ClassVar[dict[str, str]] = {
        "building_type": "Geometry Building Type RECS",
        "year_built": "Vintage",
        "floor_area": "Geometry Floor Area",
        "number_of_stories": "Geometry Stories",
        "number_of_bedrooms": "Bedrooms",
        "number_of_residential_units": "Geometry Building Number Units SFA",
        "foundation_type": "Geometry Foundation Type",
        "attic_type": "Geometry Attic Type",
        "system_type": "HVAC Cooling Type",
        "heating_system_fuel_type": "Heating Fuel",
    }

    RESIDENTIAL_TYPES = ("Multifamily", "Single-Family Attached", "Single-Family Detached", "Single-Family", "Mobile Home")

    # ---- mapping functions ------------------------------------------------------------------------
    @staticmethod
    def map_vintage(year_built: int) -> str:
        """Map year_built to a ResStock 2024 Vintage option."""
        for bound, label in ((1940, "<1940"), (1950, "1940s"), (1960, "1950s"), (1970, "1960s"), (1980, "1970s"),
                             (1990, "1980s"), (2000, "1990s"), (2010, "2000s")):
            if year_built < bound:
                return label
        return "2010s"

    @staticmethod
    def map_vintage_acs(year_built: int) -> str:
        """Map year_built to a Vintage ACS option."""
        for bound, label in ((1940, "<1940"), (1960, "1940-59"), (1980, "1960-79"), (2000, "1980-99"), (2010, "2000-09")):
            if year_built < bound:
                return label
        return "2010s"

    @staticmethod
    def map_floor_area_to_category(floor_area: float) -> str:
        """Map conditioned floor area (ft²) to a Geometry Floor Area option."""
        for bound, label in ((500, "0-499"), (750, "500-749"), (1000, "750-999"), (1500, "1000-1499"),
                             (2000, "1500-1999"), (2500, "2000-2499"), (3000, "2500-2999"), (4000, "3000-3999")):
            if floor_area < bound:
                return label
        return "4000+"

    @staticmethod
    def map_floor_area_to_bin(floor_area: float) -> str:
        """Map conditioned floor area (ft²) to a Geometry Floor Area Bin option."""
        for bound, label in ((1500, "0-1499"), (2500, "1500-2499"), (4000, "2500-3999")):
            if floor_area < bound:
                return label
        return "4000+"

    @staticmethod
    def _units(properties: dict) -> int:
        units = _to_number(properties.get("number_of_residential_units"))
        return int(units) if units is not None else 0

    @staticmethod
    def map_building_type(building_type: str, properties: dict) -> str | None:
        """Map an URBANopt building_type to a Geometry Building Type RECS option (None if not residential)."""
        if building_type == "Multifamily":
            units = GeoJSONProcessor._units(properties)
            return "Multi-Family with 2 - 4 Units" if 2 <= units <= 4 else "Multi-Family with 5+ Units"
        if building_type in ("Single-Family Attached", "Single-Family"):
            return "Single-Family Attached"
        if building_type == "Single-Family Detached":
            return "Single-Family Detached"
        if building_type == "Mobile Home":
            return "Mobile Home"
        return None

    @staticmethod
    def map_units_sfa(units, properties: dict) -> str | None:
        """Geometry Building Number Units SFA: the unit count for attached homes, ``"None"`` for other types."""
        btype = GeoJSONProcessor.map_building_type(properties.get("building_type"), properties)
        if btype is None:
            return None
        if btype != "Single-Family Attached":
            return "None"
        number = _to_number(units)
        return _valid_option("Geometry Building Number Units SFA", str(int(number))) if number is not None else None

    @staticmethod
    def map_foundation_type(foundation_type: str) -> str | None:
        mapping = {
            "slab": "Slab",
            "crawlspace - vented": "Vented Crawlspace",
            "crawlspace - unvented": "Unvented Crawlspace",
            "crawlspace - conditioned": "Conditioned Crawlspace",
            "basement - unconditioned": "Unheated Basement",
            "basement - conditioned": "Heated Basement",
            "ambient": "Ambient",
        }
        return mapping.get(foundation_type)

    @staticmethod
    def map_attic_type(attic_type: str) -> str | None:
        mapping = {
            "attic - vented": "Vented Attic",
            "attic - unvented": "Unvented Attic",
            "attic - conditioned": "Finished Attic or Cathedral Ceilings",
            "flat roof": "None",
        }
        return mapping.get(attic_type)

    @staticmethod
    def map_heating_fuel(heating_fuel: str) -> str | None:
        mapping = {
            "electricity": "Electricity",
            "natural gas": "Natural Gas",
            "fuel oil": "Fuel Oil",
            "propane": "Propane",
            "wood": "Wood",
            "other": "Other Fuel",
            "none": "None",
        }
        return mapping.get(heating_fuel)

    @staticmethod
    def map_hvac_cooling_type(system_type: str) -> str | None:
        """Map an URBANopt residential system_type to an HVAC Cooling Type option (None if unrecognized)."""
        if not isinstance(system_type, str) or not system_type.startswith("Residential - "):
            return None
        text = system_type[len("Residential - "):]
        if text in ("air-to-air heat pump", "ground-to-air heat pump"):
            return "Ducted Heat Pump"
        if text == "mini-split heat pump":
            return "Non-Ducted Heat Pump"
        if text.endswith("central air conditioner"):
            return "Central AC"
        if text.endswith("room air conditioner"):
            return "Room AC"
        if text.endswith("evaporative cooler"):
            return "Evaporative or Swamp Cooler"
        if text.endswith("no cooling"):
            return "None"
        return None

    @staticmethod
    def map_stories(number_of_stories) -> str | None:
        number = _to_number(number_of_stories)
        return _valid_option("Geometry Stories", str(int(number))) if number is not None else None

    @staticmethod
    def map_bedrooms(number_of_bedrooms) -> str | None:
        """Bedrooms options are 1..5 (5 = five or more); studios count as 1."""
        number = _to_number(number_of_bedrooms)
        return None if number is None or number < 0 else str(min(max(int(number), 1), 5))

    # ---- processing -------------------------------------------------------------------------------
    def read_geojson(self, geojson_path: str | Path) -> dict:
        return _load_geojson(geojson_path)

    def extract_residential_buildings(self, geojson: dict) -> list[dict]:
        buildings = []
        for feature in geojson.get("features", []):
            props = (feature or {}).get("properties", {}) or {}
            if props.get("type") == "Building" and props.get("building_type", "") in self.RESIDENTIAL_TYPES:
                buildings.append(feature)
        return buildings

    def map_properties(self, feature: dict) -> dict:
        """Map one GeoJSON feature to ``{Building, <parameter>: <option>}`` (unmappable values are omitted)."""
        props = feature.get("properties", {}) or {}
        building_id = props.get("id", "")
        if isinstance(building_id, str) and building_id.startswith("way/"):
            building_id = building_id.replace("way/", "")
        mapped: dict[str, str | None] = {ID_COLUMN: building_id}

        year = _to_number(props.get("year_built"))
        if year is not None:
            mapped["Vintage"] = self.map_vintage(int(year))
            mapped["Vintage ACS"] = self.map_vintage_acs(int(year))
        area = _to_number(props.get("floor_area"))
        if area is not None and area > 0:
            mapped["Geometry Floor Area"] = self.map_floor_area_to_category(area)
            mapped["Geometry Floor Area Bin"] = self.map_floor_area_to_bin(area)
        values = {
            "Geometry Building Type RECS": self.map_building_type(props.get("building_type"), props),
            "Geometry Foundation Type": self.map_foundation_type(props.get("foundation_type")),
            "Geometry Attic Type": self.map_attic_type(props.get("attic_type")),
            "Heating Fuel": self.map_heating_fuel(props.get("heating_system_fuel_type")),
            "HVAC Cooling Type": self.map_hvac_cooling_type(props.get("system_type")),
            "Geometry Stories": self.map_stories(props.get("number_of_stories")),
            "Bedrooms": self.map_bedrooms(props.get("number_of_bedrooms")),
            "Geometry Building Number Units SFA": (
                self.map_units_sfa(props.get("number_of_residential_units"), props)
                if props.get("number_of_residential_units") is not None
                else None
            ),
        }
        mapped.update({k: v for k, v in values.items() if v is not None})
        return mapped

    def geojson_to_dataframe(self, geojson_path: str | Path, all_model_attributes: list[str] | None = None) -> pd.DataFrame:
        buildings = self.extract_residential_buildings(self.read_geojson(geojson_path))
        df = pd.DataFrame([self.map_properties(b) for b in buildings], dtype=object)
        if ID_COLUMN not in df.columns:
            df[ID_COLUMN] = pd.Series(dtype=object)
        if all_model_attributes:
            for attr in all_model_attributes:
                if attr not in df.columns:
                    df[attr] = None
            df = df[[ID_COLUMN] + [c for c in all_model_attributes if c != ID_COLUMN]]
        return df

    def geojson_to_csv(
        self, geojson_path: str | Path, output_csv_path: str | Path, all_model_attributes: list[str] | None = None
    ) -> str:
        df = self.geojson_to_dataframe(geojson_path, all_model_attributes)
        if df.empty:
            raise InputValidationError(f"no residential buildings found in GeoJSON {geojson_path}")
        df.to_csv(output_csv_path, index=False)
        return f"Successfully converted {len(df)} buildings to CSV: {output_csv_path}"


def geojson_to_dataframe(geojson_path: str | Path, schema=None, context: ProjectContext | None = None) -> pd.DataFrame:
    """Residential buildings as ``Building`` + mapped parameter columns (+ project conditioning columns).

    With ``schema``: all schema columns are present (unknown cells ``None``), in schema order, followed by any mapped
    columns the schema lacks; project location columns are only filled when the schema has them and knows the label
    (e.g. ``State`` ``AK`` stays unknown to a model without Alaska data).
    """
    processor = GeoJSONProcessor()
    data = processor.read_geojson(geojson_path)
    buildings = processor.extract_residential_buildings(data)
    if not buildings:
        raise InputValidationError(f"no residential buildings found in GeoJSON {geojson_path}")
    df = pd.DataFrame([processor.map_properties(b) for b in buildings], dtype=object)
    ctx = context if context is not None else project_context_from_geojson(data)
    for column, value in ctx.conditioning_values().items():
        if schema is None or (column in schema and (schema.column(column).type != "categorical"
                                                    or value in schema.column(column).vocab)):
            df[column] = value
    if schema is None:
        return df
    missing = [name for name in schema.names if name not in df.columns]
    if missing:  # add all unknown columns at once (column-by-column inserts fragment the frame)
        df = pd.concat([df, pd.DataFrame(None, index=df.index, columns=missing, dtype=object)], axis=1)
    extra = [c for c in df.columns if c != ID_COLUMN and c not in schema]
    return df[[ID_COLUMN, *schema.names, *extra]]
