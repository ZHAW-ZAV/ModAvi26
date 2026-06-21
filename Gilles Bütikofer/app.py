    # -*- coding: utf-8 -*-
"""
Streamlit Satellite Coverage Viewer
===================================

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit skyfield plotly pandas numpy pillow

What this page does
-------------------
- Loads public TLE feeds from CelesTrak or an uploaded .txt/.tle file.
- Propagates TLEs with Skyfield / SGP4.
- Converts satellite states to WGS84 latitude/longitude/height and ECEF XYZ.
- Computes a simple spherical-Earth visibility footprint for a chosen minimum elevation.
- Plots Earth, satellites, sub-satellite points, footprint rings, and optional ISS orbit track in a Plotly 3D figure.
- Shows quick intermediate tables for altitude, lat/lon, TLE age, and footprint radius.

Units
-----
- distance: kilometers [km]
- angle: degrees [deg]
- time: UTC
- 3D frame: Earth-fixed ECEF XYZ [km]

Model notes
-----------
- WGS84 ellipsoid is used for position conversion and Earth plotting.
- Footprints are a geometric first pass using a spherical Earth with mean radius.
- No atmosphere, terrain, antenna pattern, occultation by terrain, or link budget is modeled yet.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO
import math
import re
import urllib.request
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover - Streamlit usually installs Pillow
    Image = None

try:
    from skyfield.api import EarthSatellite, load, wgs84
except Exception as exc:  # pragma: no cover
    st.error(
        "Missing dependency. Install in the environment running Streamlit:\n\n"
        "pip install streamlit skyfield plotly pandas numpy pillow"
    )
    st.stop()


# =============================================================================
# 0) Constants and built-in public feeds
# =============================================================================

@dataclass(frozen=True)
class EarthModel:
    name: str = "WGS84"
    semi_major_axis_km: float = 6378.137000
    semi_minor_axis_km: float = 6356.752314245
    mean_radius_km: float = 6371.0088
    mu_km3_s2: float = 398600.4418

    @property
    def first_eccentricity_squared(self) -> float:
        a = self.semi_major_axis_km
        b = self.semi_minor_axis_km
        return 1.0 - (b * b) / (a * a)


EARTH = EarthModel()

CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_GROUPS = {
    "GPS": "gps-ops",
    "Galileo": "galileo",
    "GLONASS": "glo-ops",
    "BeiDou": "beidou",
    "ISS": "stations",
}

DISPLAY_COUNT_OPTIONS = ["All", "1", "2", "3", "5", "10", "20", "50"]
DEFAULT_TEXTURE_URL = "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg"

KNOWN_ISS_INCLINATION_DEG = 51.6
KNOWN_ISS_ALTITUDE_RANGE_KM = (370.0, 460.0)


# =============================================================================
# 1) Data structures
# =============================================================================

@dataclass(frozen=True)
class TLERecord:
    name: str
    line1: str
    line2: str

    @property
    def satnum(self) -> str:
        return self.line1[2:7].strip()


@dataclass(frozen=True)
class SatelliteState:
    name: str
    utc: str
    tle_epoch_utc: str
    tle_age_days: float
    latitude_deg: float
    longitude_deg: float
    height_km: float
    ecef_x_km: float
    ecef_y_km: float
    ecef_z_km: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FootprintSummary:
    name: str
    min_elevation_deg: float
    central_angle_deg: float
    ground_radius_km: float
    spherical_area_km2: float
    subpoint_latitude_deg: float
    subpoint_longitude_deg: float
    satellite_height_km: float

    def as_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# 2) TLE loading and parsing
# =============================================================================

@st.cache_data(ttl=7200, show_spinner=False)
def download_celestrak_group(group: str) -> str:
    """Download CelesTrak GP data in 3-line TLE format.

    Cache TTL is 2 hours to avoid repeatedly hitting CelesTrak during interactive use.
    """
    url = f"{CELESTRAK_BASE}?GROUP={group}&FORMAT=tle"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "satellite-coverage-streamlit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_tle_text(text: str) -> list[TLERecord]:
    """Parse 3-line or 2-line TLE text.

    Supports:
    - name + line 1 + line 2
    - line 1 + line 2 only
    - mixed files with blank lines
    """
    raw_lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in raw_lines if line.strip()]
    records: list[TLERecord] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            line1 = line[:69]
            line2 = lines[i + 1][:69]
            name = f"SAT {line1[2:7].strip()}"
            records.append(TLERecord(name=name, line1=line1, line2=line2))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = line.strip()
            line1 = lines[i + 1][:69]
            line2 = lines[i + 2][:69]
            records.append(TLERecord(name=name, line1=line1, line2=line2))
            i += 3
        else:
            i += 1
    return records


def find_iss_record(records: list[TLERecord]) -> TLERecord | None:
    for r in records:
        if "ISS" in r.name.upper() or r.satnum == "25544":
            return r
    return None


def build_satellites(records: list[TLERecord], ts) -> list[EarthSatellite]:
    satellites = []
    for r in records:
        try:
            satellites.append(EarthSatellite(r.line1, r.line2, r.name, ts))
        except Exception:
            # Keep app robust for messy uploaded files.
            continue
    return satellites


def tle_checksum_ok(line: str) -> bool | None:
    if len(line) < 69 or not line[68].isdigit():
        return None
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return (total % 10) == int(line[68])


def quick_orbit_checks(record: TLERecord) -> dict:
    """Extract simple orbital checks from TLE line 2.

    These are approximate and for sanity checks only.
    """
    line2 = record.line2
    warnings: list[str] = []
    try:
        inc = float(line2[8:16])
        ecc = float("0." + line2[26:33].strip())
        mean_motion_rev_day = float(line2[52:63])
        period_min = 1440.0 / mean_motion_rev_day
        n_rad_s = mean_motion_rev_day * 2.0 * math.pi / 86400.0
        semi_major_axis_km = (EARTH.mu_km3_s2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
        perigee_alt_km = semi_major_axis_km * (1.0 - ecc) - EARTH.mean_radius_km
        apogee_alt_km = semi_major_axis_km * (1.0 + ecc) - EARTH.mean_radius_km
    except Exception:
        inc = np.nan
        ecc = np.nan
        mean_motion_rev_day = np.nan
        period_min = np.nan
        semi_major_axis_km = np.nan
        perigee_alt_km = np.nan
        apogee_alt_km = np.nan
        warnings.append("could_not_parse_line2")

    if tle_checksum_ok(record.line1) is False or tle_checksum_ok(record.line2) is False:
        warnings.append("checksum")
    if np.isfinite(ecc) and ecc > 0.10:
        warnings.append("high_eccentricity")
    if np.isfinite(perigee_alt_km) and perigee_alt_km < 120.0:
        warnings.append("low_perigee")

    return {
        "name": record.name,
        "satnum": record.satnum,
        "checksum_ok": tle_checksum_ok(record.line1) is not False and tle_checksum_ok(record.line2) is not False,
        "inclination_deg": inc,
        "eccentricity": ecc,
        "mean_motion_rev_day": mean_motion_rev_day,
        "period_min": period_min,
        "semi_major_axis_km": semi_major_axis_km,
        "perigee_alt_km": perigee_alt_km,
        "apogee_alt_km": apogee_alt_km,
        "warnings": "; ".join(warnings),
    }


# =============================================================================
# 3) Coordinate system, propagation, and footprints
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_timescale():
    return load.timescale()


def wrap_lon180(lon_deg):
    return (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0


def geodetic_to_ecef(lat_deg, lon_deg, height_km=0.0, earth: EarthModel = EARTH):
    """WGS84 geodetic latitude/longitude/height -> ECEF XYZ in km."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h = np.asarray(height_km, dtype=float)
    a = earth.semi_major_axis_km
    e2 = earth.first_eccentricity_squared
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * np.cos(lon)
    y = (N + h) * cos_lat * np.sin(lon)
    z = ((1.0 - e2) * N + h) * sin_lat
    return x, y, z


def format_time_utc(t) -> str:
    return t.utc_strftime("%Y-%m-%d %H:%M:%S UTC")


def latest_tle_epoch(satellites: list[EarthSatellite]):
    return max((sat.epoch for sat in satellites), key=lambda epoch: epoch.tt)


def now_utc_time(ts):
    dt = datetime.now(timezone.utc)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def parse_manual_utc(ts, date_value, time_value):
    dt = datetime.combine(date_value, time_value).replace(tzinfo=timezone.utc)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def satellite_state(satellite: EarthSatellite, t) -> SatelliteState | None:
    try:
        geocentric = satellite.at(t)
        geo = wgs84.geographic_position_of(geocentric)
        lat_deg = float(geo.latitude.degrees)
        lon_deg = float(wrap_lon180(geo.longitude.degrees))
        height_km = float(geo.elevation.km)
        x_ecef, y_ecef, z_ecef = geodetic_to_ecef(lat_deg, lon_deg, height_km)
        return SatelliteState(
            name=satellite.name,
            utc=format_time_utc(t),
            tle_epoch_utc=format_time_utc(satellite.epoch),
            tle_age_days=float(t.tt - satellite.epoch.tt),
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            height_km=height_km,
            ecef_x_km=float(x_ecef),
            ecef_y_km=float(y_ecef),
            ecef_z_km=float(z_ecef),
        )
    except Exception:
        return None


def satellite_states(satellites: list[EarthSatellite], t) -> list[SatelliteState]:
    states = []
    for sat in satellites:
        state = satellite_state(sat, t)
        if state is not None:
            states.append(state)
    return states


def footprint_central_angle_deg(height_km: float, min_elevation_deg: float) -> float:
    """Spherical Earth geometric coverage half-angle.

    psi is the central angle at Earth's center between sub-satellite point and
    the footprint boundary. 0 deg elevation gives the horizon cap.
    """
    R = EARTH.mean_radius_km
    h = max(float(height_km), 0.0)
    e = math.radians(float(min_elevation_deg))
    rho = R / (R + h)
    psi = math.acos(max(-1.0, min(1.0, rho * math.cos(e)))) - e
    return max(0.0, math.degrees(psi))


def great_circle_destination(lat1_deg: float, lon1_deg: float, bearing_deg, central_angle_deg: float):
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    bearing = np.radians(bearing_deg)
    delta = np.radians(central_angle_deg)
    sin_lat2 = np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(bearing)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), wrap_lon180(np.degrees(lon2))


def footprint_boundary_latlon(state: SatelliteState, min_elevation_deg: float, points: int = 241):
    central_angle_deg = footprint_central_angle_deg(state.height_km, min_elevation_deg)
    bearings = np.linspace(0.0, 360.0, points)
    return great_circle_destination(state.latitude_deg, state.longitude_deg, bearings, central_angle_deg)


def footprint_summary(state: SatelliteState, min_elevation_deg: float) -> FootprintSummary:
    psi_deg = footprint_central_angle_deg(state.height_km, min_elevation_deg)
    psi_rad = math.radians(psi_deg)
    return FootprintSummary(
        name=state.name,
        min_elevation_deg=float(min_elevation_deg),
        central_angle_deg=psi_deg,
        ground_radius_km=EARTH.mean_radius_km * psi_rad,
        spherical_area_km2=2.0 * math.pi * EARTH.mean_radius_km**2 * (1.0 - math.cos(psi_rad)),
        subpoint_latitude_deg=state.latitude_deg,
        subpoint_longitude_deg=state.longitude_deg,
        satellite_height_km=state.height_km,
    )


def orbit_track_latlon(satellite: EarthSatellite, start_t, minutes: float, samples: int = 241):
    ts = get_timescale()
    offsets_min = np.linspace(0.0, minutes, samples)
    t = ts.tt_jd(start_t.tt + offsets_min / 1440.0)
    geocentric = satellite.at(t)
    geo = wgs84.geographic_position_of(geocentric)
    return np.array(geo.latitude.degrees), wrap_lon180(np.array(geo.longitude.degrees)), np.array(geo.elevation.km)


# =============================================================================
# 4) Plotly 3D plotting
# =============================================================================

def texture_to_vertex_colors(texture_bytes: bytes | None, lat_grid, lon_grid) -> list[str] | None:
    if texture_bytes is None or Image is None:
        return None
    try:
        image = Image.open(BytesIO(texture_bytes)).convert("RGB")
        arr = np.asarray(image)
    except Exception:
        return None
    h, w = arr.shape[:2]
    u = (lon_grid.ravel() + 180.0) / 360.0
    v = (90.0 - lat_grid.ravel()) / 180.0
    cols = np.clip(np.rint(u * (w - 1)).astype(int), 0, w - 1)
    rows = np.clip(np.rint(v * (h - 1)).astype(int), 0, h - 1)
    rgb = arr[rows, cols]
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_default_earth_texture(url: str = DEFAULT_TEXTURE_URL) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "satellite-coverage-streamlit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20.0) as response:
            return response.read()
    except Exception:
        return None


def make_earth_mesh_trace(n_lat: int, n_lon: int, use_texture: bool, texture_bytes: bytes | None):
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y, z = geodetic_to_ecef(lat_grid, lon_grid, 0.0)

    # Grid faces: two triangles per lat/lon cell.
    i_faces = []
    j_faces = []
    k_faces = []
    for a in range(n_lat - 1):
        for b in range(n_lon - 1):
            p00 = a * n_lon + b
            p01 = a * n_lon + (b + 1)
            p10 = (a + 1) * n_lon + b
            p11 = (a + 1) * n_lon + (b + 1)
            i_faces.extend([p00, p01])
            j_faces.extend([p10, p10])
            k_faces.extend([p01, p11])

    vertexcolor = texture_to_vertex_colors(texture_bytes, lat_grid, lon_grid) if use_texture else None
    if vertexcolor is None:
        # Fallback: simple latitude/longitude procedural coloring.
        colors = []
        for la, lo in zip(lat_grid.ravel(), lon_grid.ravel()):
            if abs(la) > 66:
                colors.append("rgb(235,245,250)")
            elif (math.sin(math.radians(lo * 2.1)) + 0.7 * math.sin(math.radians(la * 3.4 + lo))) > 0.55:
                colors.append("rgb(60,130,70)")
            else:
                colors.append("rgb(35,85,155)")
        vertexcolor = colors

    return go.Mesh3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        i=i_faces,
        j=j_faces,
        k=k_faces,
        vertexcolor=vertexcolor,
        name="WGS84 Earth",
        opacity=1.0,
        flatshading=False,
        hoverinfo="skip",
        showscale=False,
    )


def short_name(name: str) -> str:
    match = re.search(r"PRN\s*(\d+)", name)
    if match:
        return "PRN " + match.group(1)
    if "ISS" in name.upper():
        return "ISS"
    return name[:22]


def add_line_trace(fig: go.Figure, x, y, z, name: str, width: float = 3.0, dash: str | None = None):
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"width": width, **({"dash": dash} if dash else {})},
            name=name,
            hoverinfo="name",
        )
    )


def make_coverage_figure(
    states: list[SatelliteState],
    title: str,
    min_elevation_deg: float,
    earth_n_lat: int,
    earth_n_lon: int,
    footprint_points: int,
    surface_line_offset_km: float,
    use_texture: bool,
    texture_bytes: bytes | None,
    orbit_tracks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
):
    fig = go.Figure()
    fig.add_trace(make_earth_mesh_trace(earth_n_lat, earth_n_lon, use_texture, texture_bytes))

    if states:
        fig.add_trace(
            go.Scatter3d(
                x=[s.ecef_x_km for s in states],
                y=[s.ecef_y_km for s in states],
                z=[s.ecef_z_km for s in states],
                mode="markers+text",
                marker={"size": 4},
                text=[short_name(s.name) for s in states],
                textposition="top center",
                name="satellite positions",
                customdata=np.array([[s.latitude_deg, s.longitude_deg, s.height_km, s.tle_age_days] for s in states]),
                hovertemplate=(
                    "%{text}<br>lat=%{customdata[0]:.3f} deg<br>"
                    "lon=%{customdata[1]:.3f} deg<br>height=%{customdata[2]:.1f} km<br>"
                    "TLE age=%{customdata[3]:.2f} days<extra></extra>"
                ),
            )
        )

    for state in states:
        # Sub-satellite point and radial line.
        px, py, pz = geodetic_to_ecef(state.latitude_deg, state.longitude_deg, surface_line_offset_km)
        sx, sy, sz = state.ecef_x_km, state.ecef_y_km, state.ecef_z_km
        fig.add_trace(
            go.Scatter3d(
                x=[px],
                y=[py],
                z=[pz],
                mode="markers",
                marker={"size": 3},
                name=f"{short_name(state.name)} subpoint",
                showlegend=False,
                hovertemplate=f"{state.name}<br>subpoint lat={state.latitude_deg:.3f} deg<br>subpoint lon={state.longitude_deg:.3f} deg<extra></extra>",
            )
        )
        add_line_trace(fig, [px, sx], [py, sy], [pz, sz], f"{short_name(state.name)} radial", width=2.0)

        lat_fp, lon_fp = footprint_boundary_latlon(state, min_elevation_deg, footprint_points)
        x_fp, y_fp, z_fp = geodetic_to_ecef(lat_fp, lon_fp, surface_line_offset_km)
        add_line_trace(fig, x_fp, y_fp, z_fp, f"{short_name(state.name)} footprint", width=4.0)

    if orbit_tracks:
        for track_name, (lat, lon, height) in orbit_tracks.items():
            x, y, z = geodetic_to_ecef(lat, lon, height)
            add_line_trace(fig, x, y, z, track_name, width=5.0, dash="dot")

    # Scale axes to include GPS/GNSS altitude.
    if states:
        radii = [math.sqrt(s.ecef_x_km**2 + s.ecef_y_km**2 + s.ecef_z_km**2) for s in states]
        max_radius = max([EARTH.mean_radius_km] + radii)
    else:
        max_radius = EARTH.mean_radius_km
    if orbit_tracks:
        for lat, lon, height in orbit_tracks.values():
            max_radius = max(max_radius, EARTH.mean_radius_km + float(np.nanmax(height)))
    limit = max_radius * 1.10

    fig.update_layout(
        title=title,
        height=760,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "xaxis": {"title": "ECEF X [km]", "range": [-limit, limit]},
            "yaxis": {"title": "ECEF Y [km]", "range": [-limit, limit]},
            "zaxis": {"title": "ECEF Z [km]", "range": [-limit, limit]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 0.95}},
        },
    )
    return fig


# =============================================================================
# 5) Streamlit UI helpers
# =============================================================================

def state_dataframe(states: list[SatelliteState]) -> pd.DataFrame:
    rows = []
    for s in states:
        rows.append(
            {
                "satellite": s.name,
                "lat [deg]": s.latitude_deg,
                "lon [deg]": s.longitude_deg,
                "altitude [km]": s.height_km,
                "TLE age [days]": s.tle_age_days,
                "TLE epoch [UTC]": s.tle_epoch_utc,
            }
        )
    return pd.DataFrame(rows)


def footprint_dataframe(states: list[SatelliteState], min_elevation_deg: float) -> pd.DataFrame:
    rows = []
    for s in states:
        f = footprint_summary(s, min_elevation_deg)
        rows.append(
            {
                "satellite": f.name,
                "min elevation [deg]": f.min_elevation_deg,
                "central angle [deg]": f.central_angle_deg,
                "footprint radius [km]": f.ground_radius_km,
                "footprint area [million km²]": f.spherical_area_km2 / 1e6,
            }
        )
    return pd.DataFrame(rows)


def tle_dataframe(records: list[TLERecord]) -> pd.DataFrame:
    return pd.DataFrame([quick_orbit_checks(r) for r in records])


def select_records_by_count(records: list[TLERecord], display_count: str) -> list[TLERecord]:
    if display_count == "All":
        return records
    n = int(display_count)
    return records[: min(n, len(records))]


def select_satellites(records: list[TLERecord], display_count: str, manual_select: bool) -> list[TLERecord]:
    records_sorted = sorted(records, key=lambda r: r.name)
    default_records = select_records_by_count(records_sorted, display_count)
    if not manual_select:
        return default_records
    name_to_record = {r.name: r for r in records_sorted}
    default_names = [r.name for r in default_records]
    selected_names = st.sidebar.multiselect(
        "Manual satellite selection",
        options=list(name_to_record.keys()),
        default=default_names,
        help="Use this when you want specific satellites instead of the first N alphabetically.",
    )
    return [name_to_record[name] for name in selected_names]


def compute_time_selection(ts, satellites: list[EarthSatellite], mode: str):
    if mode == "Latest TLE epoch":
        return latest_tle_epoch(satellites), "latest selected TLE epoch"
    if mode == "Now UTC":
        return now_utc_time(ts), "current UTC time"
    date_value = st.sidebar.date_input("Manual UTC date", value=datetime.now(timezone.utc).date())
    time_value = st.sidebar.time_input("Manual UTC time", value=datetime.now(timezone.utc).time().replace(microsecond=0))
    return parse_manual_utc(ts, date_value, time_value), "manual UTC time"


def show_metrics(states: list[SatelliteState], fp_df: pd.DataFrame):
    if not states:
        return
    heights = np.array([s.height_km for s in states], dtype=float)
    ages = np.array([s.tle_age_days for s in states], dtype=float)
    cols = st.columns(4)
    cols[0].metric("Displayed satellites", f"{len(states)}")
    cols[1].metric("Median altitude", f"{np.nanmedian(heights):.1f} km")
    cols[2].metric("Max TLE age", f"{np.nanmax(np.abs(ages)):.2f} days")
    if not fp_df.empty:
        cols[3].metric("Median footprint radius", f"{fp_df['footprint radius [km]'].median():.0f} km")


def load_records_from_ui() -> tuple[list[TLERecord], str, str]:
    source_mode = st.sidebar.radio(
        "TLE source",
        ["CelesTrak public feed", "Upload .txt / .tle"],
        horizontal=False,
    )

    if source_mode == "Upload .txt / .tle":
        uploaded = st.sidebar.file_uploader("Upload TLE file", type=["txt", "tle"])
        if uploaded is None:
            st.info("Upload a .txt or .tle file, or switch back to the CelesTrak public feed.")
            return [], "uploaded file", "waiting for upload"
        text = uploaded.read().decode("utf-8", errors="replace")
        records = parse_tle_text(text)
        return records, uploaded.name, "uploaded file"

    constellation = st.sidebar.selectbox("Constellation / feed", list(CELESTRAK_GROUPS.keys()), index=0)
    group = CELESTRAK_GROUPS[constellation]
    text = download_celestrak_group(group)
    records = parse_tle_text(text)
    if constellation == "ISS":
        iss = find_iss_record(records)
        records = [iss] if iss else records[:1]
    return records, f"CelesTrak GROUP={group}", constellation


def build_main_page():
    st.set_page_config(page_title="Satellite Coverage Viewer", layout="wide")
    st.title("Satellite coverage viewer")
    st.caption("3D coverage sanity checks from TLEs. Units: km, degrees, UTC. Propagation: Skyfield / SGP4.")

    with st.expander("Model notes", expanded=False):
        st.markdown(
            """
            This is a geometry-first sanity tool. Satellite states use TLE propagation and WGS84 lat/lon/height.
            The plotted frame is Earth-fixed ECEF XYZ in kilometers. Footprints are spherical-Earth line-of-sight caps
            for the selected minimum elevation angle. The plotted footprint lines are lifted slightly above the surface
            only to avoid visual z-fighting; the calculations still use the true surface.
            """
        )

    st.sidebar.header("Inputs of code")
    records, source_label, feed_label = load_records_from_ui()
    if not records:
        st.stop()

    st.sidebar.divider()
    display_count = st.sidebar.selectbox("How many satellites to display", DISPLAY_COUNT_OPTIONS, index=0)
    manual_select = st.sidebar.checkbox("Manually choose satellites", value=False)
    selected_records = select_satellites(records, display_count, manual_select)

    min_elev = st.sidebar.slider("Minimum elevation mask [deg]", min_value=0.0, max_value=45.0, value=5.0, step=1.0)
    time_mode = st.sidebar.selectbox("Snapshot time", ["Latest TLE epoch", "Now UTC", "Manual UTC"], index=0)

    st.sidebar.divider()
    st.sidebar.header("3D plot")
    earth_res = st.sidebar.select_slider(
        "Earth mesh resolution",
        options=["Fast", "Medium", "High", "Very high"],
        value="Medium",
        help="Higher values look nicer but can make browser rotation slower.",
    )
    res_map = {
        "Fast": (37, 73, 121),
        "Medium": (73, 145, 181),
        "High": (121, 241, 241),
        "Very high": (181, 361, 361),
    }
    earth_n_lat, earth_n_lon, footprint_points = res_map[earth_res]
    use_texture = st.sidebar.checkbox("Use Earth texture", value=True)
    texture_source = st.sidebar.radio("Texture source", ["Public texture URL", "Upload image", "Procedural fallback"], index=0)
    texture_bytes = None
    if use_texture and texture_source == "Public texture URL":
        texture_bytes = fetch_default_earth_texture()
    elif use_texture and texture_source == "Upload image":
        texture_upload = st.sidebar.file_uploader("Upload equirectangular Earth image", type=["jpg", "jpeg", "png"], key="earth_texture_upload")
        if texture_upload is not None:
            texture_bytes = texture_upload.read()
    elif texture_source == "Procedural fallback":
        use_texture = False

    surface_line_offset = st.sidebar.number_input(
        "Visual footprint lift [km]",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=10.0,
        help="Plot-only lift to prevent surface lines fighting with the Earth mesh.",
    )

    st.sidebar.divider()
    draw_iss_track = st.sidebar.checkbox("Show ISS one-orbit track in ISS section", value=True)

    ts = get_timescale()
    satellites = build_satellites(selected_records, ts)
    if not satellites:
        st.error("No valid satellites could be built from the selected TLE records.")
        st.stop()

    t, time_note = compute_time_selection(ts, satellites, time_mode)
    states = satellite_states(satellites, t)
    if not states:
        st.error("Propagation failed for the selected satellites.")
        st.stop()

    st.subheader(f"Main view: {feed_label}")
    st.write(
        f"Source: **{source_label}** · loaded **{len(records)}** TLE records · displaying **{len(states)}** satellites · "
        f"snapshot: **{format_time_utc(t)}** ({time_note}) · minimum elevation: **{min_elev:g}°**."
    )

    fp_df = footprint_dataframe(states, min_elev)
    show_metrics(states, fp_df)

    fig = make_coverage_figure(
        states=states,
        title=f"{feed_label} coverage | {format_time_utc(t)} | min elevation {min_elev:g} deg",
        min_elevation_deg=min_elev,
        earth_n_lat=earth_n_lat,
        earth_n_lon=earth_n_lon,
        footprint_points=footprint_points,
        surface_line_offset_km=surface_line_offset,
        use_texture=use_texture,
        texture_bytes=texture_bytes,
    )
    st.plotly_chart(fig, width='stretch')

    tab1, tab2, tab3 = st.tabs(["Propagated states", "Footprints", "TLE checks"])
    with tab1:
        st.markdown("WGS84 position at the selected snapshot. Altitude is height above the WGS84 ellipsoid.")
        st.dataframe(state_dataframe(states), width='stretch')
    with tab2:
        st.markdown("Footprint radius is the ground arc radius on a mean-radius spherical Earth.")
        st.dataframe(fp_df, width='stretch')
    with tab3:
        st.markdown("Quick TLE-derived sanity checks. Perigee/apogee are approximate two-body values from mean motion and eccentricity.")
        st.dataframe(tle_dataframe(selected_records), width='stretch')

    st.divider()
    show_iss_sanity_section(ts, min_elev, earth_n_lat, earth_n_lon, footprint_points, surface_line_offset, use_texture, texture_bytes, draw_iss_track)


def show_iss_sanity_section(
    ts,
    min_elev: float,
    earth_n_lat: int,
    earth_n_lon: int,
    footprint_points: int,
    surface_line_offset: float,
    use_texture: bool,
    texture_bytes: bytes | None,
    draw_iss_track: bool,
):
    st.subheader("ISS sanity-check section")
    st.write(
        "Same model as the main view: CelesTrak TLE, Skyfield/SGP4 propagation, WGS84 state extraction, "
        "and the same footprint geometry. This is useful because the ISS has well-known approximate altitude and inclination."
    )

    try:
        text = download_celestrak_group(CELESTRAK_GROUPS["ISS"])
        records = parse_tle_text(text)
        record = find_iss_record(records)
    except Exception as exc:
        st.warning(f"Could not load ISS from CelesTrak: {exc}")
        record = None

    if record is None:
        st.warning("ISS TLE was not found in the public stations feed.")
        return

    tle_checks = quick_orbit_checks(record)
    sat = build_satellites([record], ts)[0]
    t = sat.epoch
    state = satellite_state(sat, t)
    if state is None:
        st.warning("Could not propagate ISS TLE.")
        return

    inc = tle_checks["inclination_deg"]
    period_min = tle_checks["period_min"]
    height_ok = KNOWN_ISS_ALTITUDE_RANGE_KM[0] <= state.height_km <= KNOWN_ISS_ALTITUDE_RANGE_KM[1]
    inc_ok = abs(inc - KNOWN_ISS_INCLINATION_DEG) < 0.5 if np.isfinite(inc) else False
    period_ok = 88.0 <= period_min <= 96.0 if np.isfinite(period_min) else False

    cols = st.columns(3)
    cols[0].metric("ISS altitude", f"{state.height_km:.1f} km", "ok" if height_ok else "check")
    cols[1].metric("ISS inclination", f"{inc:.3f}°", "ok" if inc_ok else "check")
    cols[2].metric("ISS period", f"{period_min:.2f} min", "ok" if period_ok else "check")

    iss_fp_df = footprint_dataframe([state], min_elev)
    st.dataframe(
        pd.DataFrame(
            [
                {"check": "altitude", "model": f"{state.height_km:.1f} km", "reference": "370-460 km", "status": "ok" if height_ok else "check"},
                {"check": "inclination", "model": f"{inc:.3f} deg", "reference": "51.6 deg", "status": "ok" if inc_ok else "check"},
                {"check": "period", "model": f"{period_min:.2f} min", "reference": "about 90 min", "status": "ok" if period_ok else "check"},
            ]
        ),
        width='stretch',
    )

    orbit_tracks = None
    if draw_iss_track and np.isfinite(period_min):
        lat_tr, lon_tr, h_tr = orbit_track_latlon(sat, t, minutes=period_min, samples=241)
        orbit_tracks = {"ISS one-orbit track": (lat_tr, lon_tr, h_tr)}
        st.caption(
            f"ISS one-orbit latitude range: {float(np.min(lat_tr)):.2f}° to {float(np.max(lat_tr)):.2f}°; "
            f"expected near ±{inc:.2f}°."
        )

    fig = make_coverage_figure(
        states=[state],
        title=f"ISS sanity check | {format_time_utc(t)} | same footprint model",
        min_elevation_deg=min_elev,
        earth_n_lat=earth_n_lat,
        earth_n_lon=earth_n_lon,
        footprint_points=footprint_points,
        surface_line_offset_km=surface_line_offset,
        use_texture=use_texture,
        texture_bytes=texture_bytes,
        orbit_tracks=orbit_tracks,
    )
    st.plotly_chart(fig, width='stretch')
    st.dataframe(iss_fp_df, width='stretch')


# =============================================================================
# 6) App entrypoint
# =============================================================================

if __name__ == "__main__":
    build_main_page()
