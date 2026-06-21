#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:32:07 2026

@author: gillesbutikofer
"""

'''
### 1. Data import and filtering

The DME station data is imported from the local SDO Reporting HTML file. 
Since the first row of the table contains the real column names, it is used to 
rename the columns before removing it from the dataset.

Only Swiss-related DME stations are kept for the analysis. The filter includes 
stations whose responsible entity is listed as Switzerland, Skyguide, or the 
Federal Office for Civil Aviation.

### 2. Coordinate conversion

The DME coordinates are converted into decimal degrees. This is necessary 
because the original file contains coordinates in different formats, including 
compact DMS format and decimal-degree format.

This step is important because all later calculations, such as plotting, 
distance estimation, and terrain sampling, require the DME positions to be 
expressed consistently as latitude and longitude.

### 3. Switzerland boundary and terrain loading

The Switzerland boundary is loaded and used as a geographic reference for the plots. 
This does not directly affect the coverage calculation, but it helps visually 
check whether the DME positions and coverage maps are located correctly.

The terrain raster is then loaded to provide elevation data. 
This terrain information is required later to determine whether the radio 
signal between an aircraft and a DME station is blocked by mountains.

### 4. Initial validation plots

The first plots are used mainly as checkpoints. 
They verify that the DME stations are correctly positioned over Switzerland and 
that the terrain map aligns properly with the station locations.

These plots are not the final result of the analysis. Their purpose is to detect 
possible errors early, such as wrong coordinate conversion, incorrect map bounds, 
or mismatched terrain data.

### 5. DME altitude extraction

Each DME station is assigned an altitude by sampling the terrain raster at its 
longitude and latitude. This altitude is used as the starting height of the radio 
line-of-sight calculation.

A small antenna height is also added later to avoid assuming that the radio signal 
starts exactly at ground level. This is a simplified but more realistic assumption.

### 6. DME transmission power assumption

The code assigns two different power levels to the DME stations. 
Stations whose names contain a number are treated as ILS-DME and assigned a lower 
power of 100 W, while the others are treated as standard DME and assigned 1000 W.

This is an important modelling assumption because transmission power directly 
affects the estimated free-space detection range. 

### 7. Free-space detection radius

The free-space detection radius is calculated using a simplified free-space 
path-loss relationship. The calculation uses the DME transmission power, the 
assumed receiver sensitivity, and the signal wavelength.

This gives an ideal maximum range for each DME station, ignoring terrain obstruction. 
It represents the best-case radio coverage before considering mountains.

### 8. Free-space coverage map

A regular grid is created over the terrain map, with the resolution controlled 
by `nx` and `ny`. For each grid point, the code counts how many DME stations are 
within their free-space range.
This produces the first coverage matrix, where each cell represents the number of 
theoretically reachable DME stations. 
This map ignores terrain and is therefore useful as a reference case.

### 9. Terrain line-of-sight model

The aircraft altitude is defined using a flight level, for example FL100 for 10,000 ft. 
For each grid point and each DME within free-space range, the code checks whether the 
straight radio path between the DME and the aircraft is blocked by terrain.

The parameter `n_ray_samples` controls how many points are checked along this line. 
A higher value gives a more detailed obstruction check, but also increases computation time.

### 10. Terrain obstruction calculation

For every aircraft position in the grid, the code first selects only the DME stations 
that are close enough according to the free-space range. Then, only these candidate 
stations are tested with the terrain line-of-sight calculation.

This avoids wasting time ray-tracing DME stations that are already too far away. 
The result is a second coverage matrix showing how many DME stations are actually 
visible when terrain is included.

### 11. Speed improvement

The first version was slower because it repeatedly sampled the terrain raster using 
`src.sample()` inside the ray-tracing loop. The improved version loads the terrain 
into memory once and accesses it directly using raster row and column indices.

The code also preselects candidate DME stations using the free-space range before 
applying ray tracing. These two changes were the main reasons for the large speed 
improvement, around 30 times faster compared with the first version. 

### 12. Numerical artifact correction

A small margin is added inside the raster bounds when creating the grid. 
This avoids evaluating points exactly on the terrain raster boundary, where rounding 
can produce invalid pixel indices.

### 13. Coverage classification

The raw number of visible DME stations is converted into five coverage quality classes. 
The classes are: 
    no coverage for 0 DME, 
    insufficient coverage for 1–2 DME, 
    minimal coverage for 3 DME, 
    good coverage for 4 DME, 
    strong coverage for 5 or more DME.

This classification makes the final result easier to interpret than a continuous 
numerical count. It also reflects the operational idea that fewer than three DME 
stations are not enough for reliable positioning.

### 14. Final classified coverage maps

The final desired outputs are the classified coverage maps with the five coverage levels. 
The earlier plots are mainly verification steps, while these final maps summarize 
the actual quality of DME availability.

The final version is also plotted in the Swiss projected coordinate system, so the 
axes can be interpreted in kilometres instead of degrees. 
This makes distances easier to understand and avoids the misleading impression 
caused by latitude-longitude axes.


REMARKS:
    Several simplifications affect the precision of the calculated map, such as
    - Assumes that the signal travels along a straight line, visible or blocked. 
        real effects such as reflection, interferences or antenna patterns are neglected
    - Power assumptions: further investigation of the exact power of the DME might be necessary
    - Receiver sensitivity: -79 dB was a given value, however this is a key value
        in the creation of this map, and might differ from aircraft to aircraf.
    - Aircraft altitude is absolute altitude (above mean sea level)

'''




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to local HTML file
file_path = "SDO Reporting - DME.html"
tables = pd.read_html(file_path)
DME = tables[1]

# Take first row as column names
DME.columns = DME.iloc[0]
# Remove the first row from the data
DME = DME[1:].reset_index(drop=True)

swiss_entities = [
    "SWITZERLAND",
    "SKYGUIDE",
    "FEDERAL OFFICE FOR CIVIL AVIATION"
]

DME_CH = DME[DME["Responsible State"].str.strip().str.upper().isin(swiss_entities)].copy()


# --- 1) Convert coordinates ---
def coord_to_decimal(coord):
    coord = str(coord).strip()

    if not coord:
        return None

    direction = coord[-1].upper()
    value = coord[:-1]

    # Case 1: decimal degrees like 46.21517782 or 007.31261450
    # If the dot is in the first 3 chars, it is decimal format
    if "." in value and value.find(".") <= 3:
        dec = float(value)

    # Case 2: compact DMS like 472759.6 or 0055959.712
    else:
        if direction in ["N", "S"]:   # latitude -> DDMMSS.S
            deg = int(value[:2])
            min_ = int(value[2:4])
            sec = float(value[4:])
        elif direction in ["E", "W"]: # longitude -> DDDMMSS.S
            deg = int(value[:3])
            min_ = int(value[3:5])
            sec = float(value[5:])
        else:
            return None

        dec = deg + min_ / 60 + sec / 3600

    if direction in ["S", "W"]:
        dec = -dec

    return dec

DME_CH["lat"] = DME_CH["Latitude"].apply(coord_to_decimal)
DME_CH["lon"] = DME_CH["Longitude"].apply(coord_to_decimal)


# --- 2) Load country shapes and keep Switzerland ---
import geopandas as gpd

world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

switzerland = world[world["NAME"] == "Switzerland"]


# --- 3) Plot ---
fig, ax = plt.subplots(figsize=(10, 10))

switzerland.plot(ax=ax, color="white", edgecolor="black")
ax.scatter(DME_CH["lon"], DME_CH["lat"], marker="^", s=80, label="DME")

ax.set_title("DME stations in Switzerland")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.show()



# --- ALTITUDE INFORMATION 

import rasterio

src = rasterio.open("schweiz_terrain_master.tif")

print("Terrain CRS:", src.crs)

if src.crs is None:
    raise ValueError("Terrain raster has no CRS. Cannot safely use lon/lat coordinates.")

if src.crs.to_epsg() != 4326:
    print("WARNING: Terrain raster is not EPSG:4326. Coordinates must be transformed before sampling.")

print(src.bounds)     # xmin, ymin, xmax, ymax
print(src.transform)  # pixel-to-world transform
print(src.count)      # number of bands

terrain = src.read(1)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 10))

# Terrain as grayscale background
ax.imshow(
    terrain,
    cmap="gray",
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="upper"
)

# Switzerland border on top
switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# DME points
ax.scatter(DME_CH["lon"], DME_CH["lat"], marker="^", s=80, color="red", label="DME")

ax.set_xlim(src.bounds.left, src.bounds.right)
ax.set_ylim(src.bounds.bottom, src.bounds.top)

ax.set_title("DME stations in Switzerland on terrain map")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()


# Add altitude to DME_CH


# Prepare list of (lon, lat) tuples
coords = list(zip(DME_CH["lon"], DME_CH["lat"]))

# Sample terrain altitude values at each DME location.
# The antenna height is added later in the line-of-sight model.
altitudes = [val[0] for val in src.sample(coords)]

# Add terrain altitude to dataframe
DME_CH["altitude"] = altitudes



ILS_DME_power = 100 #W 
standard_DME_power = 1000 #W

DME_CH["power_W"] = np.where(
    DME_CH["Name"].str.contains(r"\d", na=False),  # contains a number
    ILS_DME_power,
    standard_DME_power
)

# Define constants 
f = 1e9 # 1GHz
c = 3e8 # 300 000 km/
Lambda = c/f

P_rx_min_dBm = -79

P_rx_min = 10**((P_rx_min_dBm-30)/10)

def detection_radius_fspl(P_tx, P_rx_min=P_rx_min, Lambda=Lambda):
    return (Lambda / (4 * np.pi)) * np.sqrt(P_tx / P_rx_min)

DME_CH["radius_m"] = DME_CH["power_W"].apply(detection_radius_fspl)
DME_CH["radius_km"] = DME_CH["radius_m"] / 1000

# Quick validation of the Swiss DME selection and simplified power classification
print("Number of Swiss DMEs:", len(DME_CH))
print(DME_CH[["Identification", "Name", "power_W", "radius_km"]])


# ------------ plot with ranges ------------

from matplotlib.patches import Ellipse

fig, ax = plt.subplots(figsize=(10, 10))

# Terrain as grayscale background
ax.imshow(
    terrain,
    cmap="gray",
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="upper"
)

# Switzerland border on top
switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# DME points
ax.scatter(DME_CH["lon"], DME_CH["lat"], marker="^", s=80, color="red", label="DME")

# Add range "circles"
for _, row in DME_CH.iterrows():
    lat = row["lat"]
    lon = row["lon"]
    r_km = row["radius_km"]

    # Convert km to degrees
    r_lat = r_km / 111.0
    r_lon = r_km / (111.0 * np.cos(np.radians(lat)))

    ellipse = Ellipse(
        (lon, lat),
        width=2 * r_lon,
        height=2 * r_lat,
        edgecolor="blue",
        facecolor="none",
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(ellipse)

ax.set_xlim(src.bounds.left, src.bounds.right)
ax.set_ylim(src.bounds.bottom, src.bounds.top)

ax.set_title("DME stations in Switzerland with free-space range")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()





# ============================================================
# COVERAGE MAPS
# 1) Free-space only
# 2) With terrain obstruction / line-of-sight
# ============================================================

from rasterio.transform import rowcol

# ---------------- USER PARAMETERS ----------------

nx = 200              # number of points in longitude direction
ny = 200              # number of points in latitude direction

FL = 100               # flight level, e.g. FL100 = 10,000 ft
n_ray_samples = 400   # number of terrain checks along each DME-aircraft line

aircraft_altitude_m = FL * 100 * 0.3048

aircraft_antenna_height_m = 0
# DME_CH["altitude"] contains terrain altitude only, so antenna height is added here.
dme_antenna_height_m = 10 # antenna height correction

# ---------------- PREPARE GRID ----------------

margin = 0.01  # degrees (~1 km)

x_grid = np.linspace(src.bounds.left + margin, src.bounds.right - margin, nx)
y_grid = np.linspace(src.bounds.bottom + margin, src.bounds.top - margin, ny)

XX, YY = np.meshgrid(x_grid, y_grid)

# ---------------- PREPARE DME ARRAYS ----------------

dme_lons = DME_CH["lon"].to_numpy()
dme_lats = DME_CH["lat"].to_numpy()
dme_alts = DME_CH["altitude"].to_numpy()
dme_ranges = DME_CH["radius_km"].to_numpy()

# ---------------- TERRAIN DATA IN MEMORY ----------------

terrain_array = terrain.copy()
transform = src.transform
nodata = src.nodata

# ---------------- BELOW-TERRAIN MASK ----------------
# Grid points where the selected flight level is below terrain elevation.
# These points are physically meaningless for reception and will be plotted in light gray.

flat_lons = XX.ravel()
flat_lats = YY.ravel()

grid_rows, grid_cols = rowcol(transform, flat_lons, flat_lats)

grid_rows = np.array(grid_rows).reshape(XX.shape)
grid_cols = np.array(grid_cols).reshape(XX.shape)

valid_grid = (
    (grid_rows >= 0) & (grid_rows < terrain_array.shape[0]) &
    (grid_cols >= 0) & (grid_cols < terrain_array.shape[1])
)

terrain_grid = np.full(XX.shape, np.nan, dtype=float)

terrain_grid[valid_grid] = terrain_array[
    grid_rows[valid_grid],
    grid_cols[valid_grid]
]

if nodata is not None:
    terrain_grid[terrain_grid == nodata] = np.nan

below_terrain_mask = terrain_grid >= aircraft_altitude_m

# ---------------- DISTANCE FUNCTION ----------------

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# ============================================================
# 1) FREE-SPACE COVERAGE
# ============================================================

coverage_free = np.zeros_like(XX, dtype=int)

for i in range(len(dme_lons)):

    dist_km = haversine_km(
        XX,
        YY,
        dme_lons[i],
        dme_lats[i]
    )

    coverage_free += dist_km <= dme_ranges[i]

# ---------------- PLOT FREE-SPACE COVERAGE ----------------

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    coverage_free,
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="lower",
    cmap="viridis"
)

switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
ax.scatter(
    dme_lons,
    dme_lats,
    marker="^",
    s=60,
    color="red",
    label="DME"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of DME in range")

ax.set_title("Number of reachable DME - free-space only")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()

# ============================================================
# 2) TERRAIN LINE-OF-SIGHT FUNCTION
# ============================================================

def is_visible_los_fast(lon_aircraft, lat_aircraft, alt_aircraft,
                        lon_dme, lat_dme, alt_dme,
                        n_samples=50):

    # Points along line between DME and aircraft
    fractions = np.linspace(0, 1, n_samples)
    lons = lon_dme + fractions * (lon_aircraft - lon_dme)
    lats = lat_dme + fractions * (lat_aircraft - lat_dme)

    # Convert lon/lat to raster row/column
    rows, cols = rowcol(transform, lons, lats)
    rows = np.array(rows)
    cols = np.array(cols)

    valid = (
        (rows >= 0) & (rows < terrain_array.shape[0]) &
        (cols >= 0) & (cols < terrain_array.shape[1])
    )

    if not np.all(valid):
        # Ignore points outside terrain instead of killing visibility.
        # Keep the same fractions as the remaining terrain samples so that
        # terrain_vals and ray_altitudes always have matching lengths.
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 3:  # not enough points to evaluate
            return True

        rows = rows[valid_idx]
        cols = cols[valid_idx]
        fractions = fractions[valid_idx]

    terrain_vals = terrain_array[rows, cols].astype(float)
    
    terrain_valid = np.isfinite(terrain_vals)
    
    if nodata is not None:
        terrain_valid &= terrain_vals != nodata
    
    terrain_vals = terrain_vals[terrain_valid]
    fractions = fractions[terrain_valid]
    
    if len(terrain_vals) < 3:
        return True
    
    z_start = alt_dme + dme_antenna_height_m
    z_end = alt_aircraft + aircraft_antenna_height_m
    
    ray_altitudes = z_start + fractions * (z_end - z_start)
    
    terrain_between = terrain_vals[1:-1]
    ray_between = ray_altitudes[1:-1]
    
    return np.all(terrain_between < ray_between)

# ============================================================
# 3) COVERAGE WITH TERRAIN OBSTRUCTION
# ============================================================

coverage_terrain = np.zeros_like(XX, dtype=int)

for iy in range(ny):
    print(f"Processing row {iy + 1}/{ny}")

    for ix in range(nx):

        lon_p = XX[iy, ix]
        lat_p = YY[iy, ix]

        # Skip points where the selected flight level would be below terrain.
        # This keeps the raw terrain-coverage matrix physically meaningful,
        # not only the final classified plot.
        if below_terrain_mask[iy, ix]:
            coverage_terrain[iy, ix] = 0
            continue

        # First check free-space range for all DME
        dist_km = haversine_km(
            lon_p,
            lat_p,
            dme_lons,
            dme_lats
        )

        candidates = np.where(dist_km <= dme_ranges)[0]

        count_visible = 0

        for i in candidates:

            visible = is_visible_los_fast(
                lon_aircraft=lon_p,
                lat_aircraft=lat_p,
                alt_aircraft=aircraft_altitude_m,
                lon_dme=dme_lons[i],
                lat_dme=dme_lats[i],
                alt_dme=dme_alts[i],
                n_samples=n_ray_samples
            )

            if visible:
                count_visible += 1

        coverage_terrain[iy, ix] = count_visible

# ---------------- PLOT TERRAIN COVERAGE ----------------

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    coverage_terrain,
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="lower",
    cmap="viridis"
)

switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
ax.scatter(
    dme_lons,
    dme_lats,
    marker="^",
    s=60,
    color="red",
    label="DME"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of visible DME")

ax.set_title(f"Number of visible DME with terrain obstruction - FL{FL}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()

# ============================================================
# 4) OPTIONAL DIFFERENCE MAP
# ============================================================

coverage_lost = coverage_free - coverage_terrain

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    coverage_lost,
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="lower",
    cmap="magma"
)

switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
ax.scatter(
    dme_lons,
    dme_lats,
    marker="^",
    s=60,
    color="cyan",
    label="DME"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("DME lost due to terrain obstruction")

ax.set_title(f"DME lost due to terrain obstruction - FL{FL}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()




# ============================================================
# CLASSIFY COVERAGE LEVELS
# ============================================================

coverage_class = np.zeros_like(coverage_terrain)

coverage_class[coverage_terrain == 0] = 0          # no coverage
coverage_class[(coverage_terrain >= 1) & (coverage_terrain <= 2)] = 1
coverage_class[coverage_terrain == 3] = 2
coverage_class[coverage_terrain == 4] = 3
coverage_class[coverage_terrain >= 5] = 4

# Copy used only for plotting
# Class 5 means aircraft altitude is below terrain
coverage_class_plot = coverage_class.copy()
coverage_class_plot[below_terrain_mask] = 5

# ============================================================
# PLOT CLASSIFIED COVERAGE
# ============================================================

from matplotlib.colors import ListedColormap

# Custom discrete colormap (5 classes)
cmap = ListedColormap([
    "black",      # 0 DME
    "red",        # 1-2 DME
    "orange",     # 3 DME
    "yellow",     # 4 DME
    "green",      # 5+ DME
    "lightgray"   # below terrain
])

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    coverage_class_plot,
    extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
    origin="lower",
    cmap=cmap,
    vmin=0,
    vmax=5,
    interpolation="nearest"
)

switzerland.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

ax.scatter(
    dme_lons,
    dme_lats,
    marker="^",
    s=60,
    color="blue",
    label="DME"
)

# Custom colorbar labels
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
cbar.ax.set_yticklabels([
    "0 DME (None)",
    "1–2 DME (Insufficient)",
    "3 DME (Minimal)",
    "4 DME (Good)",
    "5+ DME (Strong)",
    "Below terrain"
])

ax.set_title(f"DME Coverage Quality with Terrain - FL{FL}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()



# In kilometers ---
# ============================================================
# PLOT CLASSIFIED COVERAGE IN SWISS CRS WITH KM TICK LABELS
# ============================================================

from pyproj import Transformer
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# WGS84 lon/lat -> Swiss LV95 meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

# Transform grid and DME points to meters
XX_m, YY_m = transformer.transform(XX, YY)
dme_x_m, dme_y_m = transformer.transform(dme_lons, dme_lats)

# Transform Switzerland border to meters
switzerland_proj = switzerland.to_crs("EPSG:2056")

# Custom discrete colormap (5 classes)
cmap = ListedColormap([
    "black",      # 0 DME
    "red",        # 1-2 DME
    "orange",     # 3 DME
    "yellow",     # 4 DME
    "green",      # 5+ DME
    "lightgray"   # below terrain
])


fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    coverage_class_plot,
    extent=[
        XX_m.min(), XX_m.max(),
        YY_m.min(), YY_m.max()
    ],
    origin="lower",
    cmap=cmap,
    vmin=0,
    vmax=5,
    interpolation="nearest"
)

switzerland_proj.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=1
)

ax.scatter(
    dme_x_m,
    dme_y_m,
    marker="^",
    s=60,
    color="blue",
    label="DME"
)

# Show axis labels in km instead of meters
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/1000:.0f}"))

ax.set_aspect("equal", adjustable="box")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)

cbar = plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3, 4, 5])
cbar.ax.set_yticklabels([
    "0 DME (None)",
    "1–2 DME (Insufficient)",
    "3 DME (Minimal)",
    "4 DME (Good)",
    "5+ DME (Strong)",
    "Below terrain"
])

ax.set_title(f"DME Coverage Quality with Terrain - FL{FL}")
ax.set_xlabel("Distance East [km]")
ax.set_ylabel("Distance North [km]")
ax.legend()

plt.savefig(f"DME Coverage Quality with Terrain - FL{FL}.png")
plt.show()










