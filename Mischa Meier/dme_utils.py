import numpy as np
from pyproj import Transformer
import srtm
import pandas as pd

# Re-init these inside the module so workers can access them
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
inv_transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
elevation_data = srtm.get_data()


def check_vlos(p1_ecef, p2_ecef, terrain_data, samples=100):
    """
    Checks if the line between p1 and p2 is obstructed by terrain.
    p1: DME ECEF (x, y, z)
    p2: Aircraft ECEF (x, y, z)
    """
    # Create sample points along the path in ECEF
    # We skip the very first and last points to avoid ground collision at the station
    fractions = np.linspace(0.01, 0.99, samples)

    for f in fractions:
        # Interpolate ECEF point
        px = p1_ecef[0] + f * (p2_ecef[0] - p1_ecef[0])
        py = p1_ecef[1] + f * (p2_ecef[1] - p1_ecef[1])
        pz = p1_ecef[2] + f * (p2_ecef[2] - p1_ecef[2])

        # Convert back to Lat/Lon/Alt to check against SRTM
        lon, lat, alt_ray = inv_transformer.transform(px, py, pz)

        ground_elev = terrain_data.get_elevation(lat, lon)
        if ground_elev is not None and ground_elev > alt_ray:
            return False  # Obstructed

    return True  # Clear LOS


def calculate_point_coverage(args):
    """
    Worker function for a single grid point.
    args: (grid_pt, dme_list, r_enroute, r_terminal)
    """
    grid_pt, dme_list, r_enroute, r_terminal = args
    ac_ecef = grid_pt[:3]  # [x, y, z]
    visible_count = 0

    # Terrain data needs to be re-accessed or passed
    # Note: Depending on the srtm library version, you might need to
    # re-init the srtm.get_data() inside the worker if it's not picklable.
    import srtm

    local_terrain = srtm.get_data()

    for dme in dme_list:
        dme_xyz = np.array([dme["x"], dme["y"], dme["z"]])
        r_limit = r_enroute if dme["TYPE"].lower() == "enroute" else r_terminal

        # 1. Fast NumPy distance check
        dist = np.linalg.norm(ac_ecef - dme_xyz)

        if dist <= r_limit:
            # 2. VLOS Check (The bottleneck)
            if check_vlos(dme_xyz, ac_ecef, local_terrain, samples=30):
                visible_count += 1

    return visible_count
