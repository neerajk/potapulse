"""Clip the national BOFEK2020 soil-unit polygons to the Limburg province
boundary. Static local layer (already present as a zip) - no download, read
directly via /vsizip/ with bbox pushdown then an exact polygon clip, same
pattern as fetch/waterways.py's HOTOSM clip.
"""
from __future__ import annotations

import geopandas as gpd

from limburg.common import REPO_ROOT, load_config


def fetch_bofek_clipped(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    cache_path = processed_dir / "bofek2020_clipped.gpkg"
    if cache_path.exists():
        print(f"Loading cached BOFEK2020 clip from {cache_path}")
        return gpd.read_file(cache_path)

    bcfg = cfg["bofek2020"]
    zip_path = REPO_ROOT / bcfg["shapefile_zip_path"]
    if not zip_path.exists():
        raise RuntimeError(f"{zip_path} not found - this is a user-supplied static layer, not a live fetch.")

    vsi_path = f"/vsizip/{zip_path}/{bcfg['shp_name']}"
    bbox = tuple(boundary.to_crs("EPSG:28992").total_bounds)
    print(f"Reading BOFEK2020 from {zip_path.name}, bbox-filtered to {bbox}")
    units = gpd.read_file(vsi_path, bbox=bbox, engine="pyogrio")
    print(f"{len(units)} BOFEK2020 units intersect the province bbox (pre-clip)")

    units = gpd.clip(units, boundary.to_crs(units.crs))
    units = units[~units.geometry.is_empty & units.geometry.notna()].reset_index(drop=True)
    print(f"{len(units)} BOFEK2020 units after clipping to the exact province polygon")

    code_field = bcfg["code_field"]
    n_null = int(units[code_field].isna().sum())
    print(f"{n_null}/{len(units)} units have a null {code_field} (e.g. open water - expected, not an error)")

    processed_dir.mkdir(parents=True, exist_ok=True)
    units.to_file(cache_path, driver="GPKG")
    print(f"Wrote {cache_path}")
    return units


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    fetch_bofek_clipped(cfg, boundary)
