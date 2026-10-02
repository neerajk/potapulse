"""Step: fetch a Limburg-wide AHN4 DTM tile grid via PDOK's AHN WCS
(https://service.pdok.nl/rws/ahn/wcs/v1_0), for the "mosaic of obtained DTM"
panel in visualize_overview.py.

AHN4 (not AHN5) is used: confirmed live 2026-10-02 via this WCS's
GetCapabilities that only two coverages are published (dtm_05m, dsm_05m) -
no "ahn5" coverage id and no separate AHN5 WCS endpoint exists yet. dtm_05m
is therefore the best/only live-fetchable DTM option.

The native product is 0.5m - a full-resolution mosaic for all of Limburg
(~2200 km^2) would be 100s of GB, impractical for an overview figure (the
same reasoning br_simulation's own province DEM overview already documents
for Groningen, see src/br_simulation/visualization/province_overview.py).
Each tile is instead requested already downsampled to
data_sources.ahn4_dtm.overview_resolution_m via the WCS's Scaling extension
(SCALEFACTOR=native_resolution_m/overview_resolution_m) - confirmed live
2026-10-02 that this WCS advertises and honors it (GetCapabilities lists
ows:Profile .../WCS_service-extension_scaling/1.0/conf/scaling): a 20km x
20km subset at SCALEFACTOR=0.05 returns a 2000x2000px 10m/px GeoTIFF.

Tiling is still required because the WCS enforces a MAXSIZE=4000px/axis cap
on the *output* grid regardless of scaling (confirmed live: a request whose
post-scale output would exceed 4000px/axis comes back HTTP 400 with
"Raster size out of range ... MAXSIZE=4000"). Since output_px =
subset_m / overview_resolution_m, tile_size_m is kept safely under
MAXSIZE_PX * overview_resolution_m.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import rasterio
import requests
from rasterio.io import MemoryFile
from shapely.geometry import box

from limburg.common import REPO_ROOT, load_config

HTTP_TIMEOUT_S = 240
MAXSIZE_PX = 4000  # confirmed live against this WCS's GetCoverage error message
SAFETY_MARGIN = 0.9  # stay comfortably under the hard cap


def _tile_grid(bounds: tuple[float, float, float, float], tile_size_m: float) -> list[tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    tiles = []
    x = minx
    while x < maxx:
        x_end = min(x + tile_size_m, maxx)
        y = miny
        while y < maxy:
            y_end = min(y + tile_size_m, maxy)
            tiles.append((x, y, x_end, y_end))
            y = y_end
        x = x_end
    return tiles


def _fetch_tile(src_cfg: dict, bbox: tuple[float, float, float, float], target_res_m: float, out_path: Path) -> None:
    scalefactor = src_cfg["native_resolution_m"] / target_res_m
    minx, miny, maxx, maxy = bbox
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": src_cfg["coverage_id"],
        "SUBSET": [f"x({minx},{maxx})", f"y({miny},{maxy})"],
        "SCALEFACTOR": scalefactor,
        "FORMAT": "image/tiff",
    }

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(src_cfg["wcs_url"], params=params, timeout=HTTP_TIMEOUT_S)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "tiff" not in content_type.lower():
                raise RuntimeError(
                    f"WCS GetCoverage failed for tile {bbox}: HTTP {resp.status_code}, "
                    f"Content-Type={content_type!r}, body[:300]={resp.text[:300]!r}"
                )
            break
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"WCS GetCoverage permanently failed for tile {bbox}: {last_exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryFile(resp.content) as mem, mem.open() as dataset:
        profile = dataset.profile
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dataset.read())


def fetch_ahn_dtm_tiles(cfg: dict, province_boundary: gpd.GeoDataFrame) -> list[Path]:
    """Fetch every AHN4 DTM tile covering the Limburg province bounding box
    (clipped to the WCS's declared coverage extent, and skipped if it doesn't
    actually intersect the province polygon), downsampled to
    data_sources.ahn4_dtm.overview_resolution_m. Returns the fetched tile
    paths - mosaic_dtm.py merges these into one province-wide raster.
    """
    src_cfg = cfg["data_sources"]["ahn4_dtm"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"] / "ahn4"
    target_res_m = src_cfg["overview_resolution_m"]
    tile_size_m = math.floor(MAXSIZE_PX * target_res_m * SAFETY_MARGIN)

    province_bounds = tuple(province_boundary.total_bounds)
    coverage_bounds = tuple(src_cfg["coverage_bounds_28992"])
    province_geom = province_boundary.geometry.union_all()

    tile_paths: list[Path] = []
    for minx, miny, maxx, maxy in _tile_grid(province_bounds, tile_size_m):
        cminx, cminy = max(minx, coverage_bounds[0]), max(miny, coverage_bounds[1])
        cmaxx, cmaxy = min(maxx, coverage_bounds[2]), min(maxy, coverage_bounds[3])
        if cminx >= cmaxx or cminy >= cmaxy:
            continue
        if not province_geom.intersects(box(cminx, cminy, cmaxx, cmaxy)):
            continue

        tile_id = f"dtm_{int(cminx)}_{int(cminy)}"
        out_path = raw_dir / f"{tile_id}.tif"
        if out_path.exists():
            print(f"{tile_id} already fetched, skipping ({out_path})")
            tile_paths.append(out_path)
            continue

        print(
            f"Fetching AHN4 DTM tile {tile_id} "
            f"({cmaxx - cminx:.0f}m x {cmaxy - cminy:.0f}m @ {target_res_m}m/px)"
        )
        _fetch_tile(src_cfg, (cminx, cminy, cmaxx, cmaxy), target_res_m, out_path)
        tile_paths.append(out_path)

    if not tile_paths:
        raise RuntimeError("No AHN4 DTM tiles were fetched for the Limburg province bbox")
    return tile_paths


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    fetch_ahn_dtm_tiles(cfg, boundary)
