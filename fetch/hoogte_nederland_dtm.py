"""Standalone script: fetch the "Hoogte - Land 1 mtr - Digital Terrain Model
(DTM) - AHN (INSPIRE Geharmoniseerd)" dataset
(https://data.overheid.nl/en/dataset/38665-...) for the Limburg province,
downsample server-side to 10m/px, clip to the exact province polygon, and
plot it with the seed potato plots overlaid.

Source: resolved from the data.overheid.nl dataset page via its linked
Nationaal Georegister (NGR) metadata record (uuid
6a1c46ee-30ed-4e5e-89b7-6b12b8bc71bf) to the actual PDOK WCS backing it -
confirmed live 2026-10-02:
  https://service.pdok.nl/rws/hoogte-nederland-land-dtm/wcs/v1_0
  coverage "El.GridCoverage", ~1m native pixels, native CRS EPSG:4258
  (ETRS89 lat/lon), MapServer-backed WCS 2.0.1.
This is a *different* product/service from the AHN4 dtm_05m WCS used
elsewhere in this directory (fetch_ahn_dtm.py) - same underlying AHN survey,
but this is its INSPIRE-harmonized "land DTM" distribution, at coarser
native resolution and on a geographic (not RD New) native grid.

Despite the native CRS being geographic, this WCS's GetCapabilities
advertises - and a live GetCoverage request confirmed - EPSG:28992 works as
both SUBSETTINGCRS and OUTPUTCRS, so every request/response here stays in
the project's standard CRS with no separate reprojection step.

Downsampling to 10m/px happens server-side via the WCS scaling extension's
SCALESIZE parameter (explicit output width/height in pixels), not
SCALEFACTOR - confirmed live that SCALEFACTOR applied against this
service's native geographic grid produces non-square, non-round pixel sizes
once reprojected to EPSG:28992 (8.85m x 14.09m for a SCALEFACTOR nominally
aiming at 10m), whereas SCALESIZE computed directly in the output
resolution gives exact 10.0m x 10.0m pixels. This is also far more
bandwidth-efficient than fetching ~1m native tiles and resampling locally.

Tiling is required because this WCS enforces the same MAXSIZE=4000px/axis
cap as the AHN4 WCS (confirmed live via the identical "Raster size out of
range ... MAXSIZE=4000" error) - tile_size_m is kept safely under
MAXSIZE_PX * overview_resolution_m, same formula fetch_ahn_dtm.py uses.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.process.clip_to_boundary import clip_raster_to_boundary
from limburg.fetch.ahn4_dtm import MAXSIZE_PX, SAFETY_MARGIN, _tile_grid
from limburg.common import REPO_ROOT, load_config
from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots

HTTP_TIMEOUT_S = 240

apply_style()


def _fetch_tile(src_cfg: dict, bbox: tuple[float, float, float, float], target_res_m: float, out_path: Path) -> None:
    minx, miny, maxx, maxy = bbox
    width_px = round((maxx - minx) / target_res_m)
    height_px = round((maxy - miny) / target_res_m)
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{src_cfg['request_crs'].split(':')[1]}"
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": src_cfg["coverage_id"],
        "SUBSET": [f"x({minx},{maxx})", f"y({miny},{maxy})"],
        "SUBSETTINGCRS": crs_uri,
        "OUTPUTCRS": crs_uri,
        "SCALESIZE": [f"x({width_px})", f"y({height_px})"],
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


def fetch_hoogte_nederland_dtm_tiles(cfg: dict, boundary: gpd.GeoDataFrame) -> list[Path]:
    """Fetch every tile covering the Limburg province bbox, already
    downsampled server-side to
    data_sources.hoogte_nederland_dtm.overview_resolution_m - skips any grid
    cell that doesn't actually intersect the province polygon. Returns the
    fetched tile paths - mosaic_hoogte_nederland_dtm merges these into one
    province-wide raster."""
    src_cfg = cfg["data_sources"]["hoogte_nederland_dtm"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"] / "hoogte_nederland_dtm"
    target_res_m = src_cfg["overview_resolution_m"]
    tile_size_m = math.floor(MAXSIZE_PX * target_res_m * SAFETY_MARGIN)

    province_bounds = tuple(boundary.total_bounds)
    province_geom = boundary.geometry.union_all()

    tile_paths: list[Path] = []
    for minx, miny, maxx, maxy in _tile_grid(province_bounds, tile_size_m):
        if not province_geom.intersects(box(minx, miny, maxx, maxy)):
            continue

        tile_id = f"dtm1m_{int(minx)}_{int(miny)}"
        out_path = raw_dir / f"{tile_id}.tif"
        if out_path.exists():
            print(f"{tile_id} already fetched, skipping ({out_path})")
            tile_paths.append(out_path)
            continue

        print(f"Fetching {tile_id} ({maxx - minx:.0f}m x {maxy - miny:.0f}m @ {target_res_m}m/px)")
        _fetch_tile(src_cfg, (minx, miny, maxx, maxy), target_res_m, out_path)
        tile_paths.append(out_path)

    if not tile_paths:
        raise RuntimeError("No Hoogte Nederland DTM tiles were fetched for the Limburg province bbox")
    return tile_paths


def mosaic_hoogte_nederland_dtm(cfg: dict, tile_paths: list[Path]) -> Path:
    """Tiles are fetched directly in EPSG:28992 at the same resolution, so
    unlike the Sentinel mosaics elsewhere in this directory, no reprojection
    step is needed here - just a straight merge."""
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "hoogte_nederland_dtm_mosaic.tif"
    if out_path.exists():
        print(f"Hoogte Nederland DTM mosaic already computed, skipping ({out_path})")
        return out_path

    with rasterio.open(tile_paths[0]) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata

    print(f"Merging {len(tile_paths)} tile(s) into one Hoogte Nederland DTM mosaic...")
    mosaic, mosaic_transform = rio_merge(tile_paths, nodata=nodata)
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    is_nodata = np.isnan(mosaic[0]) if (nodata is not None and np.isnan(nodata)) else (mosaic[0] == nodata)
    valid = mosaic[0][~is_nodata]
    print(
        f"Hoogte Nederland DTM mosaic written: {mosaic.shape[2]}x{mosaic.shape[1]}px, "
        f"elevation range [{valid.min():.1f}, {valid.max():.1f}] m -> {out_path}"
    )
    return out_path


def visualize_hoogte_nederland_dtm(
    cfg: dict, dtm_path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame
) -> Path:
    with rasterio.open(dtm_path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    is_nodata = np.isnan(data) if (nodata is not None and np.isnan(nodata)) else (data == nodata)
    masked = np.ma.masked_where(is_nodata, data)
    valid = data[~is_nodata]

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=float(valid.min()), vmax=float(valid.max()), extent=extent, zorder=1)
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    plots.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}"
    )

    ax.set_title("Hoogte Nederland - land DTM (1m source, 10m/px) - Limburg province")
    add_map_chrome(ax, str(crs))
    add_scale_bar(ax)
    add_north_arrow(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Elevation (m)", rotation=270, labelpad=15)
    ax.legend(loc="upper right", fontsize=7)

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hoogte_nederland_dtm_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)

    tile_paths = fetch_hoogte_nederland_dtm_tiles(cfg, boundary)
    mosaic_path = mosaic_hoogte_nederland_dtm(cfg, tile_paths)

    with rasterio.open(mosaic_path) as ref:
        nodata = ref.nodata
    clipped_path = clip_raster_to_boundary(
        mosaic_path, boundary, processed_dir / "hoogte_nederland_dtm_mosaic_clipped.tif", nodata=nodata
    )

    visualize_hoogte_nederland_dtm(cfg, clipped_path, boundary, plots)


if __name__ == "__main__":
    main()
