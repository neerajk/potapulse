"""Standalone script: fetch Copernicus DEM EEA-10 (10m, Europe-only DSM
distributed by the Copernicus Data Space Ecosystem / CDSE) for the Limburg
province, clip to the exact province polygon, and plot an overview figure
with the seed potato plots overlaid. Independent of run_limburg.py's AHN4
pipeline - own tile cache, own mosaic, own figure - but reuses the cached
province boundary / seed potato plots and the same clip_raster_to_boundary
helper as the rest of this directory.

Source: CDSE STAC collection "cop-dem-eea-10-laea-tif" (confirmed live
2026-10-02 via https://catalogue.dataspace.copernicus.eu/stac/collections -
the only EEA-10 collection present; GLO-30/GLO-90 are separate, coarser
collections on the same catalog). Despite the "-laea-tif" name, the
distributed per-tile GeoTIFFs are EPSG:4326 (confirmed via item
properties["proj:code"]), one-degree cells named by grid:code "CDEM-N50E006"
etc - same 1deg grid convention as the public GLO-30 archive, just at 10m
postings.

Auth: STAC item search is public; downloading actual tile data is not. Each
item's "product" asset is a zipped product fetched via CDSE's OData $value
endpoint, which requires an OIDC bearer token. Requires a free CDSE account
(https://dataspace.copernicus.eu/) - set CDSE_USERNAME and CDSE_PASSWORD env
vars before running; the token is obtained via CDSE's public-client
password-grant flow (no client secret needed). License: ESA User License for
Copernicus Contributing Missions Data (linked from the collection's STAC
metadata) - this is CCM data, not open like GLO-30, so it isn't on the
anonymous AWS archive the rest of this project otherwise uses.

Tile-selection quirk: a bbox search against this collection also returns (a)
duplicate reprocessing versions of the same 1deg cell and (b) entries on a
different, fractional-degree grid (non-conforming grid:code width - confirmed
live 2026-10-02, e.g. "CDEM-N30E39" alongside the standard "CDEM-N50E006").
_select_canonical_items keeps only grid_code_pattern-conforming cells and,
per cell, the item with the most recent "published" timestamp, so the
pipeline downloads exactly one deterministic tile per degree cell instead of
silently picking whichever the API happened to return first.
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.process.clip_to_boundary import clip_raster_to_boundary
from limburg.common import REPO_ROOT, load_config
from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots

HTTP_TIMEOUT_S = 60
DOWNLOAD_TIMEOUT_S = 1800
DOWNLOAD_CHUNK_BYTES = 1 << 20

apply_style()


def _get_access_token(src_cfg: dict) -> str:
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "CDSE_USERNAME and CDSE_PASSWORD env vars are required (free account at "
            "https://dataspace.copernicus.eu/) to download Copernicus DEM EEA-10 tiles."
        )
    resp = requests.post(
        src_cfg["token_url"],
        data={
            "client_id": src_cfg["client_id"],
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _search_items(src_cfg: dict, bbox: tuple[float, float, float, float]) -> list[dict]:
    url = f"{src_cfg['stac_endpoint']}/collections/{src_cfg['collection']}/items"
    params = {"bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": 100}
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["features"]


def _select_canonical_items(src_cfg: dict, items: list[dict]) -> dict[str, dict]:
    """One item per standard 1deg grid cell - see module docstring on why a
    raw bbox search can't be trusted as-is. Returns {grid_code: item}."""
    pattern = re.compile(src_cfg["grid_code_pattern"])
    by_cell: dict[str, dict] = {}
    dropped_nonstandard = 0
    for item in items:
        grid_code = item["properties"].get("grid:code", "")
        if not pattern.match(grid_code):
            dropped_nonstandard += 1
            continue
        current = by_cell.get(grid_code)
        if current is None or item["properties"]["published"] > current["properties"]["published"]:
            by_cell[grid_code] = item
    if dropped_nonstandard:
        print(f"Dropped {dropped_nonstandard} non-standard-grid duplicate item(s) from the bbox search")
    print(f"Selected {len(by_cell)} canonical tile(s): {sorted(by_cell)}")
    return by_cell


def _download_and_extract_tile(item: dict, src_cfg: dict, raw_dir: Path) -> Path:
    grid_code = item["properties"]["grid:code"]
    out_tif_path = raw_dir / f"{grid_code}_DEM.tif"
    if out_tif_path.exists():
        print(f"{grid_code} already fetched, skipping ({out_tif_path})")
        return out_tif_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{grid_code}.zip"
    token = _get_access_token(src_cfg)
    product_url = item["assets"]["product"]["href"]
    print(f"Downloading {grid_code} zipped product from {product_url}")
    with requests.get(
        product_url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=DOWNLOAD_TIMEOUT_S
    ) as resp:
        if not resp.ok:
            # CDSE returns 403 with a body explaining *why* (most commonly:
            # the account hasn't accepted the Copernicus Contributing
            # Missions EULA yet, required separately from having a valid
            # token) - surface that instead of a bare "403 Forbidden".
            raise RuntimeError(
                f"Download failed for {grid_code}: HTTP {resp.status_code} - {resp.text[:500]!r}. "
                f"If this is a 403, check that your CDSE account has accepted the Copernicus "
                f"Contributing Missions data license at https://dataspace.copernicus.eu/ (account "
                f"dashboard -> Terms and Conditions / Licenses)."
            )
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("_DEM.tif"))
        with zf.open(member) as src, open(out_tif_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print(f"Extracted {grid_code} DEM tile -> {out_tif_path}")
    return out_tif_path


def fetch_copernicus_dem_eea10_tiles(cfg: dict, boundary: gpd.GeoDataFrame) -> list[Path]:
    """Fetch every Copernicus DEM EEA-10 tile covering the Limburg province
    bbox. Returns the extracted tile paths - mosaic_copernicus_dem_eea10
    merges these into one province-wide raster."""
    src_cfg = cfg["data_sources"]["copernicus_dem_eea10"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"] / "copernicus_dem_eea10"

    bbox = tuple(boundary.to_crs(4326).total_bounds)
    print(f"Searching {src_cfg['collection']} for tiles intersecting bbox {bbox}")
    items = _search_items(src_cfg, bbox)
    canonical = _select_canonical_items(src_cfg, items)
    if not canonical:
        raise RuntimeError(f"No Copernicus DEM EEA-10 tiles found for bbox {bbox}")

    return [_download_and_extract_tile(item, src_cfg, raw_dir) for item in canonical.values()]


def mosaic_copernicus_dem_eea10(cfg: dict, tile_paths: list[Path]) -> Path:
    """Reproject each EPSG:4326 tile onto the project's standard CRS
    (EPSG:28992) and merge into one seamless province-wide mosaic - same
    reproject-then-merge pattern as mosaic_ndwi.py/mosaic_s1.py use for
    Sentinel scenes that arrive in UTM."""
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "copernicus_dem_eea10_mosaic.tif"
    if out_path.exists():
        print(f"Copernicus DEM EEA-10 mosaic already computed, skipping ({out_path})")
        return out_path

    dst_crs = cfg["aoi"]["crs"]
    reproj_dir = processed_dir / "_reprojected"
    reproj_dir.mkdir(parents=True, exist_ok=True)

    reprojected_paths = []
    for p in tile_paths:
        reproj_path = reproj_dir / f"{p.stem}_{dst_crs.replace(':', '')}.tif"
        if not reproj_path.exists():
            print(f"Reprojecting {p.name} -> {dst_crs}")
            with rasterio.open(p) as src:
                nodata = src.nodata
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                profile = src.profile.copy()
                profile.update(crs=dst_crs, transform=transform, width=width, height=height, nodata=nodata)
                with rasterio.open(reproj_path, "w", **profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        src_nodata=nodata,
                        dst_nodata=nodata,
                        resampling=Resampling.bilinear,
                    )
        reprojected_paths.append(reproj_path)

    print(f"Merging {len(reprojected_paths)} reprojected tile(s) into one Copernicus DEM EEA-10 mosaic...")
    with rasterio.open(reprojected_paths[0]) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata
    mosaic, mosaic_transform = rio_merge(reprojected_paths, nodata=nodata)
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    valid = mosaic[0][mosaic[0] != nodata] if nodata is not None else mosaic[0].ravel()
    print(
        f"Copernicus DEM EEA-10 mosaic written: {mosaic.shape[2]}x{mosaic.shape[1]}px, "
        f"elevation range [{valid.min():.1f}, {valid.max():.1f}] m -> {out_path}"
    )
    return out_path


def _overlay_boundary(ax, boundary: gpd.GeoDataFrame, crs) -> None:
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)


def _overlay_plots(ax, plots: gpd.GeoDataFrame, crs) -> None:
    plots.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}"
    )


def visualize_copernicus_dem_eea10(
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
    _overlay_boundary(ax, boundary, crs)
    _overlay_plots(ax, plots, crs)

    ax.set_title("Copernicus DEM EEA-10 - Limburg province")
    add_map_chrome(ax, str(crs))
    add_scale_bar(ax)
    add_north_arrow(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Elevation (m)", rotation=270, labelpad=15)
    ax.legend(loc="upper right", fontsize=7)

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "copernicus_dem_eea10_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)

    tile_paths = fetch_copernicus_dem_eea10_tiles(cfg, boundary)
    mosaic_path = mosaic_copernicus_dem_eea10(cfg, tile_paths)

    with rasterio.open(mosaic_path) as ref:
        nodata = ref.nodata
    clipped_path = clip_raster_to_boundary(
        mosaic_path, boundary, processed_dir / "copernicus_dem_eea10_mosaic_clipped.tif", nodata=nodata
    )

    visualize_copernicus_dem_eea10(cfg, clipped_path, boundary, plots)


if __name__ == "__main__":
    main()
