"""Standalone script: fetch an AHN4 DTM mosaic for the entire Netherlands
(not just Limburg) via PDOK's AHN WCS, and plot a single-panel overview
figure. Independent of run_limburg.py's province pipeline - own tile cache,
own mosaic, own figure - but reuses the same WCS endpoint/config block
(data_sources.ahn4_dtm in config/limburg_config.yaml) and the same generic
tiling/fetch helpers as fetch_ahn_dtm.py, since those aren't province-specific.

Key difference from fetch_ahn_dtm.py: there's no province/country polygon to
fetch or clip against here. The AHN WCS only ever publishes data over Dutch
territory, so its own declared coverage_bounds_28992 (confirmed live
2026-10-02 via DescribeCoverage) already IS the Netherlands' extent - any bbox
corner outside actual land (North Sea, a sliver of Belgium/Germany near the
borders) simply comes back as nodata from the WCS itself. The visualization
step masks those nodata pixels transparent directly, with no separate
boundary fetch needed.

Resolution is coarser than the per-province script
(data_sources.ahn4_dtm.netherlands_overview_resolution_m, default 50m/px vs.
10m/px for Limburg alone) to keep the merged mosaic a manageable size for a
single overview figure - see the config comment for the arithmetic.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.fetch.ahn4_dtm import MAXSIZE_PX, SAFETY_MARGIN, _fetch_tile, _tile_grid
from limburg.common import REPO_ROOT, load_config

apply_style()


def fetch_ahn_dtm_netherlands_tiles(cfg: dict) -> list[Path]:
    """Fetch every AHN4 DTM tile covering the WCS's full declared coverage
    extent, downsampled to data_sources.ahn4_dtm.netherlands_overview_resolution_m.
    Returns the fetched tile paths - mosaic_ahn_dtm_netherlands merges these
    into one country-wide raster."""
    src_cfg = cfg["data_sources"]["ahn4_dtm"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"] / "ahn4_netherlands"
    target_res_m = src_cfg["netherlands_overview_resolution_m"]
    tile_size_m = math.floor(MAXSIZE_PX * target_res_m * SAFETY_MARGIN)
    coverage_bounds = tuple(src_cfg["coverage_bounds_28992"])

    tile_paths: list[Path] = []
    for minx, miny, maxx, maxy in _tile_grid(coverage_bounds, tile_size_m):
        tile_id = f"dtm_nl_{int(minx)}_{int(miny)}"
        out_path = raw_dir / f"{tile_id}.tif"
        if out_path.exists():
            print(f"{tile_id} already fetched, skipping ({out_path})")
            tile_paths.append(out_path)
            continue

        print(
            f"Fetching AHN4 DTM tile {tile_id} "
            f"({maxx - minx:.0f}m x {maxy - miny:.0f}m @ {target_res_m}m/px)"
        )
        _fetch_tile(src_cfg, (minx, miny, maxx, maxy), target_res_m, out_path)
        tile_paths.append(out_path)

    if not tile_paths:
        raise RuntimeError("No AHN4 DTM tiles were fetched for the Netherlands coverage bbox")
    return tile_paths


def mosaic_ahn_dtm_netherlands(cfg: dict, tile_paths: list[Path]) -> Path:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "dtm_netherlands_mosaic.tif"
    if out_path.exists():
        print(f"Netherlands DTM mosaic already computed, skipping ({out_path})")
        return out_path

    with rasterio.open(tile_paths[0]) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata

    print(f"Merging {len(tile_paths)} AHN4 DTM tile(s) into one Netherlands-wide mosaic...")
    mosaic, mosaic_transform = rio_merge(tile_paths, nodata=nodata)
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    valid = mosaic[0][mosaic[0] != nodata]
    print(
        f"Netherlands DTM mosaic written: {mosaic.shape[2]}x{mosaic.shape[1]}px, "
        f"elevation range [{valid.min():.1f}, {valid.max():.1f}] m NAP -> {out_path}"
    )
    return out_path


def visualize_netherlands_dtm(cfg: dict, dtm_path: Path) -> Path:
    """Single-panel overview. No boundary overlay - nodata pixels (sea, and
    any bbox corner outside actual AHN coverage) are masked transparent
    directly via imshow's masked-array support, which already gives a clean
    coastline without a separate country-polygon fetch."""
    with rasterio.open(dtm_path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    is_nodata = np.isnan(data) if (nodata is not None and np.isnan(nodata)) else (data == nodata)
    masked = np.ma.masked_where(is_nodata, data)
    valid = data[~is_nodata]

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=float(valid.min()), vmax=float(valid.max()), extent=extent, zorder=1)

    ax.set_title("AHN4 DTM - entire Netherlands")
    add_map_chrome(ax, str(crs))
    add_scale_bar(ax)
    add_north_arrow(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Elevation (m, NAP)", rotation=270, labelpad=15)

    fig.suptitle("PDOK AHN4 dtm_05m, whole-country mosaic", fontsize=11)

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "netherlands_dtm_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    cfg = load_config()

    tile_paths = fetch_ahn_dtm_netherlands_tiles(cfg)
    mosaic_path = mosaic_ahn_dtm_netherlands(cfg, tile_paths)
    visualize_netherlands_dtm(cfg, mosaic_path)


if __name__ == "__main__":
    main()
