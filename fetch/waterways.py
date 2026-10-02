"""Standalone script: fetch OpenStreetMap waterway lines for the Netherlands
(HOTOSM/HDX export), clip to the exact Limburg province polygon, and filter
to the subset within a configured buffer distance of a seed potato plot.
Plots a side-by-side overview: left panel is every clipped waterway colored
by its OSM waterway=* classification (with a legend), right panel is the
buffer-filtered subset against the seed potato plots. Independent of
run_limburg.py's AHN4 pipeline - reuses the cached province boundary / seed
potato plots and this directory's plotting conventions, but is otherwise
self-contained.

Source: HDX dataset "hotosm_nld_waterways" (Humanitarian OpenStreetMap Team,
ODbL) - confirmed live 2026-10-02 via HDX's CKAN API
(https://data.humdata.org/api/3/action/package_show?id=hotosm_nld_waterways)
that its SHP resource is a public, no-auth S3 object containing three
layers: waterways_points, waterways_lines, waterways_polygons. This script
uses only waterways_lines (235,098 features nationwide, confirmed via the
zip's own DBF header), since that's the layer carrying OSM's waterway=* tag
(river/stream/canal/ditch/drain/...) - waterways_polygons is water *bodies*
(lakes/ponds; water=*/natural=water), a different thing not fetched here.

The whole-NL zip (~195MB) is downloaded and cached once. Rather than
extracting it, the lines layer is read directly from inside the zip via
GDAL's /vsizip/ virtual filesystem with a bbox pushed down to the province's
bounding box (same "never materialize the multi-GB national file" approach
seed_plots.py uses for the BRP GeoPackage) - avoids ever loading the other
two layers, or waterway lines outside the province's bbox, into memory.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style
from limburg.common import REPO_ROOT, load_config
from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots

HTTP_TIMEOUT_S = 60
DOWNLOAD_TIMEOUT_S = 1800
DOWNLOAD_CHUNK_BYTES = 1 << 20

apply_style()


def _download_shp_zip(src_cfg: dict, raw_dir: Path) -> Path:
    dest = raw_dir / "hotosm_nld_waterways_osm_shp.zip"
    expected = src_cfg["shp_zip_size_bytes"]
    if dest.exists() and dest.stat().st_size == expected:
        print(f"HOTOSM waterways shapefile already downloaded, skipping ({dest})")
        return dest

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading HOTOSM Netherlands waterways shapefile (~{expected / 1e6:.0f}MB) from {src_cfg['shp_zip_url']}")
    with requests.get(src_cfg["shp_zip_url"], stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or expected
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="hotosm_nld_waterways"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if tmp.stat().st_size != expected:
        raise RuntimeError(
            f"Downloaded size {tmp.stat().st_size} != expected {expected} for {src_cfg['shp_zip_url']}"
        )
    tmp.rename(dest)
    return dest


def fetch_waterways_clipped(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Every waterway line clipped to the exact Limburg province polygon
    (no plot-proximity filtering yet - that's filter_waterways_near_plots).
    Keeps the "waterway" tag column (river/stream/canal/ditch/...) for the
    classified overview panel."""
    src_cfg = cfg["waterways"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    cache_path = processed_dir / "waterways_clipped.gpkg"
    if cache_path.exists():
        print(f"Loading cached province-clipped waterways from {cache_path}")
        return gpd.read_file(cache_path)

    zip_path = _download_shp_zip(src_cfg, raw_dir)

    bbox = tuple(boundary.to_crs(src_cfg["source_crs"]).total_bounds)
    vsi_path = f"/vsizip/{zip_path}/{src_cfg['lines_layer']}.shp"
    print(f"Reading {src_cfg['lines_layer']} from {zip_path.name}, bbox-filtered to {bbox}")
    lines = gpd.read_file(vsi_path, bbox=bbox, where="waterway IS NOT NULL", engine="pyogrio")
    if lines.crs is None:
        lines = lines.set_crs(src_cfg["source_crs"])
    print(f"{len(lines)} waterway lines intersect the province bbox (pre-clip)")

    lines = lines.to_crs(cfg["aoi"]["crs"])
    lines = gpd.clip(lines, boundary)
    lines = lines[~lines.geometry.is_empty & lines.geometry.notna()].reset_index(drop=True)
    print(f"{len(lines)} waterway lines after clipping to the exact province polygon")

    processed_dir.mkdir(parents=True, exist_ok=True)
    lines.to_file(cache_path, driver="GPKG")
    print(f"Wrote {cache_path}")
    return lines


def filter_waterways_near_plots(
    cfg: dict, waterways_clipped: gpd.GeoDataFrame, plots: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Subset of the already province-clipped waterways within
    waterways.buffer_distance_m of any seed potato plot."""
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    cache_path = processed_dir / "waterways_near_plots.gpkg"
    if cache_path.exists():
        print(f"Loading cached waterways-near-plots from {cache_path}")
        return gpd.read_file(cache_path)

    buffer_m = cfg["waterways"]["buffer_distance_m"]
    buffer_union = plots.geometry.buffer(buffer_m).union_all()
    near = waterways_clipped[waterways_clipped.intersects(buffer_union)].reset_index(drop=True)
    print(f"{len(near)} waterway lines within {buffer_m}m of a seed potato plot")

    processed_dir.mkdir(parents=True, exist_ok=True)
    near.to_file(cache_path, driver="GPKG")
    print(f"Wrote {cache_path}")
    return near


def _plot_waterways_classified(
    ax, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, waterways_clipped: gpd.GeoDataFrame
) -> None:
    """Left panel: every clipped waterway, colored by its OSM waterway=* tag
    value. Categories (and their colors) are derived from whatever's
    actually present in this province's data rather than a hardcoded OSM
    vocabulary list, since that varies by region/mapper coverage."""
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=1)
    plots.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6, zorder=2)

    categories = waterways_clipped["waterway"].fillna("unknown")
    counts = categories.value_counts()
    cmap = plt.get_cmap("tab10")
    for i, category in enumerate(counts.index):
        subset = waterways_clipped[categories == category]
        subset.plot(ax=ax, color=cmap(i % 10), linewidth=1.0, zorder=3, label=f"{category} (n={counts[category]})")

    ax.set_title(f"All waterways, classified - Limburg province (n={len(waterways_clipped)})")
    ax.legend(loc="upper right", fontsize=7, title="waterway=*")


def _plot_waterways_near_plots(
    ax, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, waterways_near: gpd.GeoDataFrame, buffer_m: float
) -> None:
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=1)
    plots.geometry.buffer(buffer_m).boundary.plot(
        ax=ax, color="#1f77b4", linewidth=0.4, linestyle="--", alpha=0.6, zorder=2, label=f"{buffer_m}m buffer"
    )
    plots.boundary.plot(ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}")
    waterways_near.plot(
        ax=ax, color="#0b5fa5", linewidth=1.2, zorder=4, label=f"Waterways within {buffer_m}m, n={len(waterways_near)}"
    )

    ax.set_title(f"Waterways within {buffer_m}m of seed potato plots")
    ax.legend(loc="upper right", fontsize=7)


def visualize_waterways(
    cfg: dict,
    boundary: gpd.GeoDataFrame,
    plots: gpd.GeoDataFrame,
    waterways_clipped: gpd.GeoDataFrame,
    waterways_near: gpd.GeoDataFrame,
) -> Path:
    buffer_m = cfg["waterways"]["buffer_distance_m"]
    crs = cfg["aoi"]["crs"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    _plot_waterways_classified(axes[0], boundary, plots, waterways_clipped)
    _plot_waterways_near_plots(axes[1], boundary, plots, waterways_near, buffer_m)

    for ax in axes:
        add_map_chrome(ax, crs)
        add_scale_bar(ax)
        add_north_arrow(ax)

    fig.suptitle("Limburg province - OSM waterways (HOTOSM/HDX)", fontsize=13, fontweight="bold")

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "waterways_near_plots_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    cfg = load_config()

    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)
    waterways_clipped = fetch_waterways_clipped(cfg, boundary)
    waterways_near = filter_waterways_near_plots(cfg, waterways_clipped, plots)
    visualize_waterways(cfg, boundary, plots, waterways_clipped, waterways_near)


if __name__ == "__main__":
    main()
