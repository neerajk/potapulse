"""Step 11: final 3-panel figure, each panel clipped to the Limburg province
boundary and overlaid with seed potato plots -
  (1) Sentinel-1 RGB mosaic (R=VV, G=VH, B=VV-VH ratio, all dB)
  (2) Sentinel-2 NDWI filtered mosaic (water pixels only)
  (3) SAR-derived water mask, plots that intersect SAR water highlighted red.

Reuses br_simulation.visualization.style (same chrome as the main pipeline's
figures and NDWI/visualize_ndwi.py) so this sits visually as one system.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.common import REPO_ROOT, load_config

apply_style()


def _overlay_boundary(ax, boundary: gpd.GeoDataFrame, crs) -> None:
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)


def _overlay_plots(ax, plots: gpd.GeoDataFrame, crs, label: str) -> None:
    plots.to_crs(crs).boundary.plot(ax=ax, color="black", linewidth=0.3, zorder=3, label=label)


def _plot_s1_rgb(ax, rgb_path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame) -> None:
    with rasterio.open(rgb_path) as src:
        rgb = src.read()  # (3, H, W), uint8, nodata=0
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    alpha = np.where(np.any(rgb > 0, axis=0), 255, 0).astype("uint8")
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
    ax.imshow(rgba, extent=extent, zorder=1)
    _overlay_boundary(ax, boundary, crs)
    _overlay_plots(ax, plots, crs, f"Seed potato plots, n={len(plots)}")
    ax.set_title("Sentinel-1 SAR mosaic (R=VV, G=VH, B=VV-VH, dB)")
    add_map_chrome(ax, str(crs))
    ax.legend(loc="upper right", fontsize=7)


def _plot_ndwi_water(ax, ndwi_water_path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame) -> float:
    with rasterio.open(ndwi_water_path) as src:
        ndwi = src.read(1)
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    im = ax.imshow(ndwi, cmap="Blues", vmin=0, vmax=0.6, extent=extent, zorder=1)
    _overlay_boundary(ax, boundary, crs)
    _overlay_plots(ax, plots, crs, f"Seed potato plots, n={len(plots)}")
    valid = ndwi[~np.isnan(ndwi)]
    pct_water = valid.size / ndwi.size
    ax.set_title(f"Sentinel-2 NDWI filtered mosaic (water pixels only, {pct_water:.1%} of extent)")
    add_map_chrome(ax, str(crs))
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("NDWI (water pixels only)", rotation=270, labelpad=15)
    ax.legend(loc="upper right", fontsize=7)
    return pct_water


def _plot_sar_water_with_plots(
    ax, water_mosaic_path: Path, boundary: gpd.GeoDataFrame, plots_overlap: gpd.GeoDataFrame
) -> None:
    with rasterio.open(water_mosaic_path) as src:
        water = src.read(1).astype("float32")
        extent = raster_extent(src.transform, src.width, src.height)
        raster_crs = src.crs
    water[water == 255] = np.nan

    ax.imshow(water, cmap="Blues", vmin=0, vmax=1, extent=extent, zorder=1)
    _overlay_boundary(ax, boundary, raster_crs)

    plots_proj = plots_overlap.to_crs(raster_crs)
    non_overlap = plots_proj[~plots_proj["overlaps_sar_water"]]
    overlap = plots_proj[plots_proj["overlaps_sar_water"]]

    non_overlap.boundary.plot(ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(non_overlap)}")
    if len(overlap):
        overlap.plot(ax=ax, facecolor="red", edgecolor="darkred", linewidth=0.5, alpha=0.7, zorder=4, label=f"Overlaps SAR water, n={len(overlap)}")

    ax.set_title("SAR surface-water mask + seed potato plot overlap")
    add_map_chrome(ax, str(raster_crs))
    ax.legend(loc="upper right", fontsize=7)


def visualize_limburg(
    cfg: dict,
    s1_rgb_path: Path,
    ndwi_water_path: Path,
    s1_water_path: Path,
    plots_overlap: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    _plot_s1_rgb(axes[0], s1_rgb_path, boundary, plots_overlap)
    _plot_ndwi_water(axes[1], ndwi_water_path, boundary, plots_overlap)
    _plot_sar_water_with_plots(axes[2], s1_water_path, boundary, plots_overlap)

    n_overlap = int(plots_overlap["overlaps_sar_water"].sum())
    fig.suptitle(
        f"Limburg province - Aug-Sep 2025 | {len(plots_overlap)} seed potato plots, "
        f"{n_overlap} ({n_overlap / len(plots_overlap):.1%}) overlap SAR-detected surface water",
        fontsize=13,
        fontweight="bold",
    )
    for ax in axes:
        add_scale_bar(ax)
        add_north_arrow(ax)

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "limburg_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    s1_rgb_path = processed_dir / "s1_rgb_mosaic_clipped.tif"
    ndwi_water_path = processed_dir / "ndwi_water_mosaic_clipped.tif"
    s1_water_path = processed_dir / "s1_water_mosaic_clipped.tif"
    boundary = gpd.read_file(processed_dir / "province_boundary.gpkg")
    plots_overlap = gpd.read_file(processed_dir / "seed_potato_plots_overlap.gpkg")
    visualize_limburg(cfg, s1_rgb_path, ndwi_water_path, s1_water_path, plots_overlap, boundary)
