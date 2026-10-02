"""Step 12: a second overview figure, one panel per input data layer -
  (1) Sentinel-2 true-color mosaic (TCI)
  (2) Sentinel-2 NDWI filtered mosaic (water pixels only)
  (3) Sentinel-1 SAR surface-water mask (filtered)
  (4) AHN4 DTM mosaic
each clipped to the exact Limburg province polygon and overlaid with the
seed potato plots. Complements visualize_limburg.py's 3-panel SAR-focused
figure with a plain 4-layer data overview (no SAR/plot overlap highlighting
here - just what each source looks like over the province).

Reuses br_simulation.visualization.style (same chrome as every other figure
in this project) so this sits visually as one system.
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


def _overlay_plots(ax, plots: gpd.GeoDataFrame, crs) -> None:
    plots.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}"
    )


def _plot_rgb(ax, path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, title: str) -> None:
    with rasterio.open(path) as src:
        rgb = src.read()[:3]  # TCI is 3-band; drop any extra band defensively
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    alpha = np.where(np.any(rgb > 0, axis=0), 255, 0).astype("uint8")
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
    ax.imshow(rgba, extent=extent, zorder=1)
    _overlay_boundary(ax, boundary, crs)
    _overlay_plots(ax, plots, crs)
    ax.set_title(title)
    add_map_chrome(ax, str(crs))
    ax.legend(loc="upper right", fontsize=7)


def _plot_scalar(
    ax,
    path: Path,
    boundary: gpd.GeoDataFrame,
    plots: gpd.GeoDataFrame,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, zorder=1)
    _overlay_boundary(ax, boundary, crs)
    _overlay_plots(ax, plots, crs)
    ax.set_title(title)
    add_map_chrome(ax, str(crs))
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    ax.legend(loc="upper right", fontsize=7)


def visualize_overview(
    cfg: dict,
    s2_visual_path: Path,
    ndwi_water_path: Path,
    s1_water_path: Path,
    dtm_path: Path,
    plots: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 20))

    _plot_rgb(axes[0, 0], s2_visual_path, boundary, plots, "Sentinel-2 true-color mosaic (TCI)")
    _plot_scalar(
        axes[0, 1], ndwi_water_path, boundary, plots,
        "Sentinel-2 NDWI filtered mosaic (water pixels only)",
        "Blues", "NDWI (water pixels only)", vmin=0, vmax=0.6,
    )
    _plot_scalar(
        axes[1, 0], s1_water_path, boundary, plots,
        "Sentinel-1 SAR surface-water mask (filtered)",
        "Blues", "SAR water (1=water)", vmin=0, vmax=1,
    )

    with rasterio.open(dtm_path) as src:
        dtm_sample, dtm_nodata = src.read(1), src.nodata
    valid_dtm = dtm_sample[dtm_sample != dtm_nodata] if dtm_nodata is not None else dtm_sample
    _plot_scalar(
        axes[1, 1], dtm_path, boundary, plots,
        "AHN4 DTM mosaic", "terrain", "Elevation (m, NAP)",
        vmin=float(valid_dtm.min()), vmax=float(valid_dtm.max()),
    )

    fig.suptitle(
        f"Limburg province - data overview | {len(plots)} seed potato plots",
        fontsize=14, fontweight="bold",
    )
    for ax in axes.flat:
        add_scale_bar(ax)
        add_north_arrow(ax)

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "limburg_full_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    s2_visual_path = processed_dir / "s2_visual_mosaic_clipped.tif"
    ndwi_water_path = processed_dir / "ndwi_water_mosaic_clipped.tif"
    s1_water_path = processed_dir / "s1_water_mosaic_clipped.tif"
    dtm_path = processed_dir / "dtm_mosaic_clipped.tif"
    boundary = gpd.read_file(processed_dir / "province_boundary.gpkg")
    plots = gpd.read_file(processed_dir / "seed_potato_plots.gpkg")
    visualize_overview(cfg, s2_visual_path, ndwi_water_path, s1_water_path, dtm_path, plots, boundary)
