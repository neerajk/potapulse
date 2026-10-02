"""Take limburg/standalone/download_limburg_dtm.py's 5m AHN DTM, downsample
it to 10m/px, "burn" the OSM waterway network (limburg.fetch.waterways'
province-clipped lines) into it as fixed-depth ditches, clip to the exact
Limburg province polygon, and plot it side by side with the existing
Sentinel-2 true-color mosaic - both panels with the seed potato plots
overlaid.

Ditch burning: coarsening a 5m DEM to 10m by simple averaging can smooth a
narrow farm ditch (often only 1-2 native pixels wide) into its surrounding
land, breaking drainage connectivity that matters for any downstream
surface-water-flow analysis. This rasterizes the OSM waterway lines onto the
10m grid and subtracts dtm_burn.burn_depth_m from the DEM at those pixels -
a simplified, fixed-depth version of the AGREE method (Hellweger 1997)
standard in DEM-hydrology preprocessing. It is a visualization/preprocessing
aid, not a surveyed bathymetry correction.

Depends on two things already having been run:
  - python limburg/standalone/download_limburg_dtm.py --province Limburg --res 5  (the source DTM)
  - python -m limburg.run_limburg                                                 (the S2 mosaic)
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.process.clip_to_boundary import clip_raster_to_boundary
from limburg.fetch.waterways import fetch_waterways_clipped
from limburg.common import REPO_ROOT, load_config
from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots

apply_style()


def _downsample(src_path: Path, target_res_m: float, out_path: Path) -> Path:
    if out_path.exists():
        print(f"Downsampled DTM already computed, skipping ({out_path})")
        return out_path

    with rasterio.open(src_path) as src:
        scale_x = src.res[0] / target_res_m
        scale_y = src.res[1] / target_res_m
        new_width = max(1, round(src.width * scale_x))
        new_height = max(1, round(src.height * scale_y))
        transform = src.transform * src.transform.scale(src.width / new_width, src.height / new_height)
        profile = src.profile.copy()
        profile.update(width=new_width, height=new_height, transform=transform)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
                resampling=Resampling.average,
            )
    print(f"Downsampled {src_path.name} -> {target_res_m:g}m/px ({new_width}x{new_height}) -> {out_path}")
    return out_path


def _burn_waterways(dtm_path: Path, waterways: gpd.GeoDataFrame, burn_depth_m: float, out_path: Path) -> Path:
    if out_path.exists():
        print(f"Ditch-burned DTM already computed, skipping ({out_path})")
        return out_path

    with rasterio.open(dtm_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        shape = (src.height, src.width)
        profile = src.profile.copy()
        crs = src.crs

    geoms = [g for g in waterways.to_crs(crs).geometry if g is not None and not g.is_empty]
    waterway_mask = rasterize(
        [(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8"
    ).astype(bool)

    is_nodata = np.isnan(data) if (nodata is not None and np.isnan(nodata)) else (data == nodata)
    burn_px = waterway_mask & ~is_nodata
    burned = data.copy()
    burned[burn_px] = burned[burn_px] - burn_depth_m
    print(f"Burned {int(burn_px.sum())} waterway pixel(s) by {burn_depth_m}m")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(burned, 1)
    return out_path


def prepare_burned_dtm(cfg: dict, boundary: gpd.GeoDataFrame, waterways_clipped: gpd.GeoDataFrame) -> Path:
    src_cfg = cfg["dtm_burn"]
    source_path = REPO_ROOT / src_cfg["source_dtm_path"]
    if not source_path.exists():
        raise RuntimeError(
            f"{source_path} not found - run download_limburg_dtm.py first "
            f"(python limburg/standalone/download_limburg_dtm.py --province Limburg --res 5)."
        )

    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    res = src_cfg["target_resolution_m"]
    downsampled_path = _downsample(source_path, res, processed_dir / f"limburg_ahn_dtm_{res:g}m.tif")
    burned_path = _burn_waterways(
        downsampled_path, waterways_clipped, src_cfg["burn_depth_m"], processed_dir / f"limburg_ahn_dtm_{res:g}m_burned.tif"
    )

    with rasterio.open(burned_path) as ref:
        nodata = ref.nodata
    return clip_raster_to_boundary(
        burned_path, boundary, processed_dir / f"limburg_ahn_dtm_{res:g}m_burned_clipped.tif", nodata=nodata
    )


def _plot_s2_mosaic(ax, s2_path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame) -> None:
    with rasterio.open(s2_path) as src:
        rgb = src.read()[:3]
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    alpha = np.where(np.any(rgb > 0, axis=0), 255, 0).astype("uint8")
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
    ax.imshow(rgba, extent=extent, zorder=1)
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    plots.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}"
    )
    ax.set_title("Sentinel-2 true-color mosaic")
    ax.legend(loc="upper right", fontsize=7)


def _plot_burned_dtm(ax, dtm_path: Path, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, burn_depth_m: float) -> None:
    with rasterio.open(dtm_path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata
        extent = raster_extent(src.transform, src.width, src.height)
        crs = src.crs
    is_nodata = np.isnan(data) if (nodata is not None and np.isnan(nodata)) else (data == nodata)
    masked = np.ma.masked_where(is_nodata, data)
    valid = data[~is_nodata]

    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=float(valid.min()), vmax=float(valid.max()), extent=extent, zorder=1)
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    plots.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=0.3, zorder=3, label=f"Seed potato plots, n={len(plots)}"
    )
    ax.set_title(f"AHN DTM, 10m/px, OSM ditches burned -{burn_depth_m:g}m")
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Elevation (m NAP)", rotation=270, labelpad=15)
    ax.legend(loc="upper right", fontsize=7)


def visualize(
    cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, s2_path: Path, burned_dtm_path: Path
) -> Path:
    crs = cfg["aoi"]["crs"]
    burn_depth_m = cfg["dtm_burn"]["burn_depth_m"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    _plot_s2_mosaic(axes[0], s2_path, boundary, plots)
    _plot_burned_dtm(axes[1], burned_dtm_path, boundary, plots, burn_depth_m)
    for ax in axes:
        add_map_chrome(ax, crs)
        add_scale_bar(ax)
        add_north_arrow(ax)

    fig.suptitle("Limburg province - S2 mosaic vs. ditch-burned AHN DTM (10m)", fontsize=13, fontweight="bold")

    out_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s2_mosaic_and_burned_dtm_overview.png"
    fig.savefig(out_path, dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)
    waterways_clipped = fetch_waterways_clipped(cfg, boundary)

    burned_dtm_path = prepare_burned_dtm(cfg, boundary, waterways_clipped)

    s2_path = processed_dir / "s2_visual_mosaic_clipped.tif"
    if not s2_path.exists():
        raise RuntimeError(f"{s2_path} not found - run `python -m limburg.run_limburg` first to produce the Sentinel-2 mosaic.")

    visualize(cfg, boundary, plots, s2_path, burned_dtm_path)


if __name__ == "__main__":
    main()
