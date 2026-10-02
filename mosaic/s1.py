"""Step 9: reproject each per-scene Sentinel-1 RGB composite and water mask
onto the project's standard CRS (EPSG:28992) and merge into two province-wide
mosaics - the "S1 mosaic RGB" and the SAR water-mask mosaic (merged with
method="max" so a water classification at any overlapping scene wins).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject

from limburg.common import REPO_ROOT, load_config


def _reproject_raster(
    src_path: Path, dst_crs: str, out_path: Path, nodata, resampling: Resampling
) -> Path:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height, nodata=nodata)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                    resampling=resampling,
                )
    return out_path


def mosaic_s1(cfg: dict, s1_outputs: list[tuple[Path, Path]]) -> tuple[Path, Path]:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    rgb_mosaic_path = processed_dir / "s1_rgb_mosaic.tif"
    water_mosaic_path = processed_dir / "s1_water_mosaic.tif"
    if rgb_mosaic_path.exists() and water_mosaic_path.exists():
        print("S1 mosaics already computed, skipping")
        return rgb_mosaic_path, water_mosaic_path

    dst_crs = cfg["aoi"]["crs"]
    reproj_dir = processed_dir / "_reprojected"

    rgb_reproj, water_reproj = [], []
    for rgb_path, water_path in s1_outputs:
        rgb_out = reproj_dir / f"{rgb_path.stem}_{dst_crs.replace(':', '')}.tif"
        if not rgb_out.exists():
            print(f"Reprojecting {rgb_path.name} -> {dst_crs}")
            _reproject_raster(rgb_path, dst_crs, rgb_out, nodata=0, resampling=Resampling.bilinear)
        rgb_reproj.append(rgb_out)

        water_out = reproj_dir / f"{water_path.stem}_{dst_crs.replace(':', '')}.tif"
        if not water_out.exists():
            print(f"Reprojecting {water_path.name} -> {dst_crs}")
            _reproject_raster(water_path, dst_crs, water_out, nodata=255, resampling=Resampling.nearest)
        water_reproj.append(water_out)

    print(f"Merging {len(rgb_reproj)} S1 RGB scene(s) into one mosaic...")
    rgb_mosaic, rgb_transform = rio_merge(rgb_reproj, nodata=0)
    with rasterio.open(rgb_reproj[0]) as ref:
        rgb_profile = ref.profile.copy()
    rgb_profile.update(height=rgb_mosaic.shape[1], width=rgb_mosaic.shape[2], transform=rgb_transform)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(rgb_mosaic_path, "w", **rgb_profile) as dst:
        dst.write(rgb_mosaic)
    print(f"Wrote {rgb_mosaic_path}")

    print(f"Merging {len(water_reproj)} S1 water-mask scene(s) into one mosaic (method=max)...")
    water_mosaic, water_transform = rio_merge(water_reproj, nodata=255, method="max")
    with rasterio.open(water_reproj[0]) as ref:
        water_profile = ref.profile.copy()
    water_profile.update(
        height=water_mosaic.shape[1], width=water_mosaic.shape[2], transform=water_transform
    )
    with rasterio.open(water_mosaic_path, "w", **water_profile) as dst:
        dst.write(water_mosaic[0], 1)

    valid = water_mosaic[0][water_mosaic[0] != 255]
    n_water = int(np.sum(valid == 1))
    print(
        f"Wrote {water_mosaic_path} - {n_water}/{valid.size} valid pixels "
        f"({n_water / valid.size:.1%}) classified as SAR water"
    )
    return rgb_mosaic_path, water_mosaic_path


if __name__ == "__main__":
    from limburg.process.compute_s1_water import compute_s1_water
    from limburg.fetch.sentinel1 import fetch_sentinel1_bands
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    s1_items = fetch_sentinel1_bands(cfg, boundary)
    s1_outputs = compute_s1_water(cfg, s1_items)
    mosaic_s1(cfg, s1_outputs)
