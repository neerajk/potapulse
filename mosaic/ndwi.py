"""Step 5: reproject each per-tile water-filtered NDWI raster onto the
project's standard CRS (EPSG:28992) and merge into one seamless province-wide
"NDWI filtered mosaic" (water pixels only, everything else NaN/transparent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject

from limburg.common import REPO_ROOT, load_config


def _reproject_tile(src_path: Path, dst_crs: str, out_path: Path) -> Path:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height, nodata=np.nan)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return out_path


def mosaic_ndwi(cfg: dict, water_ndwi_paths: list[Path]) -> Path:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "ndwi_water_mosaic.tif"
    if out_path.exists():
        print(f"NDWI filtered mosaic already computed, skipping ({out_path})")
        return out_path

    dst_crs = cfg["aoi"]["crs"]
    reproj_dir = processed_dir / "_reprojected"
    reprojected_paths = []
    for p in water_ndwi_paths:
        reproj_path = reproj_dir / f"{p.stem}_{dst_crs.replace(':', '')}.tif"
        if not reproj_path.exists():
            print(f"Reprojecting {p.name} -> {dst_crs}")
            _reproject_tile(p, dst_crs, reproj_path)
        reprojected_paths.append(reproj_path)

    print(f"Merging {len(reprojected_paths)} reprojected tile(s) into one NDWI water mosaic...")
    mosaic, mosaic_transform = rio_merge(reprojected_paths, nodata=np.nan)

    with rasterio.open(reprojected_paths[0]) as ref:
        profile = ref.profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    valid = mosaic[0][~np.isnan(mosaic[0])]
    print(
        f"NDWI water mosaic written: {mosaic.shape[2]}x{mosaic.shape[1]}px, "
        f"{valid.size}/{mosaic[0].size} water pixels ({valid.size / mosaic[0].size:.1%}) -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    from limburg.process.compute_ndwi import compute_ndwi
    from limburg.fetch.sentinel2 import fetch_sentinel2_bands
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    tile_band_paths = fetch_sentinel2_bands(cfg, boundary)
    water_paths = compute_ndwi(cfg, tile_band_paths)
    mosaic_ndwi(cfg, water_paths)
