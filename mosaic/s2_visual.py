"""Step: reproject each per-tile Sentinel-2 true-color ("visual"/TCI) raster
onto the project's standard CRS (EPSG:28992) and merge into one Limburg-wide
true-color mosaic - the plain "S2 mosaic" panel in visualize_overview.py,
alongside the water-filtered NDWI mosaic mosaic_ndwi.py already produces.

Same reproject-then-merge pattern as mosaic_s1.py's RGB composite.
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject

from limburg.common import REPO_ROOT, load_config


def _reproject_visual(src_path: Path, dst_crs: str, out_path: Path) -> Path:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height, nodata=0)

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
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.bilinear,
                )
    return out_path


def mosaic_s2_visual(cfg: dict, tile_band_paths: list[tuple[str, dict[str, Path]]]) -> Path:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "s2_visual_mosaic.tif"
    if out_path.exists():
        print(f"S2 true-color mosaic already computed, skipping ({out_path})")
        return out_path

    dst_crs = cfg["aoi"]["crs"]
    reproj_dir = processed_dir / "_reprojected"
    asset_name = cfg["sentinel2"]["visual_asset"]

    reprojected = []
    for _item_id, band_paths in tile_band_paths:
        src_path = band_paths[asset_name]
        reproj_path = reproj_dir / f"{src_path.stem}_{dst_crs.replace(':', '')}.tif"
        if not reproj_path.exists():
            print(f"Reprojecting {src_path.name} -> {dst_crs}")
            _reproject_visual(src_path, dst_crs, reproj_path)
        reprojected.append(reproj_path)

    print(f"Merging {len(reprojected)} S2 true-color tile(s) into one mosaic...")
    mosaic, mosaic_transform = rio_merge(reprojected, nodata=0)
    with rasterio.open(reprojected[0]) as ref:
        profile = ref.profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    from limburg.fetch.sentinel2 import fetch_sentinel2_bands
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    tile_band_paths = fetch_sentinel2_bands(cfg, boundary)
    mosaic_s2_visual(cfg, tile_band_paths)
