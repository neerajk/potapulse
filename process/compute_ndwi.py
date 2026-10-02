"""Step 4: NDWI = (green - nir) / (green + nir) per tile (McFeeters 1996),
with cloud/shadow pixels (per the SCL scene classification) masked to NaN
before computing the index. Then filter the NDWI raster down to water-only
pixels (NDWI > ndwi_water_threshold) - everything else becomes NaN.

Two outputs per tile: the continuous NDWI raster (diagnostic) and the
water-filtered raster (feeds mosaic_ndwi.py's final "NDWI filtered mosaic").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from limburg.common import REPO_ROOT, load_config


def _compute_one(
    cfg: dict, item_id: str, band_paths: dict[str, Path], processed_dir: Path
) -> tuple[Path, Path] | None:
    s2cfg = cfg["sentinel2"]
    ndwi_path = processed_dir / f"{item_id}_ndwi.tif"
    water_path = processed_dir / f"{item_id}_ndwi_water.tif"
    if ndwi_path.exists() and water_path.exists():
        print(f"{item_id}: NDWI + water filter already computed, skipping")
        return ndwi_path, water_path

    with rasterio.open(band_paths[s2cfg["green_asset"]]) as g_src:
        green = g_src.read(1).astype("float32")
        profile = g_src.profile.copy()
        transform, width, height, crs = g_src.transform, g_src.width, g_src.height, g_src.crs

    with rasterio.open(band_paths[s2cfg["nir_asset"]]) as n_src:
        nir = n_src.read(1).astype("float32")

    # SCL is natively 20m vs green/nir's 10m - resample (nearest, it's categorical
    # class data) onto the 10m grid before masking.
    with rasterio.open(band_paths[s2cfg["scl_asset"]]) as s_src:
        scl = np.empty((height, width), dtype="uint8")
        reproject(
            source=rasterio.band(s_src, 1),
            destination=scl,
            src_transform=s_src.transform,
            src_crs=s_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)
    ndwi[(green + nir) == 0] = np.nan
    ndwi[np.isin(scl, s2cfg["scl_mask_classes"])] = np.nan

    valid = ndwi[~np.isnan(ndwi)]
    if not valid.size:
        print(f"{item_id}: WARNING - zero valid pixels (fully cloud-masked?), skipping this tile")
        return None

    processed_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(ndwi_path, "w", **profile) as dst:
        dst.write(ndwi.astype("float32"), 1)

    water_only = np.where(ndwi > s2cfg["ndwi_water_threshold"], ndwi, np.nan).astype("float32")
    n_water = int(np.sum(~np.isnan(water_only)))
    with rasterio.open(water_path, "w", **profile) as dst:
        dst.write(water_only, 1)

    print(
        f"{item_id}: NDWI computed, {valid.size}/{ndwi.size} valid pixels "
        f"({valid.size / ndwi.size:.0%}), range [{valid.min():.3f}, {valid.max():.3f}]; "
        f"{n_water} pixels classified as water (NDWI > {s2cfg['ndwi_water_threshold']})"
    )
    return ndwi_path, water_path


def compute_ndwi(cfg: dict, tile_band_paths: list[tuple[str, dict[str, Path]]]) -> list[Path]:
    """Returns the list of water-filtered raster paths (one per tile with any
    valid pixels) - this is what mosaic_ndwi.py merges into the final product."""
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    water_paths = []
    for item_id, band_paths in tile_band_paths:
        result = _compute_one(cfg, item_id, band_paths, processed_dir)
        if result is not None:
            water_paths.append(result[1])

    if not water_paths:
        raise RuntimeError("No tile produced a usable NDWI raster (all fully cloud-masked?)")
    return water_paths


if __name__ == "__main__":
    from limburg.fetch.sentinel2 import fetch_sentinel2_bands
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    tile_band_paths = fetch_sentinel2_bands(cfg, boundary)
    compute_ndwi(cfg, tile_band_paths)
