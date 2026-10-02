"""Step 8: per Sentinel-1 scene - calibrate VV/VH to dB (calibrate_s1.py),
build a 3-band RGB composite for the "S1 mosaic RGB" visualization
(R=VV, G=VH, B=VV-VH ratio in dB, each independently percentile-stretched to
uint8 - the standard dual-pol SAR false-color convention, where water reads
dark in all three channels and double-bounce/urban reads bright red/magenta),
despeckle VV (median filter - SAR backscatter has strong multiplicative
speckle noise from coherent imaging, which otherwise makes Otsu thresholding
classify huge numbers of isolated noise pixels as "water"), and binarize a
surface-water mask via Otsu's (1979) automatic threshold on the despeckled VV
- water's specular backscatter is markedly lower than vegetated or bare land,
producing the bimodal histogram Otsu's method assumes. This VV-threshold
approach to Sentinel-1 water/flood mapping is the same one documented in
Martinis et al. (2015, Remote Sensing 7(1)), "A Fully Automated TerraSAR-X
Based Flood Service," generalized to Sentinel-1 VV in subsequent Copernicus
Emergency Management Service SAR flood-mapping guidance. Binary opening/
closing after thresholding removes remaining isolated noise pixels and fills
small holes in real water bodies - standard post-processing in that same
family of pipelines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_closing, binary_opening, median_filter
from skimage.filters import threshold_otsu

from limburg.process.calibrate_s1 import calibrate_to_db
from limburg.common import REPO_ROOT, load_config


def _percentile_stretch_to_uint8(arr: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    valid = arr[~np.isnan(arr)]
    if not valid.size:
        return np.zeros(arr.shape, dtype="uint8")
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    stretched = np.nan_to_num(stretched, nan=0.0)
    return (stretched * 255).astype("uint8")


def _compute_one(cfg: dict, item_id: str, info: dict, processed_dir: Path) -> tuple[Path, Path] | None:
    rgb_path = processed_dir / f"{item_id}_s1_rgb.tif"
    water_path = processed_dir / f"{item_id}_s1_water.tif"
    if rgb_path.exists() and water_path.exists():
        print(f"{item_id}: S1 RGB + water mask already computed, skipping")
        return rgb_path, water_path

    with rasterio.open(info["vv"]) as src:
        vv_dn = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(info["vh"]) as src:
        vh_dn = src.read(1)

    vv_db = calibrate_to_db(vv_dn, info["calibration_vv"], info["row_off"], info["col_off"])
    vh_db = calibrate_to_db(vh_dn, info["calibration_vh"], info["row_off"], info["col_off"])

    valid_mask = ~np.isnan(vv_db)
    valid_vv = vv_db[valid_mask]
    if valid_vv.size < 100:
        print(f"{item_id}: WARNING - fewer than 100 valid VV pixels after calibration, skipping")
        return None

    ratio_db = vv_db - vh_db
    r = _percentile_stretch_to_uint8(vv_db)
    g = _percentile_stretch_to_uint8(vh_db)
    b = _percentile_stretch_to_uint8(ratio_db)
    rgb = np.stack([r, g, b], axis=0)

    processed_dir.mkdir(parents=True, exist_ok=True)
    rgb_profile = profile.copy()
    rgb_profile.update(dtype="uint8", count=3, nodata=0)
    with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
        dst.write(rgb)

    # median_filter has no native NaN support (a single NaN in a window
    # poisons the whole window), so fill NaN with a representative scalar for
    # the filter pass only and re-mask them afterward - this confines the
    # effect to the true nodata rim at the scene/window edge, not real pixels.
    s1cfg = cfg["sentinel1"]
    fill_value = float(np.nanmedian(vv_db))
    vv_db_filled = np.where(valid_mask, vv_db, fill_value)
    vv_despeckled = median_filter(vv_db_filled, size=s1cfg["speckle_filter_size_px"])

    otsu_threshold_db = threshold_otsu(vv_despeckled[valid_mask])
    water_bool = (vv_despeckled < otsu_threshold_db) & valid_mask

    struct = np.ones((s1cfg["morphology_struct_size_px"],) * 2, dtype=bool)
    water_bool = binary_opening(water_bool, structure=struct)
    water_bool = binary_closing(water_bool, structure=struct) & valid_mask

    water_mask = np.where(valid_mask, water_bool.astype("uint8"), 255)
    n_water = int(np.sum(water_mask == 1))
    n_valid = int(np.sum(water_mask != 255))

    water_profile = profile.copy()
    water_profile.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(water_path, "w", **water_profile) as dst:
        dst.write(water_mask, 1)

    print(
        f"{item_id}: Otsu VV threshold={otsu_threshold_db:.2f} dB (despeckled, "
        f"{s1cfg['speckle_filter_size_px']}px median filter), "
        f"{n_water}/{n_valid} pixels ({n_water / n_valid:.1%}) classified as SAR water "
        f"after {s1cfg['morphology_struct_size_px']}px open/close cleanup"
    )
    return rgb_path, water_path


def compute_s1_water(cfg: dict, s1_items: list[tuple[str, dict]]) -> list[tuple[Path, Path]]:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    outputs = []
    for item_id, info in s1_items:
        result = _compute_one(cfg, item_id, info, processed_dir)
        if result is not None:
            outputs.append(result)

    if not outputs:
        raise RuntimeError("No Sentinel-1 scene produced a usable calibrated/water raster.")
    return outputs


if __name__ == "__main__":
    from limburg.fetch.sentinel1 import fetch_sentinel1_bands
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    s1_items = fetch_sentinel1_bands(cfg, boundary)
    compute_s1_water(cfg, s1_items)
