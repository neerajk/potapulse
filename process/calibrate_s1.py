"""Step 7: radiometric calibration of Sentinel-1 GRD digital numbers (DN) to
calibrated sigma0 backscatter (dB), per ESA's documented Sentinel-1 Level-1
calibration procedure (Miranda, N. et al., "Sentinel-1 Level 1 Detailed
Algorithm Definition," ESA S1-TN-MDA-52-7445, and as implemented by the
SNAP/S1TBX "Calibration" operator):

    sigma0_linear = DN^2 / sigmaNought_LUT(line, pixel)^2
    sigma0_dB     = 10 * log10(sigma0_linear)

sigmaNought_LUT is published as a sparse grid in the product's own
annotation/calibration/calibration-iw-*.xml (tags verified live 2026-10-02:
<calibrationVectorList><calibrationVector><line/><pixel count=.../>
<sigmaNought count=.../>...), interpolated here onto the full pixel grid with
scipy's RegularGridInterpolator (bilinear).

Planetary Computer's "sentinel-1-grd" assets are the *uncalibrated* DN GRD
product (confirmed live: raw uint16 digital numbers, not sigma0) - this is
why calibration is done explicitly here rather than skipped, unlike
"sentinel-1-rtc" which ships pre-calibrated gamma0 but requires a registered
PC account (avoided - see fetch_sentinel1.py).
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _parse_calibration_lut(xml_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lines, pixels, sigma_grid) where sigma_grid has shape
    (len(lines), len(pixels)), parsed from <calibrationVectorList>."""
    root = ET.parse(xml_path).getroot()
    vectors = root.findall(".//calibrationVector")
    if not vectors:
        raise RuntimeError(f"No <calibrationVector> entries found in {xml_path}")

    lines = np.array([int(v.findtext("line")) for v in vectors], dtype="float64")
    pixels = np.array([float(p) for p in vectors[0].findtext("pixel").split()], dtype="float64")
    sigma_grid = np.array(
        [[float(s) for s in v.findtext("sigmaNought").split()] for v in vectors],
        dtype="float64",
    )

    order = np.argsort(lines)
    return lines[order], pixels, sigma_grid[order]


def calibrate_to_db(
    dn: np.ndarray, calibration_xml_path: Path, row_off: int, col_off: int
) -> np.ndarray:
    """dn: 2D array of raw digital numbers from the clipped band GeoTIFF.
    row_off/col_off: the clipped window's offset in the *original* full-scene
    pixel grid (the LUT's coordinate system) - from fetch_sentinel1.py.
    Returns a float32 array of calibrated sigma0 in dB, NaN where dn == 0
    (outside the scene footprint / boundless-read fill)."""
    lines, pixels, sigma_grid = _parse_calibration_lut(calibration_xml_path)
    interpolator = RegularGridInterpolator(
        (lines, pixels), sigma_grid, method="linear", bounds_error=False, fill_value=None
    )

    local_rows, local_cols = np.indices(dn.shape)
    global_lines = (local_rows + row_off).ravel().astype("float64")
    global_pixels = (local_cols + col_off).ravel().astype("float64")
    # Clip query points into the LUT's own coverage so edge pixels extrapolate
    # from the nearest valid vector instead of producing NaN from the
    # interpolator itself (DN==0 masking below is the real "no data" signal).
    global_lines = np.clip(global_lines, lines.min(), lines.max())
    global_pixels = np.clip(global_pixels, pixels.min(), pixels.max())

    sigma_lut = interpolator(np.column_stack([global_lines, global_pixels])).reshape(dn.shape)

    dn_f = dn.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma0_linear = (dn_f**2) / (sigma_lut**2)
        sigma0_db = 10.0 * np.log10(sigma0_linear)

    sigma0_db[dn == 0] = np.nan
    return sigma0_db.astype("float32")
