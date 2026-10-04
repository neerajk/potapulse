"""Rational Method (Q = C*i*A) per-cell runoff generation. `duration_min` is
not used here (classic Rational Method ties duration to a time-of-
concentration lookup, not modeled in this pass) - it's used downstream in
exposure_mc's time-window check.
"""
from __future__ import annotations

import numpy as np

MM_HR_TO_M_S = 1.0 / 1000.0 / 3600.0


def compute_runoff_discharge_m3_s(runoff_coefficient_grid: np.ndarray, cell_size_m: float, intensity_mm_hr: float) -> np.ndarray:
    intensity_m_s = intensity_mm_hr * MM_HR_TO_M_S
    cell_area_m2 = cell_size_m**2
    return (runoff_coefficient_grid * intensity_m_s * cell_area_m2).astype("float32")
