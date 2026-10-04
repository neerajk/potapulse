"""Draw a random contaminated-baseline plot as the flood origin and resolve
it to a cell index on the network graph's grid, via its centroid - ground
truth on which plot is actually infected doesn't exist, so this draw is the
Monte Carlo mechanism. Origin pool = plots with `overlaps_sar_water` True
(process/exposure_overlap.py).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np


def plot_centroid_to_cell(plot_geom, transform, shape: tuple[int, int]) -> int | None:
    col_f, row_f = ~transform * (plot_geom.centroid.x, plot_geom.centroid.y)
    row, col = int(row_f), int(col_f)
    n_rows, n_cols = shape
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        return None
    return row * n_cols + col


def sample_origin_cell(contaminated_plots: gpd.GeoDataFrame, transform, shape: tuple[int, int], rng: np.random.Generator) -> int:
    candidates = contaminated_plots.sample(frac=1, random_state=rng.integers(0, 2**31 - 1))
    for _, plot in candidates.iterrows():
        cell = plot_centroid_to_cell(plot.geometry, transform, shape)
        if cell is not None:
            return cell
    raise RuntimeError("No baseline-contaminated plot's centroid falls within the network grid.")
