"""Runs N origin draws against one precomputed scenario and aggregates into
a per-plot exposure probability. Each draw is cheap (a single-cell-wide walk
bounded by the flow path length) - the precomputed scenario discharge
(Layer 1) is reused unchanged across every draw.
"""
from __future__ import annotations

from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from limburg.exposure_mc.origin_sampling import sample_origin_cell
from limburg.exposure_mc.trace_exposure import exposed_cells_to_plot_ids, trace_exposure
from limburg.network.capacity_constraints import get_default_capacity_m3_s


def run_monte_carlo(
    cfg: dict, plots: gpd.GeoDataFrame, graph: dict, discharge: np.ndarray,
    labels: np.ndarray, plot_ids: np.ndarray, time_window_min: float, n_draws: int, random_seed: int,
) -> pd.DataFrame:
    contaminated = plots[plots["overlaps_sar_water"]]
    if contaminated.empty:
        raise RuntimeError("No plots overlap SAR-detected water - cannot draw a Monte Carlo origin.")

    rng = np.random.default_rng(random_seed)
    n_cols = graph["shape"][1]
    velocity = cfg["routing"]["assumed_velocity_m_s"]
    capacity = get_default_capacity_m3_s(cfg)

    # Each draw is a single-cell-wide walk bounded by the flow path length,
    # not a raster-wide pass - n_draws in the thousands should still finish
    # in well under a minute. If this bar crawls, suspect time_window_min
    # being set absurdly high (walk runs until a true sink, which can be far
    # downstream) rather than a bug in trace_exposure itself.
    exposure_counts: Counter[str] = Counter()
    for _ in tqdm(range(n_draws), desc="Layer 2 Monte Carlo draws", unit="draw"):
        origin_cell = sample_origin_cell(contaminated, graph["transform"], graph["shape"], rng)
        exposed_cells = trace_exposure(
            origin_cell, graph["receiver"], discharge, n_cols,
            graph["cell_size_m"], velocity, capacity, time_window_min,
        )
        exposed_plot_ids = exposed_cells_to_plot_ids(exposed_cells, labels, plot_ids)
        exposure_counts.update(exposed_plot_ids)

    result = pd.DataFrame({"plot_id": plots["plot_id"]})
    result["exposure_count"] = result["plot_id"].map(exposure_counts).fillna(0).astype(int)
    result["exposure_probability"] = result["exposure_count"] / n_draws

    # 0 exposed plots across every draw almost always means the discharge
    # grid never exceeds capacity_m3_s anywhere reachable in time - check the
    # scenario's printed max discharge (scenario_grid.py) against
    # capacity_constraints.default_capacity_m3_s before suspecting the trace
    # walk itself.
    n_exposed = int((result["exposure_probability"] > 0).sum())
    print(
        f"Monte Carlo ({n_draws} draws, time_window={time_window_min:g}min): "
        f"{n_exposed}/{len(result)} plots exposed at least once "
        f"(mean probability {result['exposure_probability'].mean():.4f})"
    )
    return result
