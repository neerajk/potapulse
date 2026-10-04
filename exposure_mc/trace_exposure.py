"""From a drawn origin cell, walk the D8 receiver chain downstream,
accumulating travel time (distance / assumed_velocity_m_s). At each visited
cell, compare the precomputed scenario discharge against the capacity
threshold to decide "wet" vs. "water passed by without exceeding capacity",
and stop once the time window is exceeded.
"""
from __future__ import annotations

import numpy as np


def trace_exposure(
    origin_cell: int, receiver: np.ndarray, discharge: np.ndarray, n_cols: int,
    cell_size_m: float, assumed_velocity_m_s: float, capacity_m3_s: float, time_window_min: float,
) -> set[int]:
    exposed: set[int] = set()
    cur = origin_cell
    travel_time_min = 0.0
    max_steps = receiver.size

    for _ in range(max_steps):
        if discharge.flat[cur] >= capacity_m3_s:
            exposed.add(cur)

        nxt = receiver.flat[cur]
        if nxt == -1:
            break

        cur_row, cur_col = divmod(cur, n_cols)
        nxt_row, nxt_col = divmod(nxt, n_cols)
        step_dist_m = cell_size_m * np.hypot(nxt_row - cur_row, nxt_col - cur_col)
        travel_time_min += (step_dist_m / assumed_velocity_m_s) / 60.0

        if travel_time_min > time_window_min:
            break
        cur = nxt

    return exposed


def exposed_cells_to_plot_ids(exposed_cells: set[int], labels: np.ndarray, plot_ids: np.ndarray) -> set[str]:
    flat_labels = labels.ravel()
    result = set()
    for cell in exposed_cells:
        label = flat_labels[cell]
        if label >= 0:
            result.add(str(plot_ids[label]))
    return result
