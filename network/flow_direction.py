"""D8 flow direction, vectorized over the 8 neighbor shifts (no per-cell
Python loop).

A filled DEM has large, exactly-flat plateaus. Comparing raw float
elevations for "is this neighbor lower" is unsafe there: floating-point
noise can make two neighbors compare as mutually "lower than each other" in
different directions, creating a 2-cell receiver cycle that corrupts flow
accumulation. Fix: give every cell a unique rank consistent with elevation
order (ties broken by raster scan index), and choose receivers using that
strict total order - a cycle becomes structurally impossible. The reported
slope still uses real elevation, not the rank.
"""
from __future__ import annotations

import numpy as np

_NEIGHBORS = [
    (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
    (0, -1, 1.0),                        (0, 1, 1.0),
    (1, -1, np.sqrt(2)),  (1, 0, 1.0),  (1, 1, np.sqrt(2)),
]


def _unique_rank(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat = dem.ravel()
    flat_valid = valid.ravel()
    n = flat.size
    idx = np.arange(n)
    sort_key = np.where(flat_valid, flat, np.inf)
    order = np.lexsort((idx, sort_key))
    rank = np.empty(n, dtype="float64")
    rank[order] = np.arange(n, dtype="float64")
    rank[~flat_valid] = -1.0
    return rank.reshape(dem.shape)


def compute_d8_receivers(dem: np.ndarray, nodata: float) -> dict:
    """Returns {"receiver": flat receiver index per cell (-1 = sink/nodata),
    "rank": the strict total order used to pick receivers (also the correct
    topological processing order for downstream accumulation - see
    graph_build.py/routing.py, which use this instead of asking richdem to
    recompute flow direction a second time), "n_sinks": int}."""
    n_rows, n_cols = dem.shape
    valid = dem != nodata
    rank = _unique_rank(dem, valid)

    best_slope = np.full(dem.shape, -np.inf, dtype="float64")
    receiver = np.full(dem.shape, -1, dtype="int64")
    row_idx, col_idx = np.indices(dem.shape)

    for dy, dx, dist in _NEIGHBORS:
        shifted = np.roll(np.roll(dem, -dy, axis=0), -dx, axis=1)
        shifted_valid = np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1)
        shifted_rank = np.roll(np.roll(rank, -dy, axis=0), -dx, axis=1)

        edge_mask = np.ones_like(valid)
        if dy == -1:
            edge_mask[0, :] = False
        elif dy == 1:
            edge_mask[-1, :] = False
        if dx == -1:
            edge_mask[:, 0] = False
        elif dx == 1:
            edge_mask[:, -1] = False

        nbr_row, nbr_col = row_idx + dy, col_idx + dx
        flat_idx = nbr_row * n_cols + nbr_col

        slope = (dem - shifted) / dist
        lower_rank = shifted_rank < rank
        consider = valid & shifted_valid & edge_mask & lower_rank & (slope > best_slope)
        best_slope = np.where(consider, slope, best_slope)
        receiver = np.where(consider, flat_idx, receiver)

    receiver[~valid] = -1
    n_sinks = int(((receiver == -1) & valid).sum())
    return {"receiver": receiver.astype("int64"), "rank": rank, "n_sinks": n_sinks}


def assert_acyclic(receiver: np.ndarray, valid: np.ndarray) -> None:
    """Regression test for the tie-break fix above - O(n * path_length), for
    a small synthetic raster, not a province-scale one."""
    flat_receiver = receiver.ravel()
    flat_valid = valid.ravel()
    n = flat_receiver.size
    for start in np.flatnonzero(flat_valid):
        cur = start
        steps = 0
        while cur != -1:
            cur = flat_receiver[cur]
            steps += 1
            if steps > n:
                raise AssertionError(f"Cycle detected in D8 receiver graph starting at cell {start}")
