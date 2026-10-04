"""Downstream discharge accumulation along this project's own D8 receiver
array - NOT richdem.FlowAccumulation.

richdem.FlowAccumulation(method="D8") independently recomputes flow
direction from raw elevations every time it's called, and on Limburg's real
burned DTM (NODATA-irregular at the clipped province boundary, tens of
millions of cells) it hit its own internal C++ assertion
("elevations(ci)>=elevations(receiver)") even after FillDepressions +
ResolveFlats conditioning (graph_build.py) - confirmed live against the real
data, not a synthetic case. This project's own `receiver` array
(network/flow_direction.py) is acyclic and never-uphill BY CONSTRUCTION (a
strict unique-rank total order, not a second from-scratch elevation
comparison), so accumulating along it can't hit that failure mode at all.

Processing order matters here: every cell must be added to its receiver
only after everything upstream of it has already been folded in. The
topological order (`topo_order`, computed once in graph_build.py from the
same rank used to build `receiver`) guarantees that. Benchmarked at
Limburg's real scale (52.6M cells): ~19s per call in plain Python - fine for
the 24 scenarios in config's scenario_grid (one-time cost per scenario, not
per Monte Carlo draw).
"""
from __future__ import annotations

import numpy as np


def accumulate_discharge(receiver: np.ndarray, topo_order: np.ndarray, runoff_weights_m3_s: np.ndarray) -> np.ndarray:
    shape = runoff_weights_m3_s.shape
    flat_receiver = receiver.ravel()
    acc = runoff_weights_m3_s.ravel().astype("float64").copy()

    # Plain Python loop, deliberately not vectorized: each cell's addition
    # to its receiver depends on every upstream cell already being summed,
    # so this can't be a single vectorized pass without first bucketing
    # cells by tree depth (a real option if 19s/scenario ever stops being
    # fast enough - not needed today). If this takes dramatically longer
    # than ~20s, check that topo_order/receiver are being reloaded from the
    # cache correctly rather than suspecting this loop's logic.
    for idx in topo_order:
        r = flat_receiver[idx]
        if r != -1:
            acc[r] += acc[idx]

    return acc.reshape(shape).astype("float32")
