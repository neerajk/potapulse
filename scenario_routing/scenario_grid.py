"""Precomputes discharge accumulation for every (intensity, duration) bin in
config's scenario_grid block - the expensive, origin-independent half of
Layer 1. Each bin is one topological-order accumulation pass over Limburg's
own D8 receiver graph (routing.py - NOT richdem, see that module's
docstring for why); Layer 2 only ever reads this cache, never recomputes it
per Monte Carlo draw.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
from tqdm import tqdm

from limburg.common import REPO_ROOT
from limburg.network.graph_build import load_network_graph
from limburg.network.soil_params import compute_runoff_coefficient_grid
from limburg.scenario_routing.routing import accumulate_discharge
from limburg.scenario_routing.runoff import compute_runoff_discharge_m3_s


def _scenario_path(cfg: dict, intensity_mm_hr: float, duration_min: float) -> Path:
    cache_dir = REPO_ROOT / cfg["scenario_grid"]["cache_dir"]
    return cache_dir / f"i{intensity_mm_hr:g}_d{duration_min:g}.npz"


def build_scenario_grid(cfg: dict, boundary: gpd.GeoDataFrame, graph_path: Path) -> list[Path]:
    graph = load_network_graph(graph_path)
    coeff_grid = compute_runoff_coefficient_grid(cfg, boundary, graph)

    intensities = cfg["scenario_grid"]["intensity_bins_mm_hr"]
    durations = cfg["scenario_grid"]["duration_bins_min"]
    combos = list(product(intensities, durations))

    # Each iteration is one topological-order accumulation pass (~19s at
    # Limburg's real 52.6M-cell size, per routing.py's benchmark) - the bar
    # tracks scenarios completed, not progress within one pass (that loop
    # has no sub-progress signal). If one combo takes dramatically longer
    # than ~20-30s, check that graph["receiver"]/["topo_order"] loaded
    # correctly rather than suspecting runoff.py.
    written = []
    for intensity, duration in tqdm(combos, desc="Layer 1 scenario grid", unit="scenario"):
        out_path = _scenario_path(cfg, intensity, duration)
        if out_path.exists():
            print(f"Scenario i={intensity:g} d={duration:g} already cached, skipping ({out_path})")
            written.append(out_path)
            continue

        weights = compute_runoff_discharge_m3_s(coeff_grid, graph["cell_size_m"], intensity)
        discharge = accumulate_discharge(graph["receiver"], graph["topo_order"], weights)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, discharge=discharge, intensity_mm_hr=intensity, duration_min=duration)
        # max discharge near 0 across every scenario usually means the runoff
        # coefficient grid came back all-default (check soil_params.py's
        # texture-classification print above for an unexpectedly empty
        # BOFEK2020 clip) rather than a routing bug.
        print(f"Scenario i={intensity:g} mm/hr d={duration:g} min: max discharge {discharge.max():.3g} m3/s -> {out_path}")
        written.append(out_path)
    return written


def load_scenario(cfg: dict, intensity_mm_hr: float, duration_min: float) -> np.ndarray:
    policy = cfg["scenario_grid"]["out_of_grid_policy"]
    intensities = cfg["scenario_grid"]["intensity_bins_mm_hr"]
    durations = cfg["scenario_grid"]["duration_bins_min"]

    if intensity_mm_hr not in intensities or duration_min not in durations:
        if policy == "error":
            raise ValueError(
                f"(intensity={intensity_mm_hr}, duration={duration_min}) is not in the precomputed "
                f"grid (intensities={intensities}, durations={durations})."
            )
        intensity_mm_hr = min(intensities, key=lambda v: abs(v - intensity_mm_hr))
        duration_min = min(durations, key=lambda v: abs(v - duration_min))

    path = _scenario_path(cfg, intensity_mm_hr, duration_min)
    if not path.exists():
        raise FileNotFoundError(f"Scenario cache missing at {path} - run build_scenario_grid() first.")
    return np.load(path)["discharge"]
