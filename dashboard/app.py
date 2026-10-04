"""Interactive dashboard: sliders for rainfall intensity, duration, and
exposure time-window; runs Layer 2's Monte Carlo live per change. Assumes
`python -m limburg.run_limburg_risk` has already been run at least once (it
reads cached Layer 0/1 outputs, it does not compute them).

Run with: streamlit run limburg/dashboard/app.py
"""
from __future__ import annotations

import geopandas as gpd
import streamlit as st

from limburg.common import REPO_ROOT, load_config
from limburg.boundaries.province import fetch_province_boundary
from limburg.dashboard.map_render import render_exposure_map
from limburg.network.graph_build import load_network_graph
from limburg.network.plot_node_assignment import load_plot_labels
from limburg.scenario_routing.scenario_grid import load_scenario


@st.cache_resource
def _load_static():
    cfg = load_config()
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    graph_path = interim_dir / "network_graph.npz"
    labels_path = interim_dir / "plot_labels.npz"
    plots_path = processed_dir / "seed_potato_plots_overlap.gpkg"

    missing = [p for p in (graph_path, labels_path, plots_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing precomputed output(s): {missing} - run `python -m limburg.run_limburg_risk` first.")

    graph = load_network_graph(graph_path)
    labels, plot_ids = load_plot_labels(labels_path)
    plots = gpd.read_file(plots_path)
    boundary = fetch_province_boundary(cfg)
    return cfg, graph, labels, plot_ids, plots, boundary


def main() -> None:
    st.set_page_config(page_title="Limburg flood exposure risk", layout="wide")
    st.title("Brown rot flood-exposure risk - Limburg")

    cfg, graph, labels, plot_ids, plots, boundary = _load_static()
    sliders = cfg["dashboard"]["sliders"]
    mc_cfg = cfg["dashboard"]["monte_carlo"]

    col1, col2, col3 = st.columns(3)
    with col1:
        intensity = st.select_slider(
            "Rainfall intensity (mm/hr)", options=cfg["scenario_grid"]["intensity_bins_mm_hr"],
            value=sliders["intensity_mm_hr"]["default"],
        )
    with col2:
        duration = st.select_slider(
            "Rainfall duration (min)", options=cfg["scenario_grid"]["duration_bins_min"],
            value=sliders["duration_min"]["default"],
        )
    with col3:
        tw = sliders["time_window_min"]
        time_window = st.slider("Exposure time window (min)", tw["min"], tw["max"], tw["default"], tw["step"])

    from limburg.exposure_mc.aggregate_risk import run_monte_carlo

    discharge = load_scenario(cfg, intensity, duration)
    risk = run_monte_carlo(
        cfg, plots, graph, discharge, labels, plot_ids,
        time_window, mc_cfg["n_draws_live"], mc_cfg["random_seed"],
    )

    fig = render_exposure_map(boundary, plots, risk)
    st.pyplot(fig)
    st.caption(
        f"{mc_cfg['n_draws_live']} Monte Carlo draws against the precomputed "
        f"i={intensity}mm/hr, d={duration}min scenario."
    )


if __name__ == "__main__":
    main()
