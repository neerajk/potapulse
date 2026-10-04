"""Run Layer 0-3's non-interactive half end-to-end for Limburg:

  province boundary -> seed potato plots
  -> Sentinel-1 water mosaic -> plot/water overlap (contamination baseline)
  -> OSM waterways -> burned+filled DTM -> D8 receiver graph
  -> plot label raster
  -> scenario grid (BOFEK2020-informed runoff -> richdem discharge accumulation)
  -> Monte Carlo exposure at the dashboard's default scenario -> risk map
     + 3-4 dedicated per-scenario GIFs (most plots exposed / longest flow path)

Every step is independently skip-if-already-done, same as run_limburg.py -
if this crashes partway, just re-run the whole script; completed steps
print "already cached, skipping" and cost nothing. Every step also writes at
least one figure under outputs/figures/, with real plot polygons (never
exaggerated) and the real province outline throughout.

For the interactive sliders, use `streamlit run limburg/dashboard/app.py`
instead - it reads the same caches this script writes.
"""
from __future__ import annotations

from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots
from limburg.common import REPO_ROOT, load_config
from limburg.exposure_mc.aggregate_risk import run_monte_carlo
from limburg.fetch.bofek2020 import fetch_bofek_clipped
from limburg.fetch.sentinel1 import fetch_sentinel1_bands
from limburg.fetch.sentinel2 import fetch_sentinel2_bands
from limburg.fetch.waterways import fetch_waterways_clipped
from limburg.mosaic.ndwi import mosaic_ndwi
from limburg.mosaic.s1 import mosaic_s1
from limburg.mosaic.s2_visual import mosaic_s2_visual
from limburg.network.graph_build import build_network_graph, load_network_graph
from limburg.network.plot_node_assignment import build_plot_label_raster, load_plot_labels
from limburg.process.burn_waterways_dtm import prepare_burned_dtm
from limburg.process.clip_to_boundary import clip_raster_to_boundary
from limburg.process.compute_ndwi import compute_ndwi
from limburg.process.compute_s1_water import compute_s1_water
from limburg.process.exposure_overlap import compute_plot_water_overlap
from limburg.scenario_routing.scenario_grid import build_scenario_grid, load_scenario
from limburg.visualize.montecarlo_gif import generate_best_scenario_gifs
from limburg.visualize.pipeline_figures import (
    plot_baseline_overview,
    plot_bofek_soil_overview,
    plot_dem_burn_overview,
    plot_discharge_overview,
    plot_exposure_risk_overview,
    plot_seed_plots_overview,
    plot_waterways_overview,
)


def _banner(step_num: int, total: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f"STEP {step_num}/{total}: {title}")
    print("=" * 70)


TOTAL_STEPS = 6


def main() -> None:
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    s2_rgb_path = processed_dir / "s2_visual_mosaic_clipped.tif"
    ndwi_path = processed_dir / "ndwi_water_mosaic_clipped.tif"

    # --- Step 1: boundary + seed potato plots -------------------------------
    _banner(1, TOTAL_STEPS, "Province boundary + seed potato plots")
    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)
    print(f"-> {len(plots)} seed potato plots loaded")

    # S2 true-color mosaic is the shared imagery background for most figures
    # below - fetch/mosaic it now (skip-if-done; already cached from the
    # original run_limburg.py pipeline in most cases).
    s2_tile_band_paths = fetch_sentinel2_bands(cfg, boundary)
    s2_rgb_path = mosaic_s2_visual(cfg, s2_tile_band_paths)
    s2_rgb_path = clip_raster_to_boundary(s2_rgb_path, boundary, processed_dir / "s2_visual_mosaic_clipped.tif", nodata=0)

    plot_seed_plots_overview(cfg, boundary, plots, s2_rgb_path, figures_dir / "00_seed_plots.png")

    # --- Step 2: Sentinel-1 water + baseline contamination ------------------
    _banner(2, TOTAL_STEPS, "Sentinel-1 SAR water mosaic + plot/water overlap")
    s1_items = fetch_sentinel1_bands(cfg, boundary)
    s1_outputs = compute_s1_water(cfg, s1_items)
    _, water_mosaic_path = mosaic_s1(cfg, s1_outputs)
    water_mosaic_path = clip_raster_to_boundary(
        water_mosaic_path, boundary, processed_dir / "s1_water_mosaic_clipped.tif", nodata=255
    )
    plots = compute_plot_water_overlap(cfg, water_mosaic_path, plots)
    n_contaminated = int(plots["overlaps_sar_water"].sum())
    print(f"-> baseline contamination: {n_contaminated}/{len(plots)} plots overlap SAR-detected water")
    if n_contaminated == 0:
        print("WARNING: zero contaminated plots - Layer 2 will fail later unless this changes.")

    # NDWI (Sentinel-2 optical water signal) is this step's figure background
    # (the overlap statistic itself stays S1-based - see
    # plot_baseline_overview's docstring).
    ndwi_water_paths = compute_ndwi(cfg, s2_tile_band_paths)
    ndwi_mosaic_path = mosaic_ndwi(cfg, ndwi_water_paths)
    ndwi_mosaic_path = clip_raster_to_boundary(ndwi_mosaic_path, boundary, ndwi_path, nodata=float("nan"))
    plot_baseline_overview(cfg, boundary, plots, ndwi_mosaic_path, figures_dir / "01_baseline_water_overlap.png")

    # --- Step 3: OSM waterways ------------------------------------------------
    _banner(3, TOTAL_STEPS, "OSM waterways (ditch network for burning)")
    waterways = fetch_waterways_clipped(cfg, boundary)
    print(f"-> {len(waterways)} waterway lines clipped to Limburg")
    plot_waterways_overview(cfg, boundary, plots, waterways, s2_rgb_path, figures_dir / "02_waterways.png")

    # --- Step 4: burned+filled DTM -> D8 receiver graph + plot labels --------
    _banner(4, TOTAL_STEPS, "Burned DTM -> D8 receiver graph + plot label raster")
    graph_path = build_network_graph(cfg, boundary, waterways)
    build_plot_label_raster(cfg, plots, graph_path)
    # prepare_burned_dtm is skip-if-already-done - build_network_graph already
    # triggered this computation above, so this just reloads the cached path.
    before_path = processed_dir / f"limburg_ahn_dtm_{cfg['dtm_burn']['target_resolution_m']:g}m.tif"
    after_path = prepare_burned_dtm(cfg, boundary, waterways)
    plot_dem_burn_overview(cfg, boundary, plots, s2_rgb_path, before_path, after_path, figures_dir / "03_dem_burn.png")

    # --- Step 5: Layer 1 scenario grid (BOFEK2020 runoff -> richdem routing) -
    _banner(5, TOTAL_STEPS, "Layer 1: scenario grid (runoff + discharge accumulation)")
    bofek_units = fetch_bofek_clipped(cfg, boundary)
    plot_bofek_soil_overview(cfg, boundary, plots, bofek_units, s2_rgb_path, figures_dir / "04_bofek_soil.png")
    build_scenario_grid(cfg, boundary, graph_path)

    # --- Step 6: Layer 2 Monte Carlo exposure at the default scenario --------
    _banner(6, TOTAL_STEPS, "Layer 2: Monte Carlo exposure + risk map + scenario GIFs")
    graph = load_network_graph(graph_path)
    labels, plot_ids = load_plot_labels(REPO_ROOT / cfg["paths"]["interim_dir"] / "plot_labels.npz")

    sliders = cfg["dashboard"]["sliders"]
    mc_cfg = cfg["dashboard"]["monte_carlo"]
    intensity = sliders["intensity_mm_hr"]["default"]
    duration = sliders["duration_min"]["default"]
    time_window = sliders["time_window_min"]["default"]
    print(f"-> scenario: intensity={intensity}mm/hr, duration={duration}min, time_window={time_window}min")

    discharge = load_scenario(cfg, intensity, duration)
    plot_discharge_overview(cfg, boundary, plots, graph, discharge, intensity, duration, figures_dir / "05_discharge.png")

    risk = run_monte_carlo(
        cfg, plots, graph, discharge, labels, plot_ids,
        time_window, mc_cfg["n_draws_live"], mc_cfg["random_seed"],
    )

    results_dir = REPO_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    risk_path = results_dir / "exposure_risk_limburg.csv"
    risk.to_csv(risk_path, index=False)
    print(f"Wrote {risk_path}")

    plot_exposure_risk_overview(cfg, boundary, plots, risk, s2_rgb_path, figures_dir / "06_exposure_risk_map.png")

    generate_best_scenario_gifs(
        cfg, plots, boundary, graph, discharge, labels, plot_ids, time_window,
        mc_cfg["gif_search_draws"], mc_cfg["gif_n_scenarios"], mc_cfg["gif_time_frames"], mc_cfg["random_seed"], figures_dir,
    )

    print()
    print("=" * 70)
    print("DONE - all 6 steps complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
