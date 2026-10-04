"""Per-scenario animated GIFs: search many random origins, keep the most
illustrative few (most plots exposed in one draw; longest flow path), and
for EACH of those build a GIF that animates TIME WITHIN that one scenario -
frame i reveals however much of the flow path has been reached by that
elapsed time, not a flipbook of different random origins. Zoomed tight to
that scenario's own path extent, real plot shapes (no size exaggeration),
the province boundary line still drawn for context within the crop.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from rasterio.transform import xy as transform_xy

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.exposure_mc.origin_sampling import sample_origin_cell
from limburg.exposure_mc.trace_exposure import exposed_cells_to_plot_ids
from limburg.fetch.pdok_aerial import fetch_aerial_crop
from limburg.network.capacity_constraints import get_default_capacity_m3_s

apply_style()

STATUS_CRITICAL = "#d03b3b"


def _walk_path_with_exposure(
    origin_cell: int, receiver: np.ndarray, discharge: np.ndarray, n_cols: int,
    cell_size_m: float, assumed_velocity_m_s: float, capacity_m3_s: float, time_window_min: float,
) -> dict:
    """Same walk as exposure_mc/trace_exposure.py, but also returns the full
    path with per-step cumulative travel time, so a time-progression
    animation can reveal it incrementally - kept separate from
    trace_exposure.py since aggregate_risk.py's hot loop (run per Monte
    Carlo draw) has no use for this and shouldn't pay to build it 200 times."""
    path = [origin_cell]
    cum_time_min = [0.0]
    exposed_at_time: dict[int, float] = {}
    cur = origin_cell
    travel_time_min = 0.0

    for _ in range(receiver.size):
        if discharge.flat[cur] >= capacity_m3_s:
            exposed_at_time[cur] = travel_time_min
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
        path.append(cur)
        cum_time_min.append(travel_time_min)

    return {"origin_cell": origin_cell, "path": path, "cum_time_min": cum_time_min, "exposed_at_time": exposed_at_time}


def search_scenarios(
    plots: gpd.GeoDataFrame, graph: dict, discharge: np.ndarray, labels: np.ndarray, plot_ids: np.ndarray,
    time_window_min: float, velocity: float, capacity: float, n_search_draws: int, random_seed: int,
) -> list[dict]:
    """Searches n_search_draws random origins (cheap - ~1500/s per
    aggregate_risk.py's own benchmark) and returns every scenario with its
    exposed plot_ids attached, for select_best_scenarios() to rank."""
    contaminated = plots[plots["overlaps_sar_water"]]
    if contaminated.empty:
        raise RuntimeError("No plots overlap SAR-detected water - cannot search for scenarios.")

    rng = np.random.default_rng(random_seed)
    n_cols = graph["shape"][1]
    results = []
    for _ in range(n_search_draws):
        origin_cell = sample_origin_cell(contaminated, graph["transform"], graph["shape"], rng)
        scenario = _walk_path_with_exposure(origin_cell, graph["receiver"], discharge, n_cols, graph["cell_size_m"], velocity, capacity, time_window_min)
        scenario["exposed_plot_ids"] = exposed_cells_to_plot_ids(set(scenario["exposed_at_time"]), labels, plot_ids)
        results.append(scenario)
    return results


def select_best_scenarios(results: list[dict], n_total: int = 4) -> list[dict]:
    """Top candidates by (a) most distinct plots exposed in one draw and
    (b) longest flow path, deduplicated by origin so the same scenario isn't
    picked twice under both criteria."""
    by_most_exposed = sorted(results, key=lambda r: len(r["exposed_plot_ids"]), reverse=True)
    by_longest_path = sorted(results, key=lambda r: len(r["path"]), reverse=True)

    selected, seen_origins = [], set()
    for r in by_most_exposed[:2] + by_longest_path[:2]:
        if r["origin_cell"] not in seen_origins:
            selected.append(r)
            seen_origins.add(r["origin_cell"])
    # Backfill from by_most_exposed if dedup left fewer than n_total.
    for r in by_most_exposed:
        if len(selected) >= n_total:
            break
        if r["origin_cell"] not in seen_origins:
            selected.append(r)
            seen_origins.add(r["origin_cell"])
    return selected[:n_total]


def _discharge_to_linewidths(path_discharge: np.ndarray, min_lw: float = 1.2, max_lw: float = 7.0) -> np.ndarray:
    """Channel width scaling with sqrt(discharge) is the standard
    simplified hydraulic convention (width ~ sqrt(Q) for a roughly constant
    depth/slope) - used here only to make the trail look like a real stream
    that widens downstream, not as a calibrated channel-geometry claim.
    Scaled relative to this scenario's OWN max discharge along the path (not
    an absolute threshold), so every scenario's trail uses its full visual
    width range regardless of how big its raw discharge values are."""
    d = np.clip(path_discharge, 0, None)
    d_max = d.max() if d.max() > 0 else 1.0
    normalized = np.sqrt(d / d_max)
    return min_lw + normalized * (max_lw - min_lw)


def _draw_water_trail(ax, xs: np.ndarray, ys: np.ndarray, linewidths: np.ndarray) -> None:
    """A single flat-color line reads as an abstract vector, not water. A
    wider, low-alpha "halo" underneath a narrower core line gives a soft
    bloom/wet-edge look with two cheap LineCollection draws - no new
    dependency, same segments reused for both passes."""
    if len(xs) < 2:
        return
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_widths = (linewidths[:-1] + linewidths[1:]) / 2

    halo = LineCollection(segments, linewidths=seg_widths * 2.2, colors="#86b6ef", alpha=0.35, zorder=3, capstyle="round")
    core = LineCollection(segments, linewidths=seg_widths, colors="#1c5cab", alpha=0.95, zorder=4, capstyle="round")
    ax.add_collection(halo)
    ax.add_collection(core)


def _draw_leading_edge_glow(ax, x: float, y: float) -> None:
    """A soft multi-layer glow at the current wavefront - cheap stand-in for
    an actual particle/lighting effect, just a few overlapping translucent
    scatter markers of decreasing size, to draw the eye to where the water
    currently is (vs. the trail it already passed through)."""
    for size, alpha in [(900, 0.12), (500, 0.20), (220, 0.35)]:
        ax.scatter([x], [y], s=size, color="#9ec5f4", alpha=alpha, zorder=4, linewidths=0)
    ax.scatter([x], [y], s=70, color="#ffffff", edgecolors="#1c5cab", linewidths=1.2, zorder=5)


def generate_scenario_gif(
    cfg: dict, plots: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, graph: dict, scenario: dict,
    labels: np.ndarray, plot_ids: np.ndarray, discharge: np.ndarray,
    n_time_frames: int, out_path: Path, label: str,
) -> Path:
    n_cols = graph["shape"][1]
    path = scenario["path"]
    cum_time = np.array(scenario["cum_time_min"])
    final_time = cum_time[-1] if len(cum_time) else 0.0

    rows, cols = zip(*[divmod(c, n_cols) for c in path])
    xs, ys = transform_xy(graph["transform"], rows, cols, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    path_discharge = discharge.flat[np.array(path)]
    linewidths = _discharge_to_linewidths(path_discharge)
    buffer_m = 300
    bounds = (min(xs) - buffer_m, min(ys) - buffer_m, max(xs) + buffer_m, max(ys) + buffer_m)

    # PDOK aerial orthophoto (8cm/px) fetched ONCE for this scenario's crop
    # and reused across every frame - re-fetching tiles over the network on
    # every one of n_time_frames redraws would be slow and impolite to the
    # service, and the background never changes within one scenario anyway.
    aerial_mosaic, aerial_transform = fetch_aerial_crop(cfg, bounds)
    aerial_extent = raster_extent(aerial_transform, aerial_mosaic.shape[1], aerial_mosaic.shape[0])

    # Each exposed cell's reveal time, mapped to the plot(s) it belongs to -
    # so a frame can show "these plots are newly exposed by this time" using
    # the same path-index time axis the water trail is drawn against.
    exposed_cell_times = scenario["exposed_at_time"]

    frame_times = np.linspace(0, max(final_time, 1e-6), n_time_frames)
    fig, ax = plt.subplots(figsize=(9, 9))

    def draw_frame(i: int):
        ax.clear()
        t = frame_times[i]
        n_reveal = max(1, int(np.searchsorted(cum_time, t, side="right")))

        ax.imshow(aerial_mosaic, extent=aerial_extent, zorder=1)
        boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)

        reached_cells = {c for c in path[:n_reveal] if exposed_cell_times.get(c, np.inf) <= t}
        if reached_cells:
            exposed_now = exposed_cells_to_plot_ids(reached_cells, labels, plot_ids)
            mask = plots["plot_id"].astype(str).isin(exposed_now)
            if mask.any():
                plots[mask].boundary.plot(ax=ax, color=STATUS_CRITICAL, linewidth=2.2, zorder=3)

        _draw_water_trail(ax, xs[:n_reveal], ys[:n_reveal], linewidths[:n_reveal])
        _draw_leading_edge_glow(ax, xs[n_reveal - 1], ys[n_reveal - 1])
        ax.scatter([xs[0]], [ys[0]], marker="*", color="white", edgecolors="black", s=160, zorder=6)

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        add_map_chrome(ax, "EPSG:28992")
        add_scale_bar(ax)
        add_north_arrow(ax)
        ax.set_title(f"{label}\nt={t:.1f} min / {final_time:.1f} min")

    ani = FuncAnimation(fig, draw_frame, frames=n_time_frames, interval=400)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=3))
    print(f"Wrote {out_path} ({n_time_frames} frames, {label})")

    # "Flood extent" = this scenario's fully-played-out state (full
    # time_window_min elapsed), not a mid-animation moment - reuses the
    # exact same draw_frame() the GIF's last frame already uses, so this
    # PNG can never drift out of sync with what the animation actually shows.
    flood_extent_path = out_path.with_name(out_path.stem + "_flood_extent.png")
    draw_frame(n_time_frames - 1)
    ax.set_title(f"{label}\nFlood extent - full {final_time:.1f} min time window")
    fig.savefig(flood_extent_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {flood_extent_path}")

    plt.close(fig)
    return out_path


def generate_best_scenario_gifs(
    cfg: dict, plots: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, graph: dict, discharge: np.ndarray,
    labels: np.ndarray, plot_ids: np.ndarray, time_window_min: float,
    n_search_draws: int, n_scenario_gifs: int, n_time_frames: int, random_seed: int, out_dir: Path,
) -> list[Path]:
    velocity = cfg["routing"]["assumed_velocity_m_s"]
    capacity = get_default_capacity_m3_s(cfg)

    results = search_scenarios(plots, graph, discharge, labels, plot_ids, time_window_min, velocity, capacity, n_search_draws, random_seed)
    best = select_best_scenarios(results, n_scenario_gifs)
    print(f"Searched {n_search_draws} draws, selected {len(best)} scenario(s) for dedicated GIFs")

    written = []
    for i, scenario in enumerate(best):
        n_exp = len(scenario["exposed_plot_ids"])
        path_len = len(scenario["path"])
        label = f"Scenario {i + 1}: {n_exp} plot(s) exposed, {path_len}-cell path"
        out_path = out_dir / f"08_scenario_{i + 1}.gif"
        generate_scenario_gif(cfg, plots, boundary, graph, scenario, labels, plot_ids, discharge, n_time_frames, out_path, label)
        written.append(out_path)
    return written
