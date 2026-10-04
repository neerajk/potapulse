"""Per-step figures for run_limburg_risk.py. Real plot polygons and the real
province outline everywhere - no size exaggeration (an earlier circle-marker
version was reverted: real shapes against real S2/NDWI/soil backgrounds with
a bright high-contrast outline is the intended look, not a geometric stand-in).
Most panels show the whole province (the S2/NDWI mosaics are already
province-clipped, so their native raster extent already frames it); only
Step 6's per-plot grid zooms in tight, which is the deliberate "detail" half
of an overview+detail split, not a global workaround.

Categorical/status colors validated with the dataviz skill's palette
validator (ΔE floors, not eyeballed) - see each function's comment for which
check applies.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from rasterio.windows import from_bounds as window_from_bounds

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style, raster_extent
from limburg.fetch.pdok_aerial import fetch_aerial_crop
from limburg.network.soil_params import _classify_texture

apply_style()

# Validated categorical slots (dataviz skill palette, first 3 slots pass the
# all-pairs CVD/normal-vision floors for a choropleth; a 4th real category
# (veen) reuses the aqua slot with a hatch instead of a 4th competing hue -
# a hand-picked 4-hue set failed validation (worst pair ΔE 3.2), confirmed
# via scripts/validate_palette.js, not assumed).
SOIL_COLORS = {"zand": "#2a78d6", "leem": "#eb6834", "klei": "#1baf7a", "veen": "#1baf7a"}
SOIL_HATCH = {"veen": "////"}
SOIL_DEFAULT_COLOR = "#898781"  # muted gray - "no data"/unmatched, not a competing hue

GEWASCODE_COLORS = {"seed_nak": "#2a78d6", "seed_tbm": "#1baf7a"}

# Fixed status palette (dataviz skill - never themed, never reused for a
# generic series): critical = flagged/at-risk, not a shade of a continuous scale.
STATUS_CRITICAL = "#d03b3b"
STATUS_NEUTRAL = "#c3c2b7"

PLOT_OUTLINE_ON_IMAGERY = "#ffffff"  # white reads against S2 RGB, NDWI-blue, terrain, and the soil fills alike


def _save(fig, out_path: Path, dpi: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def _read_s2_rgba(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        rgb = src.read()[:3]
        extent = raster_extent(src.transform, src.width, src.height)
    alpha = np.where(np.any(rgb > 0, axis=0), 255, 0).astype("uint8")
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
    return rgba, extent


def _draw_s2_rgb(ax, s2_rgb_path: Path) -> None:
    rgba, extent = _read_s2_rgba(s2_rgb_path)
    ax.imshow(rgba, extent=extent, zorder=1)


def _draw_plots_outline(ax, plots: gpd.GeoDataFrame, color: str = PLOT_OUTLINE_ON_IMAGERY, linewidth: float = 1.3, zorder: int = 3) -> None:
    plots.boundary.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)


def plot_seed_plots_overview(cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, s2_rgb_path: Path, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 11))
    _draw_s2_rgb(ax, s2_rgb_path)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    for category, color in GEWASCODE_COLORS.items():
        sub = plots[plots["category"] == category]
        if not sub.empty:
            sub.boundary.plot(ax=ax, color=color, linewidth=1.8, zorder=3, label=f"{category} (n={len(sub)})")

    ax.set_title(f"Seed potato plots (n={len(plots)}) on S2 true-color imagery")
    ax.legend(loc="upper right", fontsize=8)
    add_map_chrome(ax, "EPSG:28992")
    add_scale_bar(ax)
    add_north_arrow(ax)
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def plot_baseline_overview(cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, ndwi_mosaic_path: Path, out_path: Path) -> Path:
    """Background is Sentinel-2 NDWI (optical) - the plotted
    `overlaps_sar_water` statistic is still Sentinel-1 SAR-based (that's the
    value actually used downstream as the contamination flag), so the
    backdrop and the flagged count come from two different sensors by
    design here, not an oversight."""
    n_overlap = int(plots["overlaps_sar_water"].sum())

    with rasterio.open(ndwi_mosaic_path) as src:
        ndwi = src.read(1)
        extent = raster_extent(src.transform, src.width, src.height)
    ndwi_masked = np.ma.masked_invalid(ndwi)

    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(ndwi_masked, cmap="Blues", vmin=0, vmax=1, extent=extent, zorder=1)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    _draw_plots_outline(ax, plots, color="#000000")

    ax.set_title(f"Seed potato plots on S2 NDWI (water signal) - {n_overlap}/{len(plots)} plots overlap S1 SAR water")
    plt.colorbar(im, ax=ax, shrink=0.6, label="NDWI (water-filtered)")
    add_map_chrome(ax, "EPSG:28992")
    add_scale_bar(ax)
    add_north_arrow(ax)
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def plot_waterways_overview(cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, waterways: gpd.GeoDataFrame, s2_rgb_path: Path, out_path: Path) -> Path:
    categories = waterways["waterway"].fillna("unknown")
    top3 = categories.value_counts().index[:3].tolist()
    # Fold anything past the top 3 into "other" - a choropleth/map context
    # can't safely hold more than 3 adjacent categorical hues (see
    # SOIL_COLORS comment); folding matches the skill's own prescribed move.
    colors = {top3[i]: c for i, c in enumerate(["#2a78d6", "#eb6834", "#1baf7a"][: len(top3)])}

    fig, ax = plt.subplots(figsize=(10, 11))
    _draw_s2_rgb(ax, s2_rgb_path)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    for cat in categories.unique():
        subset = waterways[categories == cat]
        color = colors.get(cat, SOIL_DEFAULT_COLOR)
        label = f"{cat} (n={len(subset)})" if cat in colors else None
        subset.plot(ax=ax, color=color, linewidth=1.2, zorder=3, label=label)
    _draw_plots_outline(ax, plots, zorder=4)

    ax.set_title(f"OSM waterways on S2 true-color imagery (n={len(waterways)} lines)")
    ax.legend(loc="upper right", fontsize=8)
    add_map_chrome(ax, "EPSG:28992")
    add_scale_bar(ax)
    add_north_arrow(ax)
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def plot_dem_burn_overview(
    cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, s2_rgb_path: Path, before_path: Path, after_path: Path, out_path: Path
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(26, 10))

    ax = axes[0]
    _draw_s2_rgb(ax, s2_rgb_path)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    _draw_plots_outline(ax, plots, zorder=3)
    ax.set_title("S2 true-color imagery")
    add_map_chrome(ax, "EPSG:28992")

    for ax, path, title in [(axes[1], before_path, "DTM before burn"), (axes[2], after_path, "DTM after burn (ditches)")]:
        with rasterio.open(path) as src:
            data = src.read(1).astype("float64")
            nodata = src.nodata
            extent = raster_extent(src.transform, src.width, src.height)
        is_nodata = np.isnan(data) if (nodata is not None and np.isnan(nodata)) else (data == nodata)
        masked = np.ma.masked_where(is_nodata, data)
        valid = data[~is_nodata]
        cmap = plt.get_cmap("terrain").copy()
        cmap.set_bad(color="#dddddd")
        im = ax.imshow(masked, cmap=cmap, vmin=float(valid.min()), vmax=float(valid.max()), extent=extent, zorder=1)
        boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
        _draw_plots_outline(ax, plots, color="#000000", zorder=3)
        ax.set_title(title)
        add_map_chrome(ax, "EPSG:28992")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Elevation (m NAP)")

    fig.suptitle("DEM burn check: imagery + elevation before/after", fontsize=14, fontweight="bold")
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def plot_bofek_soil_overview(
    cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, bofek_units: gpd.GeoDataFrame, s2_rgb_path: Path, out_path: Path
) -> Path:
    desc_field = cfg["bofek2020"]["description_field"]
    keyword_pairs = cfg["runoff"]["texture_keywords"]

    units = bofek_units.copy()
    # fillna BEFORE grouping - `df[col] == None` silently matches zero rows
    # in pandas (None/NaN never equal anything via ==), which would make
    # every "unmatched" BOFEK polygon disappear from the map without an
    # error, not just render with the fallback color.
    units["texture_class"] = units[desc_field].apply(lambda d: _classify_texture(d, keyword_pairs)).fillna("unmatched")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    _draw_s2_rgb(ax, s2_rgb_path)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    _draw_plots_outline(ax, plots, zorder=3)
    ax.set_title("S2 true-color imagery")
    add_map_chrome(ax, "EPSG:28992")

    ax = axes[1]
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.5, zorder=1)
    for texture_class in units["texture_class"].unique():
        subset = units[units["texture_class"] == texture_class]
        color = SOIL_COLORS.get(texture_class, SOIL_DEFAULT_COLOR)
        hatch = SOIL_HATCH.get(texture_class)
        subset.plot(ax=ax, color=color, hatch=hatch, edgecolor="none", zorder=2, label=f"{texture_class} (n={len(subset)})")
    _draw_plots_outline(ax, plots, color="#000000", zorder=3)
    ax.set_title("BOFEK2020 soil classification")
    ax.legend(loc="upper right", fontsize=8)
    add_map_chrome(ax, "EPSG:28992")

    fig.suptitle("Soil texture (drives per-cell runoff coefficient - see docs/OPEN_DECISIONS.md)", fontsize=13, fontweight="bold")
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def plot_discharge_overview(
    cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, graph: dict, discharge: np.ndarray,
    intensity_mm_hr: float, duration_min: float, out_path: Path,
) -> Path:
    extent = raster_extent(graph["transform"], graph["shape"][1], graph["shape"][0])
    # LogNorm can't take 0 - discharge is 0 wherever no runoff is generated
    # upstream at all (e.g. the raster's nodata halo outside the province
    # polygon), so mask those out rather than clipping them to an arbitrary
    # floor (which would misleadingly color them as "some water").
    masked = np.ma.masked_where(discharge <= 0, discharge)

    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(masked, cmap="Blues", norm=LogNorm(vmin=max(masked.min(), 1e-6), vmax=masked.max()), extent=extent, zorder=1)
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8, zorder=2)
    _draw_plots_outline(ax, plots, color="#000000", zorder=3)

    ax.set_title(f"Accumulated discharge: i={intensity_mm_hr:g}mm/hr, d={duration_min:g}min (log scale)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Discharge (m3/s, log scale)")
    add_map_chrome(ax, "EPSG:28992")
    add_scale_bar(ax)
    add_north_arrow(ax)
    return _save(fig, out_path, cfg["visualization"]["dpi"])


def _crop_window_bounds(geom, buffer_m: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def _draw_s2_crop(ax, s2_rgb_path: Path, bounds: tuple[float, float, float, float]) -> None:
    with rasterio.open(s2_rgb_path) as src:
        window = window_from_bounds(*bounds, transform=src.transform)
        rgb = src.read(window=window)[:3]
        win_transform = src.window_transform(window)
        extent = raster_extent(win_transform, rgb.shape[2], rgb.shape[1])
    alpha = np.where(np.any(rgb > 0, axis=0), 255, 0).astype("uint8")
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), alpha[..., None]], axis=-1)
    ax.imshow(rgba, extent=extent, zorder=1)


def _draw_aerial_crop(ax, cfg: dict, bounds: tuple[float, float, float, float]):
    """PDOK Actueel_orthoHR (8cm/px) instead of the 10m Sentinel-2 mosaic -
    only for tight, specific crops (see fetch/pdok_aerial.py). Returns the
    fetched (mosaic, transform) so a caller animating many frames over the
    SAME crop (scenario GIFs) can fetch once and reuse it, rather than
    re-fetching tiles over the network on every frame."""
    mosaic, transform = fetch_aerial_crop(cfg, bounds)
    extent = raster_extent(transform, mosaic.shape[1], mosaic.shape[0])
    ax.imshow(mosaic, extent=extent, zorder=1)
    return mosaic, transform


def plot_exposure_risk_overview(
    cfg: dict, boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, risk, s2_rgb_path: Path, out_path: Path
) -> Path:
    plots = plots.copy()
    plots["plot_id"] = plots["plot_id"].astype(str)
    risk = risk.copy()
    risk["plot_id"] = risk["plot_id"].astype(str)
    merged = plots.merge(risk[["plot_id", "exposure_count"]], on="plot_id", how="left")
    exposed_mask = merged["exposure_count"] > 0
    exposed_plots = merged[exposed_mask].reset_index(drop=True)
    n_exposed = len(exposed_plots)

    n_detail_cols = 2
    n_detail_rows = max(1, -(-max(n_exposed, 1) // n_detail_cols))
    fig_height = max(10, 4 * n_detail_rows)

    # Limburg is tall and narrow (bbox height/width ~2.5x) - a hardcoded wide
    # overview column leaves most of its allocated width empty once
    # add_map_chrome's equal-aspect axis renders the real shape. Size that
    # column from the province's own bounding-box aspect ratio instead of a
    # fixed guess, so the overview panel is only as wide as its map actually
    # needs at this figure height.
    pminx, pminy, pmaxx, pmaxy = boundary.total_bounds
    province_aspect = (pmaxy - pminy) / (pmaxx - pminx)  # height/width, >1 = tall
    overview_width_in = fig_height / province_aspect
    detail_width_in = 4.0
    fig_width = overview_width_in + detail_width_in * n_detail_cols

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(n_detail_rows, 1 + n_detail_cols, width_ratios=[overview_width_in] + [detail_width_in] * n_detail_cols)

    ax_overview = fig.add_subplot(gs[:, 0])
    boundary.boundary.plot(ax=ax_overview, color="#444444", linewidth=0.8, zorder=1)
    merged[~exposed_mask].plot(ax=ax_overview, color=STATUS_NEUTRAL, edgecolor="black", linewidth=0.4, zorder=2, label=f"Never exposed (n={int((~exposed_mask).sum())})")
    merged[exposed_mask].plot(ax=ax_overview, color=STATUS_CRITICAL, edgecolor="black", linewidth=0.6, zorder=3, label=f"Exposed >=1 draw (n={n_exposed})")
    origin_eligible = merged[merged["overlaps_sar_water"]]
    ax_overview.scatter(origin_eligible.geometry.centroid.x, origin_eligible.geometry.centroid.y, marker="*", color="black", s=70, zorder=4, label=f"Origin-eligible (n={len(origin_eligible)})")
    ax_overview.set_title(f"Exposure risk - whole province\n{n_exposed}/{len(merged)} plots exposed >=1 draw")
    ax_overview.legend(loc="upper right", fontsize=8)
    add_map_chrome(ax_overview, "EPSG:28992")
    add_scale_bar(ax_overview)
    add_north_arrow(ax_overview)

    for i in range(n_detail_rows * n_detail_cols):
        row, col = divmod(i, n_detail_cols)
        ax = fig.add_subplot(gs[row, 1 + col])
        if i >= n_exposed:
            ax.axis("off")
            continue
        plot_row = exposed_plots.iloc[i]
        bounds = _crop_window_bounds(plot_row.geometry, buffer_m=150)
        _draw_aerial_crop(ax, cfg, bounds)
        gpd.GeoSeries([plot_row.geometry], crs=plots.crs).boundary.plot(ax=ax, color=STATUS_CRITICAL, linewidth=2.0, zorder=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"plot_id {plot_row['plot_id']}\n{plot_row['gewas']} ({plot_row['jaar']})\n{plot_row['area_m2']:.0f} m2",
            fontsize=8,
        )

    fig.suptitle("Exposure risk: overview + individual detail on each exposed plot (real BRP metadata)", fontsize=13, fontweight="bold")
    return _save(fig, out_path, cfg["visualization"]["dpi"])
