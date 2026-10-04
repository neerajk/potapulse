"""Shared "zoom to the plots, not the whole province" helper. Limburg's 42
seed potato plots are a tiny fraction of the province's area - every figure
zoomed to the full province extent renders them as a few invisible pixels.
"""
from __future__ import annotations

import geopandas as gpd


def plots_bounds(plots: gpd.GeoDataFrame, buffer_m: float = 1500) -> tuple[float, float, float, float]:
    """Returns (minx, miny, maxx, maxy) - the plots' bounding box padded by
    buffer_m so plots at the edge aren't clipped by the axis border."""
    minx, miny, maxx, maxy = plots.total_bounds
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def apply_extent(ax, bounds: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def exaggerate_plots(plots: gpd.GeoDataFrame, radius_m: float = 600) -> gpd.GeoDataFrame:
    """Returns a DISPLAY-ONLY copy with each plot's geometry replaced by a
    fixed-radius circle around its centroid. Limburg's 42 plots are real
    farm fields (~100-400m across) scattered over a ~60km-wide area - even
    zoomed to their bounding box, true-to-scale polygons render as a few
    pixels wide, where the outline stroke visually swallows the fill color
    entirely (confirmed: a first version of the exposure map rendered as
    near-invisible slivers, legend notwithstanding). A fixed-radius circle
    is the standard cartographic fix (city dots on a world map, not true
    footprints) - never use this copy's geometry for analysis, only display."""
    exaggerated = plots.copy()
    exaggerated["geometry"] = plots.geometry.centroid.buffer(radius_m)
    return exaggerated
