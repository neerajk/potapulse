from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from br_simulation.visualization.style import add_map_chrome, add_north_arrow, add_scale_bar, apply_style

apply_style()


def render_exposure_map(boundary: gpd.GeoDataFrame, plots: gpd.GeoDataFrame, risk: pd.DataFrame):
    merged = plots.merge(risk[["plot_id", "exposure_probability"]], on="plot_id", how="left")

    fig, ax = plt.subplots(figsize=(8, 10))
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.5, zorder=1)
    merged.plot(
        ax=ax, column="exposure_probability", cmap="Reds", linewidth=0, zorder=2,
        legend=True, legend_kwds={"label": "Exposure probability", "shrink": 0.6}, vmin=0, vmax=1,
    )
    ax.set_title(f"Limburg exposure risk (mean p={merged['exposure_probability'].mean():.3f})")
    add_map_chrome(ax, "EPSG:28992")
    add_scale_bar(ax)
    add_north_arrow(ax)
    return fig
