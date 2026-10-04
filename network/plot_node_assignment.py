"""Rasterize every seed potato plot footprint onto the network graph's cell
grid, so Layer 2 can go from "this cell got exposed" to "these plot_ids are
exposed" by a single array lookup - checking a plot's whole footprint
(all_touched), not just its centroid cell.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from limburg.common import REPO_ROOT
from limburg.network.graph_build import load_network_graph


def build_plot_label_raster(cfg: dict, plots: gpd.GeoDataFrame, graph_path: Path) -> Path:
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    out_path = interim_dir / "plot_labels.npz"
    if out_path.exists():
        print(f"Plot label raster already built, skipping ({out_path})")
        return out_path

    graph = load_network_graph(graph_path)
    # .astype(str) forces a fixed-width numpy string dtype ("<U.."), not
    # pandas/numpy's default "object" array of plain Python strings -
    # npz can only save an object array via pickle, which load_plot_labels's
    # allow_pickle=False then refuses to read back ("Object arrays cannot be
    # loaded when allow_pickle=False").
    plot_ids = plots["plot_id"].to_numpy().astype(str)
    shapes = [(geom, i) for i, geom in enumerate(plots.geometry) if geom is not None and not geom.is_empty]

    labels = rasterize(
        shapes, out_shape=graph["shape"], transform=graph["transform"],
        fill=-1, dtype="int32", all_touched=True,
    )
    n_hit = int((labels >= 0).sum())
    n_plots_hit = len(np.unique(labels[labels >= 0]))
    print(f"Rasterized {len(plots)} plots onto the network grid ({n_hit} cells, {n_plots_hit}/{len(plots)} plots hit)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, labels=labels, plot_ids=plot_ids)
    return out_path


def load_plot_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path, allow_pickle=False)
    return npz["labels"], npz["plot_ids"]
