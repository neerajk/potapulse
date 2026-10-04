"""Burned 10m DTM -> cell-indexed D8 receiver graph. Every raster cell is
implicitly a node (its index = row * n_cols + col); receiver[cell] is its
single downstream neighbor's index, or -1 at a sink. A catchment/junction
graph is unnecessary at this resolution and would add a second hydrology-
correctness surface for no benefit.

burn_waterways_dtm.py's burn (a fixed-depth drop, no depression filling) can
leave local pits - FillDepressions is applied here before computing
direction, same as the main br_simulation pipeline's dem_burn.py.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import richdem as rd

from limburg.common import REPO_ROOT, load_config
from limburg.network.flow_direction import compute_d8_receivers
from limburg.process.burn_waterways_dtm import prepare_burned_dtm


def build_network_graph(cfg: dict, boundary: gpd.GeoDataFrame, waterways_clipped: gpd.GeoDataFrame) -> Path:
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    out_path = interim_dir / "network_graph.npz"
    if out_path.exists():
        print(f"Network graph already built, skipping ({out_path})")
        return out_path

    burned_dtm_path = prepare_burned_dtm(cfg, boundary, waterways_clipped)

    with rasterio.open(burned_dtm_path) as src:
        dem = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else -9999.0
        transform = src.transform
        crs = str(src.crs)
        cell_size_m = abs(transform.a)

    rd_array = rd.rdarray(dem, no_data=nodata)
    rd.FillDepressions(rd_array, in_place=True)
    # FillDepressions alone leaves large EXACTLY-flat plateaus - without a
    # real gradient across them, this module's own D8 computation below
    # would fall back entirely on its scan-order tie-break (still correct/
    # acyclic, but hydrologically arbitrary across a big flat). ResolveFlats
    # (Garbrecht & Martz 1997) imposes a local gradient so there's a
    # physically meaningful downhill direction everywhere, not just a
    # technically-valid one.
    rd.ResolveFlats(rd_array, in_place=True)
    dem = np.asarray(rd_array)

    result = compute_d8_receivers(dem, nodata)
    # A handful of sinks (province-edge outlets, real local lows after
    # filling) is normal. A LARGE sink count usually means the burned DTM
    # still has unfilled depressions reaching the raster edge - check that
    # FillDepressions above actually ran against the real nodata value (not
    # a mismatched sentinel) before suspecting compute_d8_receivers itself.
    print(f"Built D8 receiver graph: {dem.size} cells, {result['n_sinks']} sinks")

    # Topological processing order for downstream accumulation
    # (scenario_routing/routing.py): descending rank = sources (highest
    # elevation) first, so every cell is pushed to its receiver only after
    # everything upstream of it has already been added. Computed ONCE here
    # (one O(n log n) sort) and reused for every rainfall scenario - this is
    # what replaces richdem.FlowAccumulation, which kept hitting its own
    # internal assertion on this DEM's real NODATA-irregular, province-edge
    # topology even after Fill+ResolveFlats. This module's own receiver
    # array is acyclic by construction (strict rank ordering - see
    # flow_direction.py), so accumulating along it can't hit that failure
    # mode at all, by proof rather than by hoping richdem's second,
    # independent recomputation agrees.
    topo_order = np.argsort(-result["rank"].ravel()).astype("int64")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        receiver=result["receiver"], topo_order=topo_order, dem=dem.astype("float32"), nodata=np.float64(nodata),
        transform=np.array(transform)[:6], shape=np.array(dem.shape),
        cell_size_m=np.float64(cell_size_m), crs=np.array(crs),
    )
    return out_path


def load_network_graph(path: Path) -> dict:
    npz = np.load(path, allow_pickle=False)
    return {
        "receiver": npz["receiver"], "topo_order": npz["topo_order"], "dem": npz["dem"],
        "nodata": float(npz["nodata"]), "transform": rasterio.Affine(*npz["transform"]),
        "shape": tuple(npz["shape"]), "cell_size_m": float(npz["cell_size_m"]), "crs": str(npz["crs"]),
    }


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary
    from limburg.fetch.waterways import fetch_waterways_clipped

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    waterways = fetch_waterways_clipped(cfg, boundary)
    build_network_graph(cfg, boundary, waterways)
