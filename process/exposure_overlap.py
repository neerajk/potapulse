"""Step 10: polygonize the SAR water-mask mosaic and flag which seed potato
plots overlap SAR-detected surface water - the exposure layer visualize_limburg.py
highlights in the third panel.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from limburg.common import REPO_ROOT, load_config


def compute_plot_water_overlap(cfg: dict, water_mosaic_path: Path, plots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "seed_potato_plots_overlap.gpkg"
    if out_path.exists():
        print(f"Loading cached plot/SAR-water overlap from {out_path}")
        return gpd.read_file(out_path)

    with rasterio.open(water_mosaic_path) as src:
        water = src.read(1)
        transform = src.transform
        crs = src.crs

    water_mask = water == 1
    polygons = [
        shape(geom)
        for geom, val in shapes(water_mask.astype("uint8"), mask=water_mask, transform=transform)
        if val == 1
    ]
    print(f"Polygonized {len(polygons)} raw SAR water region(s)")

    # Second line of defense against residual SAR speckle (compute_s1_water.py
    # already despeckles + morphologically cleans the raster mask, but a
    # minimum-mapping-unit filter here catches anything that still slips
    # through before it can produce a false-positive plot overlap).
    min_area_m2 = cfg["sentinel1"]["min_water_body_area_m2"]
    polygons = [p for p in polygons if p.area >= min_area_m2]
    print(f"{len(polygons)} water region(s) remain after the {min_area_m2} m2 minimum-mapping-unit filter")

    plots_proj = plots.to_crs(crs).copy()
    if not polygons:
        plots_proj["overlaps_sar_water"] = False
    else:
        water_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        water_union = water_gdf.union_all()
        plots_proj["overlaps_sar_water"] = plots_proj.geometry.intersects(water_union)

    n_overlap = int(plots_proj["overlaps_sar_water"].sum())
    print(
        f"{n_overlap}/{len(plots_proj)} seed potato plots "
        f"({n_overlap / len(plots_proj):.1%}) overlap SAR-detected surface water"
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_proj.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path}")
    return plots_proj


if __name__ == "__main__":
    from limburg.process.clip_to_boundary import clip_raster_to_boundary
    from limburg.process.compute_s1_water import compute_s1_water
    from limburg.fetch.sentinel1 import fetch_sentinel1_bands
    from limburg.mosaic.s1 import mosaic_s1
    from limburg.boundaries.province import fetch_province_boundary
    from limburg.boundaries.seed_plots import load_seed_potato_plots

    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)
    s1_items = fetch_sentinel1_bands(cfg, boundary)
    s1_outputs = compute_s1_water(cfg, s1_items)
    _, water_mosaic_path = mosaic_s1(cfg, s1_outputs)
    water_mosaic_path = clip_raster_to_boundary(
        water_mosaic_path, boundary, processed_dir / "s1_water_mosaic_clipped.tif", nodata=255
    )
    compute_plot_water_overlap(cfg, water_mosaic_path, plots)
