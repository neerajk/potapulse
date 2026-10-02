"""Run the full Limburg module end-to-end:

  province boundary -> seed potato plots
  -> Sentinel-2 bands (+ true-color visual) -> NDWI -> water-filtered mosaic
     + true-color mosaic
  -> Sentinel-1 bands -> calibration -> RGB composite + Otsu water mask -> mosaics
  -> AHN4 DTM tile grid -> DTM mosaic
  -> plot/SAR-water overlap
  -> 3-panel SAR-focused visualization + 4-panel data overview visualization

Every step is independently skip-if-already-done (checked against its own
output file(s) under limburg/data/), so this is safe to re-run after a
partial failure without redoing completed work or re-spending a STAC search /
band download / WCS fetch. Delete the specific output file to force
recomputation of just that step.
"""
from __future__ import annotations

import numpy as np
import rasterio

from limburg.process.clip_to_boundary import clip_raster_to_boundary
from limburg.process.compute_ndwi import compute_ndwi
from limburg.process.compute_s1_water import compute_s1_water
from limburg.process.exposure_overlap import compute_plot_water_overlap
from limburg.fetch.ahn4_dtm import fetch_ahn_dtm_tiles
from limburg.fetch.sentinel1 import fetch_sentinel1_bands
from limburg.fetch.sentinel2 import fetch_sentinel2_bands
from limburg.common import REPO_ROOT, load_config
from limburg.mosaic.dtm import mosaic_dtm
from limburg.mosaic.ndwi import mosaic_ndwi
from limburg.mosaic.s1 import mosaic_s1
from limburg.mosaic.s2_visual import mosaic_s2_visual
from limburg.boundaries.province import fetch_province_boundary
from limburg.boundaries.seed_plots import load_seed_potato_plots
from limburg.visualize.limburg_overview import visualize_limburg
from limburg.visualize.data_overview import visualize_overview


def main() -> None:
    cfg = load_config()
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    boundary = fetch_province_boundary(cfg)
    plots = load_seed_potato_plots(cfg, boundary)

    s2_tile_band_paths = fetch_sentinel2_bands(cfg, boundary)

    ndwi_water_paths = compute_ndwi(cfg, s2_tile_band_paths)
    ndwi_water_mosaic_path = mosaic_ndwi(cfg, ndwi_water_paths)
    ndwi_water_mosaic_path = clip_raster_to_boundary(
        ndwi_water_mosaic_path, boundary, processed_dir / "ndwi_water_mosaic_clipped.tif", nodata=np.nan
    )

    s2_visual_mosaic_path = mosaic_s2_visual(cfg, s2_tile_band_paths)
    s2_visual_mosaic_path = clip_raster_to_boundary(
        s2_visual_mosaic_path, boundary, processed_dir / "s2_visual_mosaic_clipped.tif", nodata=0
    )

    s1_items = fetch_sentinel1_bands(cfg, boundary)
    s1_outputs = compute_s1_water(cfg, s1_items)
    s1_rgb_mosaic_path, s1_water_mosaic_path = mosaic_s1(cfg, s1_outputs)
    s1_rgb_mosaic_path = clip_raster_to_boundary(
        s1_rgb_mosaic_path, boundary, processed_dir / "s1_rgb_mosaic_clipped.tif", nodata=0
    )
    s1_water_mosaic_path = clip_raster_to_boundary(
        s1_water_mosaic_path, boundary, processed_dir / "s1_water_mosaic_clipped.tif", nodata=255
    )

    dtm_tile_paths = fetch_ahn_dtm_tiles(cfg, boundary)
    dtm_mosaic_path = mosaic_dtm(cfg, dtm_tile_paths)
    with rasterio.open(dtm_mosaic_path) as dtm_ref:
        dtm_nodata = dtm_ref.nodata
    dtm_mosaic_path = clip_raster_to_boundary(
        dtm_mosaic_path, boundary, processed_dir / "dtm_mosaic_clipped.tif", nodata=dtm_nodata
    )

    plots_overlap = compute_plot_water_overlap(cfg, s1_water_mosaic_path, plots)

    visualize_limburg(
        cfg, s1_rgb_mosaic_path, ndwi_water_mosaic_path, s1_water_mosaic_path, plots_overlap, boundary
    )
    visualize_overview(
        cfg, s2_visual_mosaic_path, ndwi_water_mosaic_path, s1_water_mosaic_path, dtm_mosaic_path, plots, boundary
    )


if __name__ == "__main__":
    main()
