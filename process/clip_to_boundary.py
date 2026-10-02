"""Step: mask + crop a province-wide mosaic to the exact Limburg polygon
(not just its bounding box). Without this, every mosaic keeps the raw shape
of whatever Sentinel-1/-2 scene footprints or MGRS tiles produced it - for
Sentinel-1 in particular that's two long diagonal orbit swaths, which show as
a wide "V" with large stretches of unrelated area and an apparent gap down
the middle once plotted, even though the two swaths' real coverage of the
(narrow, irregularly-shaped) Limburg polygon itself was already verified at
~100% in fetch_sentinel1.py's greedy coverage search. Clipping to the actual
polygon removes all of that excess/misleading extent so the figure's visible
data exactly fills the province shape.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask


def clip_raster_to_boundary(
    src_path: Path, boundary: gpd.GeoDataFrame, out_path: Path, nodata
) -> Path:
    if out_path.exists():
        print(f"{out_path.name} already clipped, skipping")
        return out_path

    with rasterio.open(src_path) as src:
        boundary_native = boundary.to_crs(src.crs)
        geoms = [geom.__geo_interface__ for geom in boundary_native.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=nodata, filled=True)
        profile = src.profile.copy()

    profile.update(
        height=out_image.shape[1], width=out_image.shape[2], transform=out_transform, nodata=nodata
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_image)

    nodata_is_nan = isinstance(nodata, float) and np.isnan(nodata)
    valid = np.sum(~np.isnan(out_image)) if nodata_is_nan else np.sum(out_image != nodata)
    print(
        f"Clipped {src_path.name} -> {out_path.name}: "
        f"{out_image.shape[2]}x{out_image.shape[1]}px, {int(valid)} valid px"
    )
    return out_path
