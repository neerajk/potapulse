"""Step: merge the fetched AHN4 DTM tiles into one Limburg-wide DTM mosaic.

All tiles are requested directly in EPSG:28992 at the same overview
resolution (fetch_ahn_dtm.py), so - unlike mosaic_ndwi.py/mosaic_s1.py, which
merge Sentinel scenes that arrive in UTM - no reprojection step is needed
here, just a straight merge.
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.merge import merge as rio_merge

from limburg.common import REPO_ROOT, load_config


def mosaic_dtm(cfg: dict, tile_paths: list[Path]) -> Path:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_path = processed_dir / "dtm_mosaic.tif"
    if out_path.exists():
        print(f"DTM mosaic already computed, skipping ({out_path})")
        return out_path

    with rasterio.open(tile_paths[0]) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata

    print(f"Merging {len(tile_paths)} AHN4 DTM tile(s) into one mosaic...")
    mosaic, mosaic_transform = rio_merge(tile_paths, nodata=nodata)
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    valid = mosaic[0][mosaic[0] != nodata]
    print(
        f"DTM mosaic written: {mosaic.shape[2]}x{mosaic.shape[1]}px, "
        f"elevation range [{valid.min():.1f}, {valid.max():.1f}] m NAP -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    from limburg.fetch.ahn4_dtm import fetch_ahn_dtm_tiles
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    tile_paths = fetch_ahn_dtm_tiles(cfg, boundary)
    mosaic_dtm(cfg, tile_paths)
