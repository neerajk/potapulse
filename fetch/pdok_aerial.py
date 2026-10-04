"""PDOK "Actueel_orthoHR" aerial orthophoto (8cm/px RGB) - fetched tile by
tile directly from the confirmed-live REST ResourceURL template, not via
GDAL's generic WMTS driver (see config/limburg_config.yaml's pdok_aerial
block for why that was abandoned). Only ever called for small, specific
crops - never at province scale.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from rasterio.transform import Affine

from limburg.common import REPO_ROOT

HTTP_TIMEOUT_S = 30
MAX_TILES_PER_CROP = 2000  # fail loudly rather than silently grinding through
                             # thousands of sequential-feeling fetches if a
                             # caller passes an unexpectedly large bbox


def _resolution_m(cfg: dict) -> float:
    acfg = cfg["pdok_aerial"]
    return (acfg["scale_denominator_level0"] * 0.00028) / (2 ** acfg["zoom_level"])


def _fetch_tile(cfg: dict, level: int, col: int, row: int) -> np.ndarray:
    acfg = cfg["pdok_aerial"]
    cache_dir = REPO_ROOT / acfg["cache_dir"]
    cache_path = cache_dir / f"{level}_{col}_{row}.jpg"

    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        url = acfg["resource_url_template"].format(level=level, col=col, row=row)
        resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        content = resp.content
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)

    return np.array(Image.open(BytesIO(content)).convert("RGB"))


def fetch_aerial_crop(cfg: dict, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, Affine]:
    """bounds: (minx, miny, maxx, maxy) in EPSG:28992. Returns (rgb array
    HxWx3 uint8, affine transform) covering AT LEAST the requested bounds
    (snapped outward to whole tiles - the caller crops the view with
    set_xlim/set_ylim, same pattern as pipeline_figures.py's S2 crop)."""
    acfg = cfg["pdok_aerial"]
    level, tile_px = acfg["zoom_level"], acfg["tile_px"]
    topleft_x, topleft_y = acfg["topleft_x"], acfg["topleft_y"]
    res = _resolution_m(cfg)
    tile_size_m = tile_px * res

    minx, miny, maxx, maxy = bounds
    col_start = int((minx - topleft_x) // tile_size_m)
    col_end = int((maxx - topleft_x) // tile_size_m)
    row_start = int((topleft_y - maxy) // tile_size_m)
    row_end = int((topleft_y - miny) // tile_size_m)

    n_cols, n_rows = col_end - col_start + 1, row_end - row_start + 1
    n_tiles = n_cols * n_rows
    if n_tiles > MAX_TILES_PER_CROP:
        raise RuntimeError(
            f"fetch_aerial_crop: bounds {bounds} would need {n_tiles} tiles at zoom "
            f"{level} (cap {MAX_TILES_PER_CROP}) - pass a smaller bbox or a coarser "
            f"zoom_level in config, don't silently fetch thousands of tiles."
        )
    mosaic = np.zeros((n_rows * tile_px, n_cols * tile_px, 3), dtype="uint8")

    # Sequential fetching was confirmed live to leave even one 5-panel figure
    # unfinished after 1000+ tiles - tiles are independent HTTP GETs, so a
    # thread pool (network-bound, same reasoning as the project's other
    # run_io_bound-style fetchers) is the fix, not a coarser zoom alone.
    tasks = [(r, c) for r in range(row_start, row_end + 1) for c in range(col_start, col_end + 1)]

    def _fetch_and_place(rc: tuple[int, int]) -> None:
        r, c = rc
        tile = _fetch_tile(cfg, level, c, r)
        ri, ci = r - row_start, c - col_start
        mosaic[ri * tile_px:(ri + 1) * tile_px, ci * tile_px:(ci + 1) * tile_px, :] = tile

    max_workers = min(acfg["max_parallel_fetches"], len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_fetch_and_place, tasks))

    mosaic_origin_x = topleft_x + col_start * tile_size_m
    mosaic_origin_y = topleft_y - row_start * tile_size_m
    transform = Affine(res, 0, mosaic_origin_x, 0, -res, mosaic_origin_y)
    return mosaic, transform
