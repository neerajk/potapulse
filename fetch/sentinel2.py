"""Step 3: search Element84 Earth Search STAC for every Sentinel-2 MGRS tile
intersecting the Limburg province boundary, pick the least-cloudy scene per
tile in the Aug-Sep 2025 window, then clip + extract each one's green/nir/scl
bands plus the true-color "visual" (TCI) asset via a windowed read against
the public COGs (no full-tile download).

Same endpoint/collection/asset-key pattern as br_simulation's NDWI module
(NDWI/fetch_sentinel2.py) - verified live against
https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a.
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

from limburg.common import REPO_ROOT, load_config


def search_best_item_per_tile(cfg: dict, province_boundary) -> tuple[list, list[float]]:
    """Return (selected_items, province_bbox_4326): one item per distinct MGRS
    tile intersecting the province, each the least-cloudy scene available for
    that tile in the configured date range."""
    s2cfg = cfg["sentinel2"]
    province_geom_4326 = province_boundary.to_crs("EPSG:4326").geometry.union_all()
    bbox_4326 = list(province_geom_4326.bounds)

    client = Client.open(s2cfg["stac_endpoint"])
    search = client.search(
        collections=[s2cfg["collection"]],
        bbox=bbox_4326,
        datetime=s2cfg["date_range"],
        query={"eo:cloud_cover": {"lt": s2cfg["max_cloud_cover"]}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A items found for bbox {bbox_4326} in "
            f"{s2cfg['date_range']} under {s2cfg['max_cloud_cover']}% cloud cover - "
            f"widen date_range or max_cloud_cover in config/limburg_config.yaml."
        )

    best_per_tile: dict[str, object] = {}
    for item in items:
        tile_id = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile")
        if tile_id is None:
            continue
        cloud = item.properties.get("eo:cloud_cover", 100)
        current = best_per_tile.get(tile_id)
        if current is None or cloud < current.properties.get("eo:cloud_cover", 100):
            best_per_tile[tile_id] = item

    selected = list(best_per_tile.values())
    print(f"Found {len(selected)} distinct MGRS tile(s) intersecting Limburg:")
    for item in selected:
        tile_id = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile")
        print(
            f"  {tile_id}: {item.id}, cloud_cover={item.properties.get('eo:cloud_cover'):.1f}%, "
            f"date={item.properties.get('datetime')}"
        )

    union_geom = box(*selected[0].bbox)
    for item in selected[1:]:
        union_geom = union_geom.union(box(*item.bbox))
    covered_pct = union_geom.intersection(province_geom_4326).area / province_geom_4326.area * 100
    if covered_pct < 99.0:
        print(
            f"WARNING: selected tiles' union covers only ~{covered_pct:.0f}% of Limburg - "
            f"some area will be missing from the mosaic (likely a tile with no clear-enough "
            f"scene in this date range); widen max_cloud_cover/date_range to fill it."
        )
    else:
        print(f"Selected tiles' union covers ~{covered_pct:.0f}% of Limburg.")

    return selected, bbox_4326


def _clip_asset(href: str, bbox_4326: tuple[float, float, float, float], out_path: Path) -> Path:
    """Reads every band in the source asset (1 for green/nir/scl, 3 for the
    true-color "visual" TCI) so this one helper covers both cases."""
    with rasterio.open(href) as src:
        bounds_native = transform_bounds("EPSG:4326", src.crs, *bbox_4326)
        window = from_bounds(*bounds_native, transform=src.transform)
        data = src.read(window=window, boundless=True, fill_value=src.nodata or 0)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform, driver="GTiff")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path


def fetch_sentinel2_bands(cfg: dict, province_boundary) -> list[tuple[str, dict[str, Path]]]:
    """Fetch green/nir/scl for every selected tile, each clipped to that
    tile's own intersection with the province bbox."""
    s2cfg = cfg["sentinel2"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    items, province_bbox_4326 = search_best_item_per_tile(cfg, province_boundary)
    province_box_4326 = box(*province_bbox_4326)

    results: list[tuple[str, dict[str, Path]]] = []
    for item in items:
        clip_bbox = box(*item.bbox).intersection(province_box_4326).bounds

        band_paths: dict[str, Path] = {}
        for band_key in ("green_asset", "nir_asset", "scl_asset", "visual_asset"):
            asset_name = s2cfg[band_key]
            out_path = raw_dir / f"{item.id}_{asset_name}.tif"
            if out_path.exists():
                print(f"{item.id}/{asset_name} already fetched, skipping ({out_path})")
            else:
                href = item.assets[asset_name].href
                print(f"Fetching + clipping {item.id}/{asset_name} from {href}")
                _clip_asset(href, clip_bbox, out_path)
            band_paths[asset_name] = out_path
        results.append((item.id, band_paths))

    return results


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    fetch_sentinel2_bands(cfg, boundary)
