"""Step 6: search Microsoft Planetary Computer's "sentinel-1-grd" STAC
collection for IW-mode scenes covering Limburg in the Aug-Sep 2025 window,
greedily select the fewest scenes needed to cover the province, sign each
scene's VV/VH + calibration-LUT assets (verified live 2026-10-02: PC signs
"sentinel-1-grd" anonymously, no subscription key required - unlike
"sentinel-1-rtc", which PC's own collection docs say needs a registered
account, deliberately avoided here), then clip VV/VH to the province bbox via
a windowed read.

GRD "measurement" GeoTIFFs are *not* map-projected - ESA ships them in ground-
range geometry georeferenced only by a sparse grid of GCPs (ground control
points) in the SAFE annotation, not a CRS+affine transform (confirmed live:
rasterio.open(href).crs is None for this collection, unlike the S2 COGs in
fetch_sentinel2.py). To window-crop to the province bbox without downloading
the full ~25000x17000px scene, this fits an approximate affine transform from
those GCPs (rasterio.transform.from_gcps - a least-squares fit, standard
practice for subsetting un-georeferenced SAR products) and uses it purely to
locate the AOI's pixel window; the output keeps that approximate
transform/CRS so mosaic_s1.py's reprojection step still has something to warp
from. This is an approximation (GRD ground-range geometry isn't perfectly
affine over a full scene), adequate for AOI-sized windows and a qualitative
water mask, but not survey-grade geolocation - re-verify against a known
water body if sub-pixel accuracy ever matters.

The clip window's (row_off, col_off) is saved alongside each scene, because
the calibration LUT (annotation/calibration/calibration-iw-*.xml) is indexed
in the *original full-scene* line/pixel space - calibrate_s1.py needs those
offsets to map the clipped array's local pixels back to LUT coordinates
without ever materializing the full scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import planetary_computer as pc
import rasterio
import requests
from pystac_client import Client
from rasterio.transform import from_gcps
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from limburg.common import REPO_ROOT, load_config

HTTP_TIMEOUT_S = 120


def _select_items_for_coverage(items: list, aoi_geom_4326, min_coverage_fraction: float) -> list:
    """Greedily add items (most recent first) until their footprint union
    covers >= min_coverage_fraction of the AOI, or items run out. Minimizes
    the number of scenes to calibrate/mosaic rather than blindly taking all
    search results."""
    items_sorted = sorted(items, key=lambda it: it.datetime, reverse=True)
    selected = []
    covered = None
    aoi_area = aoi_geom_4326.area

    for item in items_sorted:
        footprint = box(*item.bbox)
        if not footprint.intersects(aoi_geom_4326):
            continue
        new_covered = footprint if covered is None else covered.union(footprint)
        gained = new_covered.intersection(aoi_geom_4326).area - (
            0 if covered is None else covered.intersection(aoi_geom_4326).area
        )
        if gained <= 0 and covered is not None:
            continue
        selected.append(item)
        covered = new_covered
        frac = covered.intersection(aoi_geom_4326).area / aoi_area
        if frac >= min_coverage_fraction:
            break

    if not selected:
        raise RuntimeError("No Sentinel-1 GRD scene intersects the Limburg AOI in this date range.")

    final_frac = covered.intersection(aoi_geom_4326).area / aoi_area
    print(f"Selected {len(selected)} Sentinel-1 scene(s) covering ~{final_frac:.0%} of Limburg:")
    for item in selected:
        print(
            f"  {item.id}: {item.datetime.date()}, "
            f"orbit={item.properties.get('sat:relative_orbit')} "
            f"({item.properties.get('sat:orbit_state')})"
        )
    if final_frac < min_coverage_fraction:
        print(
            f"WARNING: coverage capped at ~{final_frac:.0%} (< {min_coverage_fraction:.0%} target) - "
            f"no further scenes in the date range add new area; widen date_range to fill the gap."
        )
    return selected


def search_sentinel1_items(cfg: dict, province_boundary) -> tuple[list, object]:
    s1cfg = cfg["sentinel1"]
    province_geom_4326 = province_boundary.to_crs("EPSG:4326").geometry.union_all()
    bbox_4326 = list(province_geom_4326.bounds)

    client = Client.open(s1cfg["stac_endpoint"])
    search = client.search(
        collections=[s1cfg["collection"]],
        bbox=bbox_4326,
        datetime=s1cfg["date_range"],
        query={"sar:instrument_mode": {"eq": s1cfg["instrument_mode"]}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-1 GRD {s1cfg['instrument_mode']} items found for bbox {bbox_4326} in "
            f"{s1cfg['date_range']} - widen date_range in config/limburg_config.yaml."
        )

    selected = _select_items_for_coverage(items, province_geom_4326, s1cfg["min_coverage_fraction"])
    return selected, province_geom_4326


def _download(url: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def _window_from_rotated_bounds(bounds: tuple[float, float, float, float], transform) -> Window:
    """rasterio.windows.from_bounds assumes an axis-aligned (non-rotated)
    transform and raises "Bounds and transform are inconsistent" otherwise.
    A GCP-fitted Sentinel-1 transform IS rotated (the ground track isn't
    north-aligned), so instead this inverts the transform against all 4 bbox
    corners (not just 2) and takes the pixel-space bounding box of those -
    the standard way to window a rotated/sheared affine transform."""
    left, bottom, right, top = bounds
    inv = ~transform
    cols, rows = zip(
        *(inv * (x, y) for x, y in [(left, bottom), (left, top), (right, bottom), (right, top)])
    )
    col_off, row_off = min(cols), min(rows)
    col_end, row_end = max(cols), max(rows)
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _clip_band(href: str, bbox_4326: tuple[float, float, float, float], out_path: Path) -> dict:
    with rasterio.open(href) as src:
        gcps, gcp_crs = src.get_gcps()
        if not gcps:
            raise RuntimeError(
                f"{href}: no CRS and no GCPs found - rasterio can't georeference this "
                f"GRD band by either path, refusing to guess a window."
            )
        approx_transform = from_gcps(gcps)
        bounds_native = transform_bounds("EPSG:4326", gcp_crs, *bbox_4326)
        window = _window_from_rotated_bounds(bounds_native, approx_transform)
        # Round outward so the saved offsets stay exact integers (what the
        # calibration-LUT interpolation needs for a correct local->global
        # pixel mapping) rather than rasterio's fractional default window.
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window, boundless=True, fill_value=0)
        out_transform = window_transform(window, approx_transform)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=out_transform,
            crs=gcp_crs,
            driver="GTiff",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    return {"col_off": int(window.col_off), "row_off": int(window.row_off)}


def fetch_sentinel1_bands(cfg: dict, province_boundary) -> list[tuple[str, dict]]:
    """Returns a list of (item_id, info) where info has keys: vv, vh,
    calibration_vv, calibration_vh (Paths) and col_off/row_off (ints)."""
    s1cfg = cfg["sentinel1"]
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"] / "s1"
    items, province_geom_4326 = search_sentinel1_items(cfg, province_boundary)
    bbox_4326 = tuple(province_geom_4326.bounds)

    results: list[tuple[str, dict]] = []
    for item in items:
        signed = pc.sign(item)
        window_json = raw_dir / f"{item.id}_window.json"

        info: dict = {}
        if window_json.exists():
            print(f"{item.id}: already fetched, skipping band clip")
            offsets = json.loads(window_json.read_text())
        else:
            vv_path = raw_dir / f"{item.id}_vv.tif"
            offsets = _clip_band(signed.assets[s1cfg["vv_asset"]].href, bbox_4326, vv_path)
            vh_path = raw_dir / f"{item.id}_vh.tif"
            _clip_band(signed.assets[s1cfg["vh_asset"]].href, bbox_4326, vh_path)
            window_json.write_text(json.dumps(offsets))
            print(f"{item.id}: clipped VV/VH, window offsets {offsets}")

        info["vv"] = raw_dir / f"{item.id}_vv.tif"
        info["vh"] = raw_dir / f"{item.id}_vh.tif"
        info["col_off"] = offsets["col_off"]
        info["row_off"] = offsets["row_off"]

        cal_vv_path = raw_dir / f"{item.id}_calibration_vv.xml"
        cal_vh_path = raw_dir / f"{item.id}_calibration_vh.xml"
        _download(signed.assets[s1cfg["calibration_vv_asset"]].href, cal_vv_path)
        _download(signed.assets[s1cfg["calibration_vh_asset"]].href, cal_vh_path)
        info["calibration_vv"] = cal_vv_path
        info["calibration_vh"] = cal_vh_path

        results.append((item.id, info))

    return results


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    fetch_sentinel1_bands(cfg, boundary)
