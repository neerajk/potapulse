"""Step 2: fetch seed potato (pootaardappelen) plots from the live PDOK BRP
Gewaspercelen feed, filtered to the configured gewascodes and clipped to the
Limburg province boundary.

Ported from br_simulation's Groningen fetch
(src/br_simulation/data_engineering/seed_plots.py), itself ported from the
sibling bruinrot/brainrot project's verified BRP extraction
(src/bruinrot/data/brp.py): the Atom feed is parsed live, the full-NL
GeoPackage is downloaded once and cached, then bbox + gewascode filtering is
pushed down to GDAL/OGR so the multi-GB national file is never materialized
in memory.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import geopandas as gpd
import requests
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from limburg.common import REPO_ROOT, load_config

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
HTTP_TIMEOUT_S = 60
DOWNLOAD_CHUNK_BYTES = 1 << 20


def _discover_gpkg_download(src_cfg: dict[str, Any], concept_year: int) -> dict[str, Any]:
    resp = requests.get(src_cfg["atom_feed_url"], timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    for entry in root.findall("a:entry", ATOM_NS):
        if entry.findtext("a:id", default="", namespaces=ATOM_NS) != src_cfg["gpkg_entry_id"]:
            continue
        for link in entry.findall("a:link", ATOM_NS):
            href = link.get("href", "")
            m = re.search(r"concept_(\d{4})\.gpkg$", href)
            if m and int(m.group(1)) == concept_year:
                return {"url": href, "length_bytes": int(link.get("length", 0)) or None}

    raise RuntimeError(
        f"No concept_{concept_year}.gpkg link found in the live BRP Atom feed - "
        f"RVO may have republished under a different year/filename, refusing to guess."
    )


def _ensure_downloaded(url: str, dest: Path, expected_length: int | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_length and dest.stat().st_size == expected_length:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading BRP GeoPackage (full NL, ~GB-scale) from {url}")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_S) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or expected_length
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="BRP GeoPackage"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_length and tmp.stat().st_size != expected_length:
        raise RuntimeError(
            f"Downloaded size {tmp.stat().st_size} != expected {expected_length} for {url}"
        )
    tmp.rename(dest)
    return dest


def _read_filtered(
    gpkg_path: Path, codes: list[int], bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    layers = gpd.list_layers(gpkg_path)
    layer_names = list(layers["name"])
    if len(layer_names) != 1:
        raise RuntimeError(f"Expected exactly 1 layer in {gpkg_path}, found {layer_names}")

    where = "gewascode IN (" + ",".join(str(c) for c in codes) + ")"
    return gpd.read_file(
        gpkg_path,
        layer=layer_names[0],
        bbox=bbox,
        where=where,
        engine="pyogrio",
        fid_as_index=True,
    )


def _polygonal_only(geom):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None


def _clean_and_clip(
    gdf: gpd.GeoDataFrame,
    province_boundary: gpd.GeoDataFrame,
    codes_cfg: dict[int, dict[str, Any]],
    min_plausible_area_m2: float,
) -> gpd.GeoDataFrame:
    n_in = len(gdf)
    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    gdf["geometry"] = gdf.geometry.apply(_polygonal_only)
    gdf = gdf[gdf.geometry.notna()]

    gdf["_wkb"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop_duplicates(subset=["_wkb", "gewascode"]).drop(columns=["_wkb"])

    gdf = gpd.clip(gdf, province_boundary)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    gdf["category"] = gdf["gewascode"].map(lambda c: codes_cfg[int(c)]["category_label"])
    gdf["area_m2"] = gdf.geometry.area
    n_small = int((gdf["area_m2"] < min_plausible_area_m2).sum())
    if n_small:
        print(
            f"WARNING: {n_small} plots below min_plausible_area_m2={min_plausible_area_m2} "
            f"(likely digitising slivers)"
        )

    gdf["plot_id"] = gdf.index.astype(str)
    print(f"BRP seed potato plots: {n_in} raw -> {len(gdf)} after clean/clip")
    return gdf.reset_index(drop=True)


def load_seed_potato_plots(cfg: dict, province_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    cache_path = processed_dir / "seed_potato_plots.gpkg"
    if cache_path.exists():
        print(f"Loading cached seed potato plots from {cache_path}")
        return gpd.read_file(cache_path)

    src = cfg["seed_potato_plots"]
    codes_cfg = {int(k): v for k, v in src["gewascodes"].items()}
    codes = list(codes_cfg.keys())

    entry = _discover_gpkg_download(src, src["concept_year"])
    cache_dir = REPO_ROOT / src.get("cache_dir", "limburg/data/raw/brp_cache")
    gpkg_path = cache_dir / Path(entry["url"]).name
    gpkg_path = _ensure_downloaded(entry["url"], gpkg_path, entry["length_bytes"])

    bbox = tuple(province_boundary.total_bounds)
    print(f"Filtering BRP GeoPackage to gewascodes {codes} within province bbox")
    raw = _read_filtered(gpkg_path, codes, bbox)
    if raw.empty:
        raise RuntimeError(f"No plots with gewascode in {codes} found within the province bbox")

    plots = _clean_and_clip(raw, province_boundary, codes_cfg, src.get("min_plausible_area_m2", 1.0))

    processed_dir.mkdir(parents=True, exist_ok=True)
    plots.to_file(cache_path, driver="GPKG")
    print(f"Wrote {cache_path}")
    return plots


if __name__ == "__main__":
    from limburg.boundaries.province import fetch_province_boundary

    cfg = load_config()
    boundary = fetch_province_boundary(cfg)
    load_seed_potato_plots(cfg, boundary)
