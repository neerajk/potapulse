"""Step 1: fetch the exact Limburg province polygon from PDOK's "BRK Bestuurlijke
Gebieden" OGC API Features (collection "provinciegebied").

Same endpoint family as br_simulation's Groningen fetch
(src/br_simulation/data_engineering/province_boundary.py) and the sibling
bruinrot/brainrot project's original (src/bruinrot/data/aoi.py) - requests
EPSG:28992 explicitly and checks the response's Content-Crs header rather than
assuming the server honored the request, and verifies the returned province
code matches the expected one (confirmed live 2026-10-02: naam=Limburg ->
code=31) before trusting the geometry.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests

from limburg.common import REPO_ROOT, load_config

HTTP_TIMEOUT_S = 60


def fetch_province_boundary(cfg: dict) -> gpd.GeoDataFrame:
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    cache_path = processed_dir / "province_boundary.gpkg"
    if cache_path.exists():
        print(f"Loading cached province boundary from {cache_path}")
        return gpd.read_file(cache_path)

    rcfg = cfg["region"]
    crs_uri = "http://www.opengis.net/def/crs/EPSG/0/28992"
    url = f"{rcfg['province_ogc_api_base']}/collections/{rcfg['province_collection']}/items"
    params = {"f": "json", rcfg["province_name_field"]: rcfg["province"], "crs": crs_uri}

    print(f"Fetching province boundary for {rcfg['province']} from PDOK OGC API")
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()

    content_crs = resp.headers.get("Content-Crs", "")
    if "28992" not in content_crs:
        raise RuntimeError(
            f"PDOK provinciegebied did not confirm EPSG:28992 in Content-Crs "
            f"(got {content_crs!r}); refusing to assume the CRS."
        )

    data = resp.json()
    feats = data.get("features", [])
    if len(feats) != 1:
        raise RuntimeError(
            f"Expected exactly 1 province feature for "
            f"{rcfg['province_name_field']}={rcfg['province']!r}, got {len(feats)}."
        )

    code = feats[0]["properties"].get(rcfg["province_code_field"])
    if str(code) != str(rcfg["province_expected_code"]):
        raise RuntimeError(
            f"Province code mismatch: expected {rcfg['province_expected_code']!r}, "
            f"got {code!r}. Not proceeding on an unverified match."
        )

    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:28992")

    processed_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GPKG")
    print(f"Wrote {cache_path}")
    return gdf


if __name__ == "__main__":
    cfg = load_config()
    fetch_province_boundary(cfg)
