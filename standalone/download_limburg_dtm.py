#!/usr/bin/env python3
"""Download the AHN elevation model for a Dutch province, as one clipped GeoTIFF.

Standalone - no project code needed. Lives in limburg/standalone/ specifically
because of that: it's the one script in this directory that doesn't import
anything from the limburg package, so it's kept out of the package proper.
Defaults to Limburg at 5 m.

    python limburg/standalone/download_limburg_dtm.py                    # Limburg, DTM, 5 m
    python limburg/standalone/download_limburg_dtm.py --res 1            # 1 m  (~8.8 GB for Limburg)
    python limburg/standalone/download_limburg_dtm.py --res 25           # quick context map
    python limburg/standalone/download_limburg_dtm.py --province Gelderland
    python limburg/standalone/download_limburg_dtm.py --coverage dsm_05m # surface, not terrain
    python limburg/standalone/download_limburg_dtm.py --estimate-only    # print sizes and stop

Requires: requests, numpy, rasterio, geopandas, shapely, matplotlib.
    pip install requests numpy rasterio geopandas shapely matplotlib

What it does
------------
1. Takes the province boundary from PDOK Bestuurlijke Gebieden, so the clip is
   the legal province and not a hand-drawn box.
2. Tiles the bounding box and pulls each tile from the PDOK AHN WCS, which
   honours the WCS 2.0 scaling extension - &scalesize=x(N),y(M) returns an
   N x M grid resampled server-side. So any resolution costs one request per
   tile instead of a tile pyramid download.
3. Mosaics, clips to the boundary, writes a tiled GeoTIFF with overviews.

Two things worth knowing before you trust the output
----------------------------------------------------
THE SERVER MAY NOT GIVE YOU THE EXTENT YOU ASKED FOR. Where a request runs off
the edge of AHN's coverage - the Belgian and German borders, the coast - the
service clips it, and returns the cell count you asked for over a SMALLER
extent. Ask for y(300000,310000) at 400 x 400 and you can get y(306250,310000)
at 400 x 400. Placing that tile by its index puts South Limburg kilometres from
where it belongs and nothing warns you. This script therefore reads each tile's
own transform and places it by coordinates. If you write your own version, this
is the bug to avoid.

NODATA IS WATER. AHN is lidar, and a water surface returns nothing, so the Maas,
the Maasplassen and the quarry lakes come back empty. For Limburg that is about
5% of the province, and the share GROWS at finer resolution because a coarse
cell averages a lake edge into the land around it. Empty water is correct
output, not a failed download.

Elevations are metres above NAP, in EPSG:28992 (RD New).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling as RioResampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

import matplotlib
matplotlib.use("Agg")  # headless - this script only ever writes a PNG file
import matplotlib.pyplot as plt

AHN_WCS = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
BG_WFS = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"
RD = "EPSG:28992"
LIMBURG_DIR = Path(__file__).resolve().parents[1]  # this file lives in limburg/standalone/

log = lambda *a: print(*a, flush=True)


def plot_png(mosaic, geom, transform, stem: str, out_dir: Path, province: str, res: float, coverage: str) -> Path:
    """PNG of the clipped mosaic - nodata (water, per the module docstring)
    masked transparent, province boundary overlaid. Written to
    limburg/outputs/figures/ (anchored to this script's own location, not
    --out/cwd, so it lands in the same place regardless of where this is
    invoked from) - same convention the rest of this project's figures use."""
    fig_dir = LIMBURG_DIR / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    valid = mosaic[np.isfinite(mosaic)]
    masked = np.ma.masked_invalid(mosaic)
    left, top = transform * (0, 0)
    right, bottom = transform * (mosaic.shape[1], mosaic.shape[0])

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(
        masked, cmap=cmap, extent=(left, right, bottom, top),
        vmin=float(valid.min()), vmax=float(valid.max()),
    )
    gpd.GeoSeries([geom], crs=RD).boundary.plot(ax=ax, color="#333333", linewidth=0.8, zorder=2)

    kind = "DSM" if "dsm" in coverage else "DTM"
    ax.set_title(f"{province} - AHN {kind} ({res:g} m/px, clipped to province boundary)")
    ax.set_xlabel("Easting (m, RD New)")
    ax.set_ylabel("Northing (m, RD New)")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Elevation (m NAP)", rotation=270, labelpad=15)

    png_path = fig_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {png_path}")
    return png_path


def province_boundary(name: str, cache: Path) -> "gpd.GeoDataFrame":
    cp = cache / f"province_{name.lower().replace(' ', '_')}.gpkg"
    if cp.exists():
        return gpd.read_file(cp)
    log(f"fetching {name} boundary from PDOK Bestuurlijke Gebieden")
    js = requests.get(BG_WFS, params={
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": "bestuurlijkegebieden:Provinciegebied",
        "outputFormat": "application/json", "srsName": RD, "count": 50,
    }, timeout=300).json()
    g = gpd.GeoDataFrame.from_features(js["features"], crs=RD)
    col = next((c for c in g.columns if "naam" in c.lower()), None)
    if col is None:
        raise RuntimeError(f"no name column in {list(g.columns)}")
    hit = g[g[col].str.contains(name, case=False, na=False)].reset_index(drop=True)
    if not len(hit):
        raise SystemExit(f"province {name!r} not found. Available: "
                         f"{sorted(g[col].dropna().unique())}")
    hit.to_file(cp, driver="GPKG")
    return hit


def fetch_tile(bbox, nx, ny, coverage, cache: Path, retries: int = 3):
    """One WCS tile. Returns (array, transform, bounds) AS THE SERVER GAVE IT."""
    x0, y0, x1, y1 = bbox
    url = (f"{AHN_WCS}?service=WCS&version=2.0.1&request=GetCoverage"
           f"&coverageId={coverage}&format=image/tiff"
           f"&subset=x({x0:.0f},{x1:.0f})&subset=y({y0:.0f},{y1:.0f})"
           f"&scalesize=x({nx}),y({ny})")
    cp = cache / f"ahn_{hashlib.sha1(url.encode()).hexdigest()[:16]}.tif"
    if not cp.exists():
        last = None
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=900)
                r.raise_for_status()
                if r.content[:2] not in (b"II", b"MM"):
                    raise RuntimeError(f"not a GeoTIFF: {r.content[:200]!r}")
                tmp = cp.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(cp)          # atomic, so a kill never leaves a half tile
                break
            except Exception as e:                        # noqa: BLE001
                last = e
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"tile failed after {retries} tries: {last}")
    with rasterio.open(cp) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a = np.where(np.isclose(a, s.nodata), np.nan, a)
        # AHN ships huge sentinel values for nodata in some tiles
        a = np.where((a > 1e30) | (a < -1e4), np.nan, a)
        return a, s.transform, s.bounds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--province", default="Limburg")
    ap.add_argument("--res", type=float, default=5.0,
                    help="output cell size in metres (default 5)")
    ap.add_argument("--coverage", default="dtm_05m", choices=["dtm_05m", "dsm_05m"],
                    help="dtm_05m = bare earth (default), dsm_05m = top of everything")
    ap.add_argument("--tile-km", type=float, default=10.0)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--cache", default=None, help="tile cache dir (default <out>/.ahn_cache)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the estimate exceeds 20 GB")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache).expanduser() if args.cache else out_dir / ".ahn_cache"
    cache.mkdir(parents=True, exist_ok=True)

    prov = province_boundary(args.province, cache)
    geom = (prov.geometry.union_all() if hasattr(prov.geometry, "union_all")
            else prov.geometry.unary_union)
    x0, y0, x1, y1 = geom.bounds
    area_km2 = geom.area / 1e6
    log(f"\n{args.province}: {area_km2:,.0f} km2 land")
    log(f"  bbox RD ({x0:,.0f}, {y0:,.0f}) - ({x1:,.0f}, {y1:,.0f})"
        f"  [{(x1-x0)/1000:.0f} x {(y1-y0)/1000:.0f} km]")

    log("\n  size at each resolution (float32, before compression):")
    for r in (0.5, 1.0, 2.0, 5.0, 25.0):
        cells = geom.area / (r * r)
        log(f"    {r:>5} m  {cells/1e9:8.2f} G cells  {cells*4/1e9:8.1f} GB"
            + ("   <- requested" if abs(r - args.res) < 1e-9 else ""))

    res = args.res
    est_gb = (geom.area / (res * res)) * 4 / 1e9
    log(f"\n  requested {res:g} m -> about {est_gb:.1f} GB uncompressed, "
        f"typically {est_gb*0.3:.1f}-{est_gb*0.5:.1f} GB on disk after DEFLATE")
    if args.estimate_only:
        return 0
    if est_gb > 20 and not args.force:
        log(f"\nREFUSING: {est_gb:.0f} GB exceeds the 20 GB guard. Use a coarser "
            f"--res, or --force if you know the disk can take it.")
        return 2

    tile_m = args.tile_km * 1000.0
    X0 = np.floor(x0 / tile_m) * tile_m
    Y0 = np.floor(y0 / tile_m) * tile_m
    X1 = np.ceil(x1 / tile_m) * tile_m
    Y1 = np.ceil(y1 / tile_m) * tile_m
    nx_tot = int(round((X1 - X0) / res))
    ny_tot = int(round((Y1 - Y0) / res))
    transform = from_origin(X0, Y1, res, res)
    log(f"\nmosaic {nx_tot:,} x {ny_tot:,} cells "
        f"({nx_tot*ny_tot*4/1e9:.1f} GB in memory)")
    mosaic = np.full((ny_tot, nx_tot), np.nan, "float32")

    xs = np.arange(X0, X1, tile_m)
    ys = np.arange(Y0, Y1, tile_m)
    todo = [(tx, ty) for ty in ys for tx in xs
            if geom.intersects(box(tx, ty, tx + tile_m, ty + tile_m))]
    log(f"{len(todo)} tiles of {args.tile_km:g} km touch the province "
        f"(of {len(xs)*len(ys)} in the bbox)\n")

    n_cells_per_tile = int(round(tile_m / res))
    failed, t_start = [], time.time()
    for k, (tx, ty) in enumerate(todo, 1):
        bb = (tx, ty, tx + tile_m, ty + tile_m)
        try:
            a, src_tr, src_b = fetch_tile(bb, n_cells_per_tile, n_cells_per_tile,
                                          args.coverage, cache)
        except Exception as e:                            # noqa: BLE001
            log(f"  tile {k}/{len(todo)} FAILED: {type(e).__name__}: {str(e)[:120]}")
            failed.append(bb)
            continue
        if not np.isfinite(a).any():
            continue                                      # entirely outside NL

        # Place by the server's own georeferencing, into the window it covers.
        c0 = max(int(np.floor((src_b.left - X0) / res)), 0)
        r0 = max(int(np.floor((Y1 - src_b.top) / res)), 0)
        w = min(int(np.ceil((src_b.right - src_b.left) / res)), nx_tot - c0)
        h = min(int(np.ceil((src_b.top - src_b.bottom) / res)), ny_tot - r0)
        if w <= 0 or h <= 0:
            continue
        patch = np.full((h, w), np.nan, "float32")
        reproject(source=a, destination=patch,
                  src_transform=src_tr, src_crs=RD,
                  dst_transform=from_origin(X0 + c0 * res, Y1 - r0 * res, res, res),
                  dst_crs=RD, src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        win = mosaic[r0:r0 + h, c0:c0 + w]
        np.copyto(win, patch, where=np.isfinite(patch))

        if k % 10 == 0 or k == len(todo):
            el = time.time() - t_start
            log(f"  tile {k}/{len(todo)}  {np.isfinite(mosaic).mean():5.1%} of grid "
                f"filled  [{el:.0f}s, ~{el/k*(len(todo)-k):.0f}s left]")

    if failed:
        log(f"\n{len(failed)} tile(s) failed. Re-run to retry only those - good "
            f"tiles are cached in {cache}")

    log("\nclipping to the province boundary")
    outside = geometry_mask([geom], out_shape=mosaic.shape, transform=transform,
                            invert=False)
    mosaic[outside] = np.nan
    finite = np.isfinite(mosaic)
    n_ok = int(finite.sum())
    cov = n_ok * res * res / geom.area
    lo = float(np.nanmin(mosaic)) if n_ok else float("nan")
    hi = float(np.nanmax(mosaic)) if n_ok else float("nan")
    log(f"  {n_ok:,} valid cells ({n_ok*res*res/1e6:,.0f} km2), "
        f"{lo:.2f} to {hi:.2f} m NAP")
    log(f"  coverage {cov:.1%} of province area "
        f"({(1-cov)*area_km2:,.0f} km2 nodata - this is open water, see the docstring)")

    # --- QA, as assertions. Each one has caught something real.
    assert n_ok > 0, "empty mosaic - every tile failed"
    assert -15.0 <= lo and hi <= 400.0, (
        f"elevation out of plausible NL range: {lo:.1f} to {hi:.1f} m NAP")
    assert cov > 0.90, (
        f"only {cov:.1%} of the province has data. Nodata is water and should be "
        f"a few percent; a big shortfall means tiles are missing or misplaced.")
    if not failed and args.province.lower() == "limburg":
        # Vaalserberg, 322.4 m, the highest point in the Netherlands. If the
        # southern tiles were misplaced this is the number that goes wrong.
        assert hi > 300, (f"max elevation {hi:.1f} m but Limburg contains the "
                          f"Vaalserberg at 322 m - South Limburg is missing")
        log(f"  QA: highest point {hi:.2f} m NAP (Vaalserberg is 322.4 m) - "
            f"South Limburg present")

    stem = f"{args.province.lower().replace(' ', '_')}_ahn_" \
           f"{'dsm' if 'dsm' in args.coverage else 'dtm'}_{res:g}m"
    tif = out_dir / f"{stem}.tif"
    log(f"\nwriting {tif}")
    with rasterio.open(
            tif, "w", driver="GTiff", height=ny_tot, width=nx_tot, count=1,
            dtype="float32", crs=RD, transform=transform, nodata=np.nan,
            compress="deflate", predictor=3, tiled=True, blockxsize=512,
            blockysize=512, BIGTIFF="IF_SAFER") as dst:
        dst.write(mosaic, 1)
        dst.build_overviews([2, 4, 8, 16, 32], RioResampling.average)
        dst.update_tags(
            source=f"PDOK AHN WCS {args.coverage}", crs="EPSG:28992",
            units="m NAP", resampling="server-side via WCS scalesize",
            clip=f"PDOK Bestuurlijke Gebieden, Provinciegebied {args.province}",
            nodata_meaning="water surface and filtered points, not missing data")
    size_mb = tif.stat().st_size / 1e6
    log(f"  {size_mb:,.0f} MB on disk")

    plot_png(mosaic, geom, transform, stem, out_dir, args.province, res, args.coverage)

    prov.to_file(out_dir / f"{stem.rsplit('_', 2)[0]}_boundary.gpkg", driver="GPKG")
    json.dump({
        "province": args.province, "coverage": args.coverage, "res_m": res,
        "crs": RD, "units": "m NAP",
        "bounds_rd": [X0, Y0, X1, Y1], "shape": [ny_tot, nx_tot],
        "valid_cells": n_ok, "coverage_frac": round(cov, 4),
        "elev_min_m": round(lo, 3), "elev_max_m": round(hi, 3),
        "tiles_requested": len(todo), "tiles_failed": len(failed),
        "file_mb": round(size_mb, 1),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, open(out_dir / f"{stem}_manifest.json", "w"), indent=2)
    log(f"  wrote {out_dir / (stem + '_manifest.json')}")
    log("\ndone" + (f" WITH {len(failed)} FAILED TILE(S) - re-run to fill them"
                    if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
