# limburg

Brown rot (*Ralstonia solanacearum*) surface-water exposure pipeline for
Limburg province seed potato plots. Independent of the main `br_simulation`
pipeline (own AOI, own boundary/plots cache) but reuses its shared plotting
chrome (`br_simulation.visualization.style`).

## Layout

```
limburg/
├── common.py              REPO_ROOT / load_config() - imported by everything below
├── run_limburg.py          entrypoint: runs the full AHN4+Sentinel pipeline end-to-end
├── config/
│   └── limburg_config.yaml every data source, path, and tunable constant
├── boundaries/              province polygon + seed potato plots (fetched once, cached)
│   ├── province.py
│   └── seed_plots.py
├── fetch/                   one module per external data source
│   ├── sentinel1.py
│   ├── sentinel2.py
│   ├── ahn4_dtm.py                province-scoped AHN4 DTM (PDOK WCS)
│   ├── ahn4_dtm_netherlands.py    whole-country AHN4 DTM overview
│   ├── hoogte_nederland_dtm.py    "Hoogte - Land 1 mtr" DTM (different PDOK WCS)
│   ├── copernicus_dem_eea10.py    Copernicus DEM EEA-10 (CDSE, gated CCM access)
│   └── waterways.py               OSM waterway lines (HOTOSM/HDX)
├── process/                  per-scene/per-tile computation steps
│   ├── calibrate_s1.py
│   ├── compute_ndwi.py
│   ├── compute_s1_water.py
│   ├── clip_to_boundary.py        generic raster-to-polygon clip, reused everywhere
│   ├── exposure_overlap.py        plot x SAR-water intersection
│   └── burn_waterways_dtm.py      downsample + ditch-burn + S2-vs-DTM comparison figure
├── mosaic/                   per-source tile merge (+ reprojection where needed)
│   ├── dtm.py
│   ├── ndwi.py
│   ├── s1.py
│   └── s2_visual.py
├── visualize/                 final figures for the main run_limburg.py pipeline
│   ├── limburg_overview.py        3-panel SAR-focused figure
│   └── data_overview.py           4-panel full data-source overview
├── standalone/
│   └── download_limburg_dtm.py    deliberately dependency-free AHN DTM downloader
│                                   (no `limburg` imports - works with just
│                                   requests/numpy/rasterio/geopandas/shapely/matplotlib)
├── data/
│   ├── raw/      per-source cache, one subfolder per fetch/*.py module
│   └── processed/ mosaics, clips, gpkgs - every pipeline step's cached output
└── outputs/
    └── figures/   every PNG this directory produces
```

## Running it

Everything under `boundaries/`, `fetch/`, `process/`, `mosaic/`, `visualize/`,
and `run_limburg.py` is now a proper package (`limburg.*`) using absolute
imports - **run these with `-m` from the repo root** (`br_simulation/`), not
as a bare file path:

```bash
cd /Users/neerajkaroshi/Desktop/Projects/br_simulation

# full pipeline (boundary -> plots -> S2/S1/AHN4 -> mosaics -> overlap -> figures)
python -m limburg.run_limburg

# any individual step, same convention
python -m limburg.fetch.waterways
python -m limburg.process.burn_waterways_dtm
python -m limburg.fetch.hoogte_nederland_dtm
```

`standalone/download_limburg_dtm.py` is the one exception - it has zero
`limburg` imports by design, so it still runs as a plain script, same as
before:

```bash
python limburg/standalone/download_limburg_dtm.py --province Limburg --res 5
```

Every step is independently skip-if-already-done (checked against its own
cached output under `data/` or `outputs/`), so re-running is always safe.
Delete the specific cached file to force recomputation of just that step.
