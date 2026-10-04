"""Per-cell runoff coefficient from BOFEK2020's real soil-unit polygons,
classified by the compound leading term in their Dutch OMSCHR description
(see config/limburg_config.yaml's `runoff.texture_keywords` comment for why
bare substring matching would misclassify mixed-profile descriptions).
Unmatched/null-code polygons and any cell outside BOFEK2020 coverage get
`runoff.runoff_coefficient_default`.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from limburg.fetch.bofek2020 import fetch_bofek_clipped


def _classify_texture(description, keyword_pairs: list[list[str]]) -> str | None:
    # pyogrio/pandas represents a null OMSCHR (e.g. open water polygons) as
    # float('nan'), not None or '' - `not description` doesn't catch that
    # (nan is truthy), so check the type explicitly before calling .lower().
    if not isinstance(description, str):
        return None
    text = description.lower()
    best_class, best_pos = None, len(text) + 1
    for phrase, texture_class in keyword_pairs:
        pos = text.find(phrase)
        if pos != -1 and pos < best_pos:
            best_pos, best_class = pos, texture_class
    return best_class


def compute_runoff_coefficient_grid(cfg: dict, boundary: gpd.GeoDataFrame, graph: dict) -> np.ndarray:
    units = fetch_bofek_clipped(cfg, boundary)
    desc_field = cfg["bofek2020"]["description_field"]
    keyword_pairs = cfg["runoff"]["texture_keywords"]
    coefficients = cfg["runoff"]["texture_coefficients"]
    default = cfg["runoff"]["runoff_coefficient_default"]

    units = units.copy()
    units["texture_class"] = units[desc_field].apply(lambda d: _classify_texture(d, keyword_pairs))
    units["runoff_coefficient"] = units["texture_class"].map(coefficients).fillna(default)

    # (no tqdm here - .apply() over a few thousand polygons is sub-second;
    # the print below is the actual debugging signal to watch.) If
    # "(unmatched/null)" dominates this breakdown, texture_keywords in
    # limburg_config.yaml stopped matching real OMSCHR text (e.g. BOFEK2020
    # republished with different wording) - check a raw OMSCHR sample before
    # assuming the runoff coefficients downstream are soil-informed.
    class_counts = units["texture_class"].fillna("(unmatched/null)").value_counts()
    print(f"BOFEK2020 texture classification ({len(units)} units): {class_counts.to_dict()}")

    shapes = [
        (geom, coeff)
        for geom, coeff in zip(units.geometry, units["runoff_coefficient"])
        if geom is not None and not geom.is_empty
    ]
    coeff_grid = rasterize(
        shapes, out_shape=graph["shape"], transform=graph["transform"],
        fill=default, dtype="float32", all_touched=True,
    )
    print(f"Runoff coefficient grid: min={coeff_grid.min():.2f}, max={coeff_grid.max():.2f}, mean={coeff_grid.mean():.3f}")
    return coeff_grid
