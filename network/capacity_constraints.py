"""Discharge capacity a cell can carry before it's treated as "wetting"
nearby ground. A single uniform placeholder (config: capacity_constraints.
default_capacity_m3_s) - no verified per-structure culvert capacity rating
exists for any Limburg waterschap (location/geometry only, no discharge
rating), so there's nothing more specific to use yet.
"""
from __future__ import annotations


def get_default_capacity_m3_s(cfg: dict) -> float:
    return float(cfg["capacity_constraints"]["default_capacity_m3_s"])
