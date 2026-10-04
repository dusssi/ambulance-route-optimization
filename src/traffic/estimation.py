"""Deterministic image-based traffic-load proxy (not physical density)."""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_SATURATION_COUNT = 20


@dataclass(frozen=True)
class TrafficLoadEstimate:
    vehicle_count: int
    saturation_count: int
    load_fraction: float
    formula: str
    label: str = "Image-based vehicle-count load proxy"


def estimate_traffic_load(
    vehicle_count: int,
    saturation_count: int = DEFAULT_SATURATION_COUNT,
) -> TrafficLoadEstimate:
    """Map a count to [0, 1] using min(vehicle_count / saturation_count, 1).

    ``saturation_count`` is a transparent prototype design choice, not an
    empirically calibrated density threshold. The result depends on one image's
    detections and must not be interpreted as speed, queue length, congestion,
    or travel time.
    """
    if isinstance(vehicle_count, bool) or not isinstance(vehicle_count, int) or vehicle_count < 0:
        raise ValueError("vehicle_count must be a nonnegative integer")
    if isinstance(saturation_count, bool) or not isinstance(saturation_count, int) or saturation_count <= 0:
        raise ValueError("saturation_count must be a positive integer")
    load = min(vehicle_count / saturation_count, 1.0)
    if not math.isfinite(load):  # Defensive; integer inputs make this unlikely.
        raise ValueError("traffic-load calculation produced a non-finite value")
    return TrafficLoadEstimate(
        vehicle_count=vehicle_count,
        saturation_count=saturation_count,
        load_fraction=load,
        formula=f"min(vehicle_count / {saturation_count}, 1) = min({vehicle_count} / {saturation_count}, 1)",
    )
