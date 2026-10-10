"""Fixed-threshold categorical companion to the continuous traffic-load proxy.

This module labels a supported-vehicle count as Low/Medium/High for display.
The thresholds are transparent prototype design choices, exactly like the
saturation count in ``src.traffic.estimation``; they are not calibrated
against road capacity, camera field-of-view, speed, queue length, or travel
time. Routing edge costs use only the continuous ``[0, 1]`` proxy from
``src.traffic.estimation`` -- never this categorical label -- so the two
modules are complementary views of one count, not duplicate estimators.
"""
from typing import Literal

TrafficLevel = Literal["Low", "Medium", "High"]

# Illustrative design choices (documented in README), not calibrated limits.
LOW_MAX_COUNT = 5
MEDIUM_MAX_COUNT = 15


def estimate_density(vehicle_count: int) -> TrafficLevel:
    """Classify traffic based on the number of detected vehicles.

    Counts up to ``LOW_MAX_COUNT`` are Low, up to ``MEDIUM_MAX_COUNT`` are
    Medium, and anything above is High. The same integer validation as
    ``estimate_traffic_load`` applies so both views of one count reject the
    same invalid inputs.
    """
    if isinstance(vehicle_count, bool) or not isinstance(vehicle_count, int) or vehicle_count < 0:
        raise ValueError("vehicle_count must be a nonnegative integer")

    if vehicle_count <= LOW_MAX_COUNT:
        return "Low"
    elif vehicle_count <= MEDIUM_MAX_COUNT:
        return "Medium"
    else:
        return "High"
