
from typing import Literal

TrafficLevel = Literal["Low", "Medium", "High"]


def estimate_density(vehicle_count: int) -> TrafficLevel:
    """Classify traffic based on the number of detected vehicles."""
    if vehicle_count < 0:
        raise ValueError("Vehicle count cannot be negative.")

    if vehicle_count <= 5:
        return "Low"
    elif vehicle_count <= 15:
        return "Medium"
    else:
        return "High"
