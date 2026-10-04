"""Transparent illustrative traffic scenarios for the demo road graph."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.routing.graph import RoadEdge


@dataclass(frozen=True)
class TrafficScenario:
    name: str
    description: str
    edge_loads: dict[tuple[str, str], float]


DEMO_EDGES: tuple[tuple[str, str, float], ...] = (
    ("Ambulance Base", "Junction A", 2.0),
    ("Junction A", "Emergency Site", 6.0),
    ("Ambulance Base", "Junction B", 4.0),
    ("Junction B", "Emergency Site", 5.0),
    ("Junction A", "Junction B", 2.5),
)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


SCENARIOS: tuple[TrafficScenario, ...] = (
    TrafficScenario(
        name="Baseline — low illustrative load",
        description="All edges have load 0.0. This is a synthetic baseline, not a measured traffic state.",
        edge_loads={},
    ),
    TrafficScenario(
        name="Illustrative congestion on Junction A–Emergency Site",
        description=(
            "The Junction A–Emergency Site edge has load 1.0; all other edges are 0.0. "
            "This synthetic scenario makes the alternate route cheaper under alpha=1.0."
        ),
        edge_loads={_edge_key("Junction A", "Emergency Site"): 1.0},
    ),
)


def scenario_edges(scenario: TrafficScenario) -> list[RoadEdge]:
    """Return editable base edges initialized with the selected scenario loads."""
    return [
        RoadEdge(
            source=source,
            target=target,
            base_cost=base_cost,
            traffic_load=scenario.edge_loads.get(_edge_key(source, target), 0.0),
        )
        for source, target, base_cost in DEMO_EDGES
    ]


def apply_edge_load_override(
    edges: list[RoadEdge],
    source: str,
    target: str,
    traffic_load: float,
) -> list[RoadEdge]:
    """Return edges with a selected undirected edge assigned one explicit load.

    This supports the dashboard's manual illustrative assignment of a computed
    image proxy to a chosen graph edge. It does not infer a camera/road mapping.
    """
    load = float(traffic_load)
    if not math.isfinite(load) or not 0.0 <= load <= 1.0:
        raise ValueError("image-derived edge load must be finite and within [0, 1]")
    desired = _edge_key(str(source).strip(), str(target).strip())
    matches = [index for index, edge in enumerate(edges) if _edge_key(edge.source, edge.target) == desired]
    if not matches:
        raise ValueError(f"selected edge {source!r} ↔ {target!r} is not present in the graph")
    if len(matches) > 1:
        raise ValueError(f"selected edge {source!r} ↔ {target!r} is ambiguous because it appears more than once")
    updated = list(edges)
    index = matches[0]
    edge = updated[index]
    updated[index] = RoadEdge(edge.source, edge.target, edge.base_cost, load)
    return updated
