"""Validated weighted road graph and traffic-adjusted edge costs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RoadEdge:
    source: str
    target: str
    base_cost: float
    traffic_load: float = 0.0


@dataclass(frozen=True)
class AdjustedRoadEdge:
    source: str
    target: str
    base_cost: float
    traffic_load: float
    adjusted_cost: float


@dataclass(frozen=True)
class WeightedGraph:
    adjacency: dict[str, tuple[tuple[str, float], ...]]
    edges: tuple[AdjustedRoadEdge, ...]
    directed: bool
    alpha: float

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self.adjacency))

    def neighbors(self, node: str) -> tuple[tuple[str, float], ...]:
        return self.adjacency.get(node, ())


def adjusted_edge_cost(base_cost: float, traffic_load: float, alpha: float = 1.0) -> float:
    """Compute ``base_cost * (1 + alpha * traffic_load)`` with strict checks."""
    values = (base_cost, traffic_load, alpha)
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise ValueError("base cost, traffic load, and alpha must be finite numbers")
    base, load, penalty = (float(value) for value in values)
    if base < 0:
        raise ValueError("base edge cost must be nonnegative for Dijkstra/A*")
    if not 0.0 <= load <= 1.0:
        raise ValueError("traffic load must be within [0, 1]")
    if penalty < 0:
        raise ValueError("alpha must be nonnegative")
    result = base * (1.0 + penalty * load)
    if not math.isfinite(result):
        raise ValueError("adjusted edge cost is not finite")
    return result


def build_weighted_graph(
    edges: Iterable[RoadEdge],
    alpha: float = 1.0,
    directed: bool = False,
) -> WeightedGraph:
    """Build a deterministic graph; duplicate edges and invalid costs are errors.

    Edges are undirected by default. Nodes are created from edge endpoints; an
    isolated node is not representable by an edge table and is therefore not
    available for routing.
    """
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be a finite, nonnegative number")
    alpha = float(alpha)
    adjusted_edges: list[AdjustedRoadEdge] = []
    adjacency: dict[str, list[tuple[str, float]]] = {}
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges, 1):
        source = str(edge.source).strip()
        target = str(edge.target).strip()
        if not source or not target:
            raise ValueError(f"edge {index} must have non-empty source and target nodes")
        if source == target:
            raise ValueError(f"edge {index} cannot connect a node to itself")
        key = (source, target) if directed else tuple(sorted((source, target)))
        if key in seen:
            raise ValueError(f"duplicate road edge: {source!r} to {target!r}")
        seen.add(key)
        cost = adjusted_edge_cost(edge.base_cost, edge.traffic_load, alpha)
        adjusted = AdjustedRoadEdge(source, target, float(edge.base_cost), float(edge.traffic_load), cost)
        adjusted_edges.append(adjusted)
        adjacency.setdefault(source, []).append((target, cost))
        adjacency.setdefault(target, [])
        if not directed:
            adjacency[target].append((source, cost))

    stable_adjacency = {
        node: tuple(sorted(neighbors, key=lambda item: (item[0], item[1])))
        for node, neighbors in sorted(adjacency.items())
    }
    return WeightedGraph(
        adjacency=stable_adjacency,
        edges=tuple(adjusted_edges),
        directed=bool(directed),
        alpha=alpha,
    )
