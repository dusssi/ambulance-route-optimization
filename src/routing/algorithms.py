"""Dijkstra and A* route search over the same nonnegative weighted graph."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

from src.routing.graph import WeightedGraph


@dataclass(frozen=True)
class RouteResult:
    algorithm: str
    source: str
    destination: str
    path: tuple[str, ...]
    total_cost: float | None
    expanded_nodes: int
    found: bool


@dataclass(frozen=True)
class AlgorithmComparison:
    dijkstra: RouteResult
    astar: RouteResult
    dijkstra_seconds: float
    astar_seconds: float
    heuristic_description: str


def _search(graph: WeightedGraph, source: str, destination: str, algorithm: str) -> RouteResult:
    source = str(source)
    destination = str(destination)
    if source not in graph.adjacency:
        raise ValueError(f"source node {source!r} is not in the graph")
    if destination not in graph.adjacency:
        raise ValueError(f"destination node {destination!r} is not in the graph")
    if source == destination:
        # Convention: a popped/settled node counts as expanded, including the
        # destination; the source-equals-destination case therefore expands 1.
        return RouteResult(algorithm, source, destination, (source,), 0.0, 1, True)

    distances: dict[str, float] = {source: 0.0}
    parents: dict[str, str] = {}
    queue: list[tuple[float, float, str]] = [(0.0, 0.0, source)]
    settled: set[str] = set()
    expanded = 0

    while queue:
        priority, queued_cost, node = heapq.heappop(queue)
        del priority  # A* uses h=0; priority and g are identical here.
        current_cost = distances.get(node, math.inf)
        if queued_cost != current_cost or node in settled:
            continue
        settled.add(node)
        expanded += 1
        if node == destination:
            path = [destination]
            while path[-1] != source:
                path.append(parents[path[-1]])
            path.reverse()
            return RouteResult(algorithm, source, destination, tuple(path), current_cost, expanded, True)

        for neighbor, edge_cost in graph.neighbors(node):
            if edge_cost < 0 or not math.isfinite(edge_cost):
                raise ValueError("routing requires finite, nonnegative edge costs")
            candidate = current_cost + edge_cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                parents[neighbor] = node
                # Zero heuristic is admissible for all nonnegative costs. The
                # extra g field and node name provide stable tie ordering.
                heapq.heappush(queue, (candidate, candidate, neighbor))

    return RouteResult(algorithm, source, destination, (), None, expanded, False)


def dijkstra(graph: WeightedGraph, source: str, destination: str) -> RouteResult:
    """Return a minimum-cost path using Dijkstra's algorithm.

    ``expanded_nodes`` counts each non-stale node removed from the priority
    queue and settled, including the destination if reachable.
    """
    return _search(graph, source, destination, "Dijkstra")


def astar(graph: WeightedGraph, source: str, destination: str) -> RouteResult:
    """Return an optimal path using A* with the admissible zero heuristic.

    The illustrative graph has no calibrated geometric coordinates, so h(n)=0
    is the only generally defensible heuristic here. Search is consequently
    Dijkstra-style uniform-cost search; this is disclosed in the dashboard and
    README rather than implying an A* speed advantage.
    """
    return _search(graph, source, destination, "A*")


def compare_algorithms(graph: WeightedGraph, source: str, destination: str) -> AlgorithmComparison:
    """Measure both algorithms on the exact same already-built graph."""
    start = time.perf_counter()
    dijkstra_result = dijkstra(graph, source, destination)
    dijkstra_seconds = time.perf_counter() - start
    start = time.perf_counter()
    astar_result = astar(graph, source, destination)
    astar_seconds = time.perf_counter() - start
    return AlgorithmComparison(
        dijkstra=dijkstra_result,
        astar=astar_result,
        dijkstra_seconds=dijkstra_seconds,
        astar_seconds=astar_seconds,
        heuristic_description="h(n) = 0 (admissible; A* behaves as Dijkstra-style uniform-cost search)",
    )
