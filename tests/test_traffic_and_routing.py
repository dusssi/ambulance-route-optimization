from __future__ import annotations

import math

import pytest

from src.routing.algorithms import astar, compare_algorithms, dijkstra
from src.routing.graph import RoadEdge, adjusted_edge_cost, build_weighted_graph
from src.routing.scenarios import SCENARIOS, apply_edge_load_override, scenario_edges
from src.traffic.estimation import estimate_traffic_load


def test_traffic_proxy_is_bounded_and_reproducible() -> None:
    assert estimate_traffic_load(0).load_fraction == 0.0
    assert estimate_traffic_load(7, 20).load_fraction == 0.35
    assert estimate_traffic_load(20, 20).load_fraction == 1.0
    assert estimate_traffic_load(100, 20) == estimate_traffic_load(100, 20)
    assert 0 <= estimate_traffic_load(10_000).load_fraction <= 1


@pytest.mark.parametrize("count,saturation", [(-1, 20), (1, 0), (True, 20), (1, False), (1.2, 20)])
def test_traffic_proxy_rejects_invalid_parameters(count: object, saturation: object) -> None:
    with pytest.raises(ValueError):
        estimate_traffic_load(count, saturation)  # type: ignore[arg-type]


def test_traffic_adjusted_edge_cost_and_parameter_validation() -> None:
    assert adjusted_edge_cost(10, 0.5, 2) == 20
    assert adjusted_edge_cost(0, 1, 3) == 0
    for args in [(-1, 0.2, 1), (2, -0.1, 1), (2, 1.1, 1), (2, 0.2, -1), (math.inf, 0, 1)]:
        with pytest.raises(ValueError):
            adjusted_edge_cost(*args)


def test_dijkstra_and_zero_heuristic_astar_agree_on_optimal_cost() -> None:
    graph = build_weighted_graph([
        RoadEdge("S", "A", 1),
        RoadEdge("A", "T", 5),
        RoadEdge("S", "B", 3),
        RoadEdge("B", "T", 3),
        RoadEdge("A", "B", 1),
    ])
    shortest = dijkstra(graph, "S", "T")
    informed = astar(graph, "S", "T")
    assert shortest.found and informed.found
    assert shortest.total_cost == informed.total_cost == 5
    assert shortest.path == informed.path
    assert shortest.expanded_nodes >= 1


def test_supplied_scenarios_genuinely_change_route() -> None:
    baseline = build_weighted_graph(scenario_edges(SCENARIOS[0]), alpha=1.0)
    congested = build_weighted_graph(scenario_edges(SCENARIOS[1]), alpha=1.0)
    baseline_result = compare_algorithms(baseline, "Ambulance Base", "Emergency Site")
    congested_result = compare_algorithms(congested, "Ambulance Base", "Emergency Site")
    assert baseline_result.dijkstra.path == ("Ambulance Base", "Junction A", "Emergency Site")
    assert congested_result.dijkstra.path == ("Ambulance Base", "Junction B", "Emergency Site")
    assert baseline_result.dijkstra.total_cost == baseline_result.astar.total_cost
    assert congested_result.dijkstra.total_cost == congested_result.astar.total_cost


def test_computed_image_proxy_can_be_assigned_to_edge_and_change_route() -> None:
    # Synthetic count input verifies the integration formula only; it is not a
    # real detector output or dataset measurement.
    image_proxy = estimate_traffic_load(20, saturation_count=20)
    roads = scenario_edges(SCENARIOS[0])
    adjusted_roads = apply_edge_load_override(
        roads, "Junction A", "Emergency Site", image_proxy.load_fraction
    )
    graph = build_weighted_graph(adjusted_roads, alpha=1.0)
    result = dijkstra(graph, "Ambulance Base", "Emergency Site")
    assert result.path == ("Ambulance Base", "Junction B", "Emergency Site")
    assert result.total_cost == 9.0
    assert next(edge for edge in roads if edge.source == "Junction A" and edge.target == "Emergency Site").traffic_load == 0.0


def test_unreachable_destination_and_source_equals_destination() -> None:
    graph = build_weighted_graph([RoadEdge("A", "B", 2), RoadEdge("C", "D", 3)])
    result = dijkstra(graph, "A", "D")
    assert not result.found and result.path == () and result.total_cost is None
    same = astar(graph, "A", "A")
    assert same.found and same.path == ("A",) and same.total_cost == 0
    assert same.expanded_nodes == 1


def test_invalid_nodes_edges_and_directionality() -> None:
    with pytest.raises(ValueError, match="source node"):
        dijkstra(build_weighted_graph([RoadEdge("A", "B", 1)]), "missing", "B")
    with pytest.raises(ValueError, match="nonnegative"):
        build_weighted_graph([RoadEdge("A", "B", -1)])
    with pytest.raises(ValueError, match="duplicate"):
        build_weighted_graph([RoadEdge("A", "B", 1), RoadEdge("B", "A", 2)])
    directed = build_weighted_graph([RoadEdge("A", "B", 1)], directed=True)
    assert dijkstra(directed, "B", "A").found is False
