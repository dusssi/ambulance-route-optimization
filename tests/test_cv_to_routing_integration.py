"""End-to-end integration: detection vehicle count -> traffic load/density ->
traffic-adjusted edge costs -> Dijkstra/A* on the same graph.

The ``DetectionRun`` objects below are synthetic test doubles (no model weights
or real images); they verify module wiring and arithmetic only, never detector
accuracy or real-world travel times.
"""
from __future__ import annotations

import math

import pytest

from src.detection.models import Detection, DetectionRun
from src.routing.algorithms import compare_algorithms, dijkstra
from src.routing.graph import RoadEdge, build_weighted_graph
from src.routing.scenarios import SCENARIOS, apply_edge_load_override, scenario_edges
from src.traffic import estimate_density, estimate_traffic_load

SATURATION = 20  # Matches the documented default design choice.


def _detection_run(vehicle_count: int) -> DetectionRun:
    detections = tuple(
        Detection(2, "car", "car", 0.9, (float(i), 0.0, float(i) + 10.0, 10.0))
        for i in range(vehicle_count)
    )
    return DetectionRun(
        image_key="synthetic",
        image_name="synthetic.jpg",
        dataset_name="synthetic",
        model_source="synthetic-test-double.pt",
        confidence_threshold=0.25,
        detections=detections,
        counts={"car": vehicle_count, "truck": 0, "bus": 0, "motorcycle": 0},
        model_class_names={2: "car"},
        vehicle_class_ids={2: "car"},
        inference_seconds=0.0,
    )


def _graph_with_image_load(load: float, alpha: float = 1.0):
    roads = apply_edge_load_override(
        scenario_edges(SCENARIOS[0]), "Junction A", "Emergency Site", load
    )
    return build_weighted_graph(roads, alpha=alpha)


def _adjusted_cost(graph, source: str, target: str) -> float:
    edge = next(
        edge
        for edge in graph.edges
        if {edge.source, edge.target} == {source, target}
    )
    return edge.adjusted_cost


def test_detection_count_feeds_load_density_edges_and_both_algorithms() -> None:
    run = _detection_run(20)  # Synthetic test double; not a real YOLO result.
    proxy = estimate_traffic_load(run.vehicle_count, SATURATION)
    assert proxy.load_fraction == 1.0
    assert estimate_density(run.vehicle_count) == "High"

    graph = _graph_with_image_load(proxy.load_fraction)
    assert math.isclose(_adjusted_cost(graph, "Junction A", "Emergency Site"), 12.0)

    comparison = compare_algorithms(graph, "Ambulance Base", "Emergency Site")
    dijkstra_result, astar_result = comparison.dijkstra, comparison.astar
    assert dijkstra_result.found and astar_result.found
    assert dijkstra_result.path == astar_result.path == ("Ambulance Base", "Junction B", "Emergency Site")
    assert math.isclose(dijkstra_result.total_cost, 9.0)
    assert dijkstra_result.total_cost == astar_result.total_cost
    # h(n)=0 is documented: with identical priority keys both algorithms expand
    # the same nodes on the same graph; no speed advantage is claimed or tested.
    assert astar_result.expanded_nodes == dijkstra_result.expanded_nodes >= 1
    assert comparison.dijkstra_seconds >= 0.0 and comparison.astar_seconds >= 0.0
    assert "h(n) = 0" in comparison.heuristic_description


def test_zero_detections_yield_zero_load_low_density_and_baseline_route() -> None:
    run = _detection_run(0)
    assert estimate_traffic_load(run.vehicle_count, SATURATION).load_fraction == 0.0
    assert estimate_density(run.vehicle_count) == "Low"
    graph = _graph_with_image_load(0.0)
    result = dijkstra(graph, "Ambulance Base", "Emergency Site")
    assert result.path == ("Ambulance Base", "Junction A", "Emergency Site")
    assert math.isclose(result.total_cost, 8.0)


def test_partial_load_scales_edge_cost_without_changing_route() -> None:
    run = _detection_run(2)
    proxy = estimate_traffic_load(run.vehicle_count, SATURATION)
    assert proxy.load_fraction == pytest.approx(0.1)
    assert estimate_density(run.vehicle_count) == "Low"
    graph = _graph_with_image_load(proxy.load_fraction)
    assert _adjusted_cost(graph, "Junction A", "Emergency Site") == pytest.approx(6.6)
    result = dijkstra(graph, "Ambulance Base", "Emergency Site")
    assert result.path == ("Ambulance Base", "Junction A", "Emergency Site")
    assert result.total_cost == pytest.approx(8.6)


@pytest.mark.parametrize("count,level", [(0, "Low"), (5, "Low"), (6, "Medium"), (15, "Medium"), (16, "High")])
def test_density_classifier_fixed_thresholds(count: int, level: str) -> None:
    assert estimate_density(count) == level


@pytest.mark.parametrize("count", [-1, True, 2.5, "3"])
def test_density_classifier_rejects_invalid_counts(count: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        estimate_density(count)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_load", [-0.1, 1.5, math.nan, math.inf])
def test_edge_load_override_rejects_out_of_range_loads(bad_load: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_edge_load_override(scenario_edges(SCENARIOS[0]), "Junction A", "Emergency Site", bad_load)


def test_edge_load_override_rejects_missing_and_ambiguous_edges() -> None:
    with pytest.raises(ValueError, match="not present"):
        apply_edge_load_override(scenario_edges(SCENARIOS[0]), "Nowhere", "Junction A", 0.5)
    duplicated = [RoadEdge("A", "B", 1.0), RoadEdge("B", "A", 2.0)]
    with pytest.raises(ValueError, match="ambiguous"):
        apply_edge_load_override(duplicated, "A", "B", 0.5)


def test_traffic_package_exposes_single_implementation_of_each_estimator() -> None:
    import src.traffic as traffic

    assert traffic.estimate_traffic_load is estimate_traffic_load
    assert traffic.estimate_density is estimate_density
    assert traffic.DEFAULT_SATURATION_COUNT == 20
