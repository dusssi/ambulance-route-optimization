from __future__ import annotations

import csv
import io

from src.data.annotations.common import GroundTruthBox
from src.detection.models import Detection, DetectionRun
from src.evaluation.detection_metrics import evaluate_image, intersection_over_union
from src.exports.csv_export import (
    DETECTION_COLUMNS,
    EVALUATION_COLUMNS,
    detections_to_csv,
    evaluation_to_csv,
)


def test_iou_and_matched_image_evaluation() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    truth = [GroundTruthBox("car", "car", (0, 0, 10, 10))]
    predictions = [
        Detection(2, "car", "car", 0.9, (0, 0, 10, 10)),
        Detection(7, "truck", "truck", 0.8, (20, 20, 30, 30)),
    ]
    result = evaluate_image(predictions, truth, "a.jpg", confidence_threshold=0.25, iou_threshold=0.5)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.vehicle_count_mae == 1.0
    assert result.sample_size == 1
    assert "not a dataset-level" in result.notes


def _run(detections: tuple[Detection, ...]) -> DetectionRun:
    return DetectionRun(
        image_key="/tmp/image.jpg",
        image_name="image.jpg",
        dataset_name="BDD100K",
        model_source="yolo11n.pt",
        confidence_threshold=0.25,
        detections=detections,
        counts={"car": len(detections), "truck": 0, "bus": 0, "motorcycle": 0},
        model_class_names={2: "car"},
        vehicle_class_ids={2: "car"},
        inference_seconds=0.01,
    )


def test_detection_csv_has_defined_schema_and_actual_values() -> None:
    detection = Detection(2, "car", "car", 0.91, (1.0, 2.0, 30.0, 40.0))
    text = detections_to_csv(_run((detection,)))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0]) == DETECTION_COLUMNS
    assert len(rows) == 1
    assert rows[0]["vehicle_class"] == "car"
    assert float(rows[0]["confidence"]) == 0.91
    assert rows[0]["x2"] == "30.0"


def test_duplicate_predictions_are_one_true_positive_plus_a_false_positive() -> None:
    truth = [GroundTruthBox("car", "car", (0, 0, 10, 10))]
    duplicate_boxes = [
        Detection(2, "car", "car", 0.95, (0, 0, 10, 10)),
        Detection(2, "car", "car", 0.85, (0, 0, 10, 10)),
    ]
    result = evaluate_image(duplicate_boxes, truth, "duplicate.jpg", 0.25, 0.50)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.predicted_vehicle_count == 2


def test_evaluation_rejects_invalid_supported_prediction_boxes() -> None:
    import pytest

    malformed = Detection(2, "car", "car", 0.9, (-1.0, 0.0, 5.0, 5.0))
    with pytest.raises(ValueError, match="nonnegative origin"):
        evaluate_image([malformed], (), "bad-box.jpg", 0.25)


def test_evaluation_checks_box_bounds_when_image_dimensions_are_available() -> None:
    import pytest

    outside = GroundTruthBox("car", "car", (0, 0, 101, 10))
    with pytest.raises(ValueError, match="exceeds image bounds"):
        evaluate_image((), [outside], "outside.jpg", 0.25, image_size=(100, 80))


def test_empty_detection_csv_is_header_only_and_evaluation_csv_is_one_row() -> None:
    empty_text = detections_to_csv(_run(()))
    assert next(csv.reader(io.StringIO(empty_text))) == list(DETECTION_COLUMNS)
    assert len(list(csv.reader(io.StringIO(empty_text)))) == 1
    evaluation = evaluate_image((), (), "image.jpg", 0.25)
    text = evaluation_to_csv(evaluation, "IDD")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0]) == EVALUATION_COLUMNS
    assert len(rows) == 1
    assert rows[0]["dataset"] == "IDD"
    assert rows[0]["sample_size"] == "1"
