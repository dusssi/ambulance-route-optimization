from __future__ import annotations

from PIL import Image
import pytest

from src.detection.yolo import DetectionError, YOLOVehicleDetector, verified_vehicle_mapping


def test_model_class_ids_are_derived_from_loaded_name_mapping() -> None:
    model_names, vehicle_ids = verified_vehicle_mapping({
        5: "person",
        8: "car",
        2: "truck",
        14: "bus",
        21: "motorcycle",
    })
    assert model_names[8] == "car"
    assert vehicle_ids == {8: "car", 2: "truck", 14: "bus", 21: "motorcycle"}


def test_model_class_mapping_accepts_name_lists_and_explicit_aliases() -> None:
    names, vehicles = verified_vehicle_mapping(["person", "motor", "bus"])
    assert names == {0: "person", 1: "motor", 2: "bus"}
    assert vehicles == {1: "motorcycle", 2: "bus"}


def test_unsupported_model_classes_and_invalid_confidence_are_actionable() -> None:
    with pytest.raises(DetectionError, match="no class names matching"):
        verified_vehicle_mapping({0: "person", 1: "bicycle"})
    detector = YOLOVehicleDetector("not-loaded.pt")
    with pytest.raises(ValueError, match="Confidence threshold"):
        detector.detect(Image.new("RGB", (10, 10)), "key", "image.jpg", "upload", 1.1)
