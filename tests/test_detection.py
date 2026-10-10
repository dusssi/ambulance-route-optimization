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


def test_yolo_result_conversion_filters_invalid_boxes_and_records_inference_settings() -> None:
    from types import SimpleNamespace

    from src.detection.yolo import DEFAULT_IMAGE_SIZE, DEFAULT_MAX_DETECTIONS, DEFAULT_NMS_IOU

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeBoxes:
        xyxy = FakeTensor([[1.0, 2.0, 10.0, 15.0], [0.0, 0.0, 101.0, 20.0], [2.0, 2.0, 8.0, 8.0]])
        conf = FakeTensor([0.9, 0.8, 0.95])
        cls = FakeTensor([2.0, 2.0, 3.0])

        def __len__(self):
            return 3

    class FakeModel:
        names = {2: "car", 3: "person"}

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(boxes=FakeBoxes())]

    detector = YOLOVehicleDetector("synthetic.pt")
    detector._model = FakeModel()
    detector._names = {2: "car", 3: "person"}
    detector._vehicle_ids = {2: "car"}
    run = detector.detect(Image.new("RGB", (100, 80)), "key", "road.jpg", "BDD100K", 0.25)

    assert len(run.detections) == 1
    assert run.detections[0].box_xyxy == (1.0, 2.0, 10.0, 15.0)
    assert run.image_size == (100, 80)
    assert detector._model.kwargs["conf"] == 0.25
    assert detector._model.kwargs["classes"] == [2]
    assert detector._model.kwargs["imgsz"] == DEFAULT_IMAGE_SIZE
    assert detector._model.kwargs["iou"] == DEFAULT_NMS_IOU
    assert detector._model.kwargs["max_det"] == DEFAULT_MAX_DETECTIONS


def test_unsupported_model_classes_and_invalid_confidence_are_actionable() -> None:
    with pytest.raises(DetectionError, match="no class names matching"):
        verified_vehicle_mapping({0: "person", 1: "bicycle"})
    detector = YOLOVehicleDetector("not-loaded.pt")
    with pytest.raises(ValueError, match="Confidence threshold"):
        detector.detect(Image.new("RGB", (10, 10)), "key", "image.jpg", "upload", 1.1)
