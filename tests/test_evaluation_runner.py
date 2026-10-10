from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.discovery import discover_dataset
from src.detection.models import Detection, DetectionRun
from src.evaluation.dataset_runner import (
    discover_bdd100k_image_annotation_pairs,
    run_bdd100k_subset,
)


class FakeDetector:
    """Deterministic detector fixture; no model weights or real inference."""

    model_source = "synthetic-test-double.pt"

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def detect(
        self,
        image: Image.Image,
        image_key: str,
        image_name: str,
        dataset_name: str,
        confidence_threshold: float,
    ) -> DetectionRun:
        self.calls.append((image_name, confidence_threshold))
        detections = (
            Detection(2, "car", "car", 0.90, (10.0, 10.0, 30.0, 30.0)),
            Detection(2, "car", "car", 0.70, (10.0, 10.0, 30.0, 30.0)),
        )
        return DetectionRun(
            image_key=image_key,
            image_name=image_name,
            dataset_name=dataset_name,
            model_source=self.model_source,
            confidence_threshold=confidence_threshold,
            detections=detections,
            counts={"car": 2, "truck": 0, "bus": 0, "motorcycle": 0},
            model_class_names={2: "car"},
            vehicle_class_ids={2: "car"},
            inference_seconds=0.01,
        )


def _make_pair(root: Path, name: str, objects: list[dict], *, corrupt_image: bool = False) -> tuple[Path, Path]:
    image = root / "images" / "train" / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if corrupt_image:
        image.write_bytes(b"not a jpeg")
    else:
        Image.new("RGB", (100, 80), color=(70, 80, 90)).save(image)
    annotation = root / "labels" / "train" / f"{name}.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(
        json.dumps({
            "name": f"images/train/{name}.jpg",
            "frames": [{"objects": objects}],
        }),
        encoding="utf-8",
    )
    return image, annotation


def _car_box() -> dict:
    return {
        "category": "car",
        "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
    }


def test_subset_evaluation_compares_thresholds_with_one_inference_per_image(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "road-1", [_car_box()])
    detector = FakeDetector()

    result = run_bdd100k_subset(
        root,
        detector,
        sample_size=1,
        seed=42,
        confidence_thresholds=(0.50, 0.80),
        iou_threshold=0.50,
    )

    assert result.sample_size == result.requested_sample_size == 1
    assert result.candidate_pair_count == result.attempted_pair_count == 1
    assert result.seed == 42
    assert result.iou_threshold == 0.50
    assert len(detector.calls) == 1
    assert detector.calls[0][1] == 0.50  # Inference runs once at the minimum threshold.

    lower, higher = result.evaluated_images[0].evaluations
    assert (lower.true_positives, lower.false_positives, lower.false_negatives) == (1, 1, 0)
    assert lower.precision == 0.5
    assert higher.true_positives == 1 and higher.false_positives == 0
    assert higher.precision == 1.0
    assert result.aggregates["0.5"]["sample_size"] == 1
    assert result.aggregates["0.8"]["true_positives"] == 1


def test_subset_uses_unique_same_stem_for_single_record_without_name(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _image, annotation = _make_pair(root, "road-no-name", [_car_box()])
    annotation.write_text(
        json.dumps({"frames": [{"objects": [_car_box()]}]}),
        encoding="utf-8",
    )
    detector = FakeDetector()

    result = run_bdd100k_subset(root, detector, sample_size=1)

    assert result.sample_size == 1
    assert result.skipped_by_reason == {}
    assert len(detector.calls) == 1
    assert result.evaluated_images[0].evaluations[0].ground_truth_vehicle_count == 1
    assert result.evaluated_images[0].evaluations[0].true_positives == 1


def test_subset_skips_truncated_streamed_json_instead_of_aborting(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    image = root / "images" / "broken.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image)
    annotation = root / "labels" / "broken.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        '[{"name":"broken.jpg","frames":[{"objects":[]}]}, {"name":',
        encoding="utf-8",
    )
    detector = FakeDetector()

    result = run_bdd100k_subset(root, detector, sample_size=1)

    assert result.sample_size == 0
    assert result.skipped_by_reason == {"annotation_unrecognized": 1}
    assert "invalid JSON" in result.skipped_examples[0]
    assert detector.calls == []


def test_subset_sampling_is_repeatable_and_reports_invalid_images(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "a-good", [_car_box()])
    _make_pair(root, "b-corrupt", [_car_box()], corrupt_image=True)
    _make_pair(root, "c-bad-box", [{
        "category": "car",
        "box2d": {"x1": 30, "y1": 10, "x2": 5, "y2": 50},
    }])

    first_detector = FakeDetector()
    second_detector = FakeDetector()
    first = run_bdd100k_subset(root, first_detector, sample_size=1, seed=17)
    second = run_bdd100k_subset(root, second_detector, sample_size=1, seed=17)

    assert [item.image_name for item in first.evaluated_images] == [item.image_name for item in second.evaluated_images]
    assert first.skipped_by_reason == second.skipped_by_reason
    assert first.sample_size == 1
    assert sum(first.skipped_by_reason.values()) == first.attempted_pair_count - first.sample_size


def test_subset_aggregates_micro_detection_counts_and_mean_per_image_count_mae(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "with-car", [_car_box()])
    _make_pair(root, "no-supported-vehicles", [{
        "category": "traffic sign",
        "box2d": {"x1": 1, "y1": 1, "x2": 10, "y2": 12},
    }])

    result = run_bdd100k_subset(
        root,
        FakeDetector(),
        sample_size=2,
        seed=42,
        confidence_thresholds=(0.80,),
    )
    aggregate = result.aggregates["0.8"]

    assert result.sample_size == 2
    assert aggregate["sample_size"] == 2
    assert (aggregate["true_positives"], aggregate["false_positives"], aggregate["false_negatives"]) == (1, 1, 0)
    assert aggregate["precision"] == 0.5
    assert aggregate["recall"] == 1.0
    assert aggregate["f1"] == pytest.approx(2 / 3)
    assert aggregate["vehicle_count_mae"] == 0.5


def test_corrupt_and_malformed_records_are_skipped_with_reasons(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "corrupt", [_car_box()], corrupt_image=True)
    _make_pair(root, "malformed", [{
        "category": "car",
        "box2d": {"x1": 20, "y1": 10, "x2": 5, "y2": 30},
    }])
    detector = FakeDetector()

    result = run_bdd100k_subset(root, detector, sample_size=1, seed=42)

    assert result.sample_size == 0
    assert result.skipped_by_reason == {"corrupt_image": 1, "malformed_annotation": 1}
    assert detector.calls == []


def test_out_of_bounds_ground_truth_is_not_counted_as_a_valid_sample(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "outside", [{
        "category": "car",
        "box2d": {"x1": 1, "y1": 1, "x2": 101, "y2": 20},
    }])
    detector = FakeDetector()

    result = run_bdd100k_subset(root, detector, sample_size=1)

    assert result.sample_size == 0
    assert result.skipped_by_reason == {"ground_truth_box_out_of_bounds": 1}
    assert detector.calls == []


def test_empty_supported_vehicle_annotation_is_a_valid_sample(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "empty-labels", [{
        "category": "traffic sign",
        "box2d": {"x1": 1, "y1": 1, "x2": 10, "y2": 12},
    }])
    result = run_bdd100k_subset(root, FakeDetector(), sample_size=1)
    evaluation = result.evaluated_images[0].evaluations[0]
    assert result.sample_size == 1
    assert evaluation.ground_truth_vehicle_count == 0
    assert evaluation.false_positives == 2
    assert evaluation.false_negatives == 0


def test_pair_discovery_skips_ambiguous_and_unmatched_names(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root, "unique", [_car_box()])
    _make_pair(root, "duplicate", [_car_box()])
    duplicate = root / "extra" / "duplicate.png"
    duplicate.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(duplicate)
    orphan = root / "labels" / "missing.json"
    orphan.write_text("{}", encoding="utf-8")

    pairs, diagnostics = discover_bdd100k_image_annotation_pairs(discover_dataset("BDD100K", root))
    assert len(pairs) == 1
    assert pairs[0].image_path.stem == "unique"
    assert diagnostics["ambiguous_basename_groups"] == 1
    assert diagnostics["json_files_without_matching_image"] == 1
