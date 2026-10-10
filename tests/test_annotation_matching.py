"""Regression tests for annotation-identifier-to-image resolution.

Covers the BDD100K per-image JSON convention where the record ``name`` is an
extensionless image stem (e.g. ``cabc30fc-e7726578`` for ``...jpg``), plus the
exact-filename, relative-path, ambiguity, missing-file, malformed-record, and
duplicate-identifier behaviors that must remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.annotations.bdd100k import parse_bdd100k_json
from src.data.annotations.common import ImageLookup
from src.data.annotations.scan import scan_annotations
from src.data.discovery import discover_dataset
from src.detection.models import Detection, DetectionRun
from src.evaluation.dataset_runner import run_bdd100k_subset


def _make_image(path: Path, size: tuple[int, int] = (100, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(110, 130, 150)).save(path)
    return path


class FakeDetector:
    """Deterministic detector fixture; no model weights or real inference."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect(self, image, image_key, image_name, dataset_name, confidence_threshold):
        self.calls.append(image_name)
        detections = (Detection(2, "car", "car", 0.9, (10.0, 10.0, 30.0, 30.0)),)
        return DetectionRun(
            image_key=image_key,
            image_name=image_name,
            dataset_name=dataset_name,
            model_source="synthetic-test-double.pt",
            confidence_threshold=confidence_threshold,
            detections=detections,
            counts={"car": 1, "truck": 0, "bus": 0, "motorcycle": 0},
            model_class_names={2: "car"},
            vehicle_class_ids={2: "car"},
            inference_seconds=0.01,
        )


# --- ImageLookup unit-level resolution strategies ----------------------------


def test_lookup_resolves_exact_relative_path() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "train" / "road.jpg"])
    resolved, error = lookup.resolve("images/train/road.jpg")
    assert error is None
    assert resolved == (root / "images" / "train" / "road.jpg").resolve()


def test_lookup_resolves_exact_filename_from_any_folder() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "train" / "road.jpg"])
    resolved, error = lookup.resolve("road.jpg")
    assert error is None and resolved is not None
    assert resolved.name == "road.jpg"


def test_lookup_resolves_unique_extensionless_stem() -> None:
    """BDD100K record names omit the image extension and must still resolve."""
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "100k" / "train" / "cabc30fc-e7726578.jpg"])
    resolved, error = lookup.resolve("cabc30fc-e7726578")
    assert error is None
    assert resolved == (root / "images" / "100k" / "train" / "cabc30fc-e7726578.jpg").resolve()


def test_lookup_stem_match_is_case_insensitive() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "Road.JPG"])
    resolved, error = lookup.resolve("ROAD")
    assert error is None and resolved is not None
    assert resolved.name.casefold() == "road.jpg"


def test_lookup_relative_path_with_extra_root_folder_still_matches() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "train" / "road.jpg"])
    resolved, error = lookup.resolve("bdd100k/images/train/road.jpg")
    assert error is None and resolved is not None
    assert resolved.name == "road.jpg"


def test_lookup_rejects_ambiguous_stem_instead_of_guessing() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(
        root,
        [root / "images" / "train" / "road.jpg", root / "images" / "val" / "road.png"],
    )
    resolved, error = lookup.resolve("road")
    assert resolved is None
    assert error is not None and "ambiguous" in error


def test_lookup_reports_missing_image_with_original_identifier() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "road.jpg"])
    resolved, error = lookup.resolve("does-not-exist-1234")
    assert resolved is None
    assert error is not None
    assert "does-not-exist-1234" in error


def test_lookup_rejects_empty_identifier() -> None:
    root = Path("/tmp/fake-root").resolve()
    lookup = ImageLookup(root, [root / "images" / "road.jpg"])
    resolved, error = lookup.resolve("   ")
    assert resolved is None
    assert error == "empty image identifier"


def test_lookup_uses_indexed_stems_not_rescan(tmp_path: Path) -> None:
    """Stem resolution must rely on init-time indexes (no per-call scanning)."""
    root = tmp_path / "root"
    _make_image(root / "images" / "a.jpg")
    lookup = ImageLookup(root, discover_dataset("BDD100K", root).image_paths)
    assert lookup.stems == {"a": [(root / "images" / "a.jpg").resolve()]}
    resolved, error = lookup.resolve("a")
    assert error is None and resolved == (root / "images" / "a.jpg").resolve()


# --- Scan-level behavior with BDD100K extensionless identifiers --------------


def _write_extensionless_pair(root: Path, stem: str, objects: list[dict]) -> None:
    _make_image(root / "images" / "100k" / "train" / f"{stem}.jpg", (1280, 720))
    labels = root / "labels" / "100k" / "train" / f"{stem}.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps({
        "name": stem,
        "frames": [{"timestamp": 10000, "objects": objects}],
    }), encoding="utf-8")


def _car_box() -> dict:
    return {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}}


def test_scan_matches_extensionless_bdd_identifier(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "cabc30fc-e7726578", [_car_box()])
    scan = scan_annotations(discover_dataset("BDD100K", root))
    image_path = root / "images" / "100k" / "train" / "cabc30fc-e7726578.jpg"
    assert scan.parsed_formats == {"BDD100K detection JSON": 1}
    assert scan.is_matched(image_path)
    assert scan.has_valid_match(image_path)
    assert scan.unmatched_annotation_names == []
    boxes = scan.boxes_for(image_path)
    assert len(boxes) == 1 and boxes[0].vehicle_class == "car"


def test_subset_runner_evaluates_extensionless_identifier(tmp_path: Path) -> None:
    """Regression: extensionless names must not become annotation_record_mismatch."""
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "cabc30fc-e7726578", [_car_box()])
    detector = FakeDetector()

    result = run_bdd100k_subset(
        root, detector, sample_size=1, seed=42,
        confidence_thresholds=(0.10, 0.15, 0.25), iou_threshold=0.50,
    )

    assert result.sample_size == result.requested_sample_size == 1
    assert result.skipped_by_reason == {}
    assert detector.calls == ["images/100k/train/cabc30fc-e7726578.jpg"]
    evaluation = result.evaluated_images[0].evaluations[2]
    assert evaluation.confidence_threshold == 0.25
    assert evaluation.ground_truth_vehicle_count == 1
    assert set(result.aggregates) == {"0.1", "0.15", "0.25"}


def test_subset_runner_ambiguous_stem_is_never_silently_matched(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_image(root / "images" / "train" / "road.jpg")
    _make_image(root / "extra" / "road.png")
    annotation = root / "labels" / "train" / "road.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(json.dumps({
        "name": "road",
        "frames": [{"objects": [_car_box()]}],
    }), encoding="utf-8")

    pairs, diagnostics = _pairs(root)
    assert pairs == ()
    assert diagnostics["ambiguous_basename_groups"] == 1

    with pytest.raises(ValueError, match="No unique same-stem"):
        run_bdd100k_subset(root, FakeDetector(), sample_size=1)


def _pairs(root: Path):
    from src.evaluation.dataset_runner import discover_bdd100k_image_annotation_pairs

    return discover_bdd100k_image_annotation_pairs(discover_dataset("BDD100K", root))


def test_scan_reports_ambiguous_stem_identifier_without_guessing(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_image(root / "images" / "train" / "road.jpg")
    _make_image(root / "extra" / "road.png")
    labels = root / "labels" / "ann.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps([{
        "name": "road",
        "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 9, "y2": 9}}],
    }]), encoding="utf-8")

    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert "road" in scan.unmatched_annotation_names
    assert not scan.is_matched(root / "images" / "train" / "road.jpg")
    assert not scan.is_matched(root / "extra" / "road.png")
    assert any("ambiguous extensionless image stem" in issue for issue in scan.issues)


# --- Malformed records, duplicates, and empty annotations --------------------


def test_duplicate_identifiers_in_one_file_are_matched_once(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "cabc30fc-e7726578", [_car_box()])
    annotation = root / "labels" / "100k" / "train" / "cabc30fc-e7726578.json"
    records = [json.loads(annotation.read_text(encoding="utf-8"))] * 2
    annotation.write_text(json.dumps(records), encoding="utf-8")

    scan = scan_annotations(discover_dataset("BDD100K", root))
    image_path = root / "images" / "100k" / "train" / "cabc30fc-e7726578.jpg"
    assert scan.is_matched(image_path)
    assert scan.unmatched_annotation_names == []
    # Both duplicate records contribute ground-truth boxes; matching is not lost.
    assert len(scan.boxes_for(image_path)) == 2


def test_subset_runner_rejects_multi_record_per_image_file(tmp_path: Path) -> None:
    """The subset runner requires exactly one record per paired per-image file."""
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "cabc30fc-e7726578", [_car_box()])
    annotation = root / "labels" / "100k" / "train" / "cabc30fc-e7726578.json"
    record = json.loads(annotation.read_text(encoding="utf-8"))
    annotation.write_text(json.dumps([record, record]), encoding="utf-8")

    result = run_bdd100k_subset(root, FakeDetector(), sample_size=1, seed=42)
    assert result.sample_size == 0
    assert result.skipped_by_reason == {"annotation_record_mismatch": 1}
    assert "2 record(s)" in result.skipped_examples[0]


def test_record_without_identifier_is_reported_not_matched(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_image(root / "images" / "road.jpg")
    labels = root / "labels" / "ann.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps([{"frames": [{"objects": [_car_box()]}]}]), encoding="utf-8")

    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.matched_image_paths == set()
    assert any("no valid image identifier" in issue for issue in scan.issues)


def test_empty_frames_array_is_recognized_with_zero_boxes(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "empty-frame", [])
    scan = scan_annotations(discover_dataset("BDD100K", root))
    image_path = root / "images" / "100k" / "train" / "empty-frame.jpg"
    assert scan.is_matched(image_path)
    assert scan.has_valid_match(image_path)
    assert scan.boxes_for(image_path) == []


def test_malformed_json_is_reported_not_matched(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_image(root / "images" / "broken.jpg")
    annotation = root / "labels" / "broken.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text('{"name": "broken.jpg", "frames": [', encoding="utf-8")

    parsed = parse_bdd100k_json(annotation)
    assert not parsed.recognized
    assert any("invalid JSON" in issue for issue in parsed.issues)

    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.matched_image_paths == set()
    assert str(annotation) in scan.unsupported_files


def test_invalid_box_coordinates_mark_image_invalid(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _write_extensionless_pair(root, "bad-box", [{
        "category": "car",
        "box2d": {"x1": 50, "y1": 10, "x2": 5, "y2": 30},
    }])
    scan = scan_annotations(discover_dataset("BDD100K", root))
    image_path = root / "images" / "100k" / "train" / "bad-box.jpg"
    assert scan.is_matched(image_path)
    assert scan.is_invalid(image_path)
    assert not scan.has_valid_match(image_path)
    assert any("invalid box" in issue for issue in scan.issues)


def test_existing_relative_name_behavior_is_unchanged(tmp_path: Path) -> None:
    """Annotations naming images with full relative paths keep working."""
    root = tmp_path / "bdd100k"
    image_path = root / "images" / "train" / "road.jpg"
    _make_image(image_path)
    labels = root / "labels" / "det.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps([{
        "name": "images/train/road.jpg",
        "labels": [{"category": "car", "box2d": {"x1": 2, "y1": 3, "x2": 40, "y2": 31}}],
    }]), encoding="utf-8")

    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.is_matched(image_path)
    assert len(scan.boxes_for(image_path)) == 1
