from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluate_bdd100k import _output_directory, _write_results
from src.detection.models import Detection, DetectionRun
from src.evaluation.dataset_runner import run_bdd100k_subset


class FakeDetector:
    def detect(self, image, image_key, image_name, dataset_name, confidence_threshold):
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


def _make_pair(root: Path) -> None:
    image = root / "images" / "train" / "road-1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80)).save(image)
    label = root / "labels" / "train" / "road-1.json"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(json.dumps({
        "name": "images/train/road-1.jpg",
        "frames": [{"objects": [{
            "category": "car",
            "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
        }]}],
    }), encoding="utf-8")


def test_cli_artifacts_include_per_image_metrics_and_reproducibility_manifest(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    _make_pair(root)
    result = run_bdd100k_subset(
        root,
        FakeDetector(),
        sample_size=1,
        seed=42,
        confidence_thresholds=(0.10, 0.25),
        iou_threshold=0.50,
    )
    output = tmp_path / "results"
    output.mkdir()

    csv_path, summary_path, manifest_path = _write_results(
        output_dir=output,
        result=result,
        model_source="synthetic-test-double.pt",
        model_class_names={2: "car"},
        vehicle_class_ids={2: "car"},
        device=None,
        runtime={
            "torch": {"version": "test", "cuda_available": False, "device_names": []},
            "ram_bytes": {"MemTotal": 1234},
            "free_disk_bytes": 5678,
        },
        max_annotation_bytes=8 * 1024 * 1024,
    )

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(rows) == 2  # One row per image and threshold.
    assert {row["confidence_threshold"] for row in rows} == {"0.1", "0.25"}
    assert summary["evaluated_sample_size"] == 1
    assert summary["metrics_by_confidence"]["0.1"]["sample_size"] == 1
    assert manifest["sampling"]["seed"] == 42
    assert manifest["model"]["supported_vehicle_ids_from_loaded_checkpoint"] == {"2": "car"}
    assert "unique same-stem image pair" in manifest["dataset"]["annotation_parser"]
    assert "Not mAP" in summary["metric_scope"]


def test_output_directory_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserve previous results"):
        _output_directory(output)
