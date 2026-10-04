"""CSV serialization for computed detections and available evaluation results."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping

from src.detection.models import DetectionRun
from src.evaluation.detection_metrics import ImageEvaluation

DETECTION_COLUMNS = (
    "dataset", "image_name", "model_source", "confidence_threshold", "class_id",
    "model_class_name", "vehicle_class", "confidence", "x1", "y1", "x2", "y2",
)
EVALUATION_COLUMNS = (
    "dataset", "image_name", "sample_size", "confidence_threshold", "iou_threshold",
    "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1",
    "predicted_vehicle_count", "ground_truth_vehicle_count", "vehicle_count_mae",
    "category_mapping", "matching_rule", "notes",
)


def _to_csv(rows: Iterable[Mapping[str, object]], columns: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def detections_to_csv(run: DetectionRun) -> str:
    """Export one row per actual detected bounding box (header even if empty)."""
    return _to_csv(run.csv_rows(), DETECTION_COLUMNS)


def evaluation_to_csv(evaluation: ImageEvaluation, dataset_name: str) -> str:
    """Export one row for one actually evaluated, annotation-matched image."""
    row = {field: getattr(evaluation, field) for field in EVALUATION_COLUMNS if hasattr(evaluation, field)}
    row["dataset"] = dataset_name
    return _to_csv((row,), EVALUATION_COLUMNS)


def write_csv(path: str | Path, csv_text: str) -> Path:
    """Write an explicitly requested CSV under a caller-chosen local path."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(csv_text, encoding="utf-8", newline="")
    return destination
