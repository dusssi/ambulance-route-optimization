"""Single-image, class-aware detection metrics for matched ground truth only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from src.data.annotations.common import GroundTruthBox
from src.data.categories import VEHICLE_CLASSES
from src.detection.models import Detection

DEFAULT_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class ImageEvaluation:
    image_name: str
    sample_size: int
    confidence_threshold: float
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    predicted_vehicle_count: int
    ground_truth_vehicle_count: int
    vehicle_count_mae: float
    category_mapping: str
    matching_rule: str
    notes: str


def _validated_box(box, description: str) -> tuple[float, float, float, float]:
    try:
        raw_values = tuple(box)
        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError("boolean coordinates are invalid")
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description} must contain four numeric coordinates") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{description} must contain four finite coordinates")
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{description} must have nonnegative origin and positive width and height")
    return x1, y1, x2, y2


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Compute continuous-coordinate IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = _validated_box(first, "first box")
    bx1, by1, bx2, by2 = _validated_box(second, "second box")
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_image(
    predictions: Iterable[Detection],
    ground_truth: Iterable[GroundTruthBox],
    image_name: str,
    confidence_threshold: float,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    image_size: tuple[int, int] | None = None,
) -> ImageEvaluation:
    """Evaluate one image with confidence filtering and greedy one-to-one IoU.

    A prediction is a true positive if it has the same supported class as an
    unmatched ground-truth box and IoU >= the configured threshold. Predictions
    are considered in descending confidence order. This is per-image precision,
    recall, F1, and supported-class count MAE—not dataset-level mAP.
    """
    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(float(confidence_threshold))
        or not 0.0 <= float(confidence_threshold) <= 1.0
    ):
        raise ValueError("confidence_threshold must be a finite value within [0, 1]")
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not math.isfinite(float(iou_threshold))
        or not 0.0 < float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be a finite value within (0, 1]")
    threshold = float(confidence_threshold)
    iou_limit = float(iou_threshold)
    width: int | None = None
    height: int | None = None
    if image_size is not None:
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
        ):
            raise ValueError("image_size must contain positive integer width and height")
        width, height = image_size

    pred = []
    for item in predictions:
        if item.vehicle_class not in VEHICLE_CLASSES:
            continue
        if (
            isinstance(item.confidence, bool)
            or not isinstance(item.confidence, (int, float))
            or not math.isfinite(float(item.confidence))
            or not 0.0 <= float(item.confidence) <= 1.0
        ):
            raise ValueError("supported predictions must have finite confidence values within [0, 1]")
        box = _validated_box(item.box_xyxy, f"prediction box for {item.vehicle_class}")
        if width is not None and height is not None and (box[2] > width or box[3] > height):
            raise ValueError(f"prediction box for {item.vehicle_class} exceeds image bounds {width}x{height}")
        if item.confidence >= threshold:
            pred.append((item, box))

    truth = []
    for item in ground_truth:
        if item.vehicle_class not in VEHICLE_CLASSES:
            continue
        box = _validated_box(item.box_xyxy, f"ground-truth box for {item.vehicle_class}")
        if width is not None and height is not None and (box[2] > width or box[3] > height):
            raise ValueError(f"ground-truth box for {item.vehicle_class} exceeds image bounds {width}x{height}")
        truth.append((item, box))

    pred.sort(key=lambda pair: (-pair[0].confidence, pair[0].vehicle_class, pair[1]))
    pred_items = [item for item, _ in pred]
    pred_boxes = [box for _, box in pred]
    truth_items = [item for item, _ in truth]
    truth_boxes = [box for _, box in truth]
    matched_truth: set[int] = set()
    true_positives = 0
    false_positives = 0
    for prediction, prediction_box in zip(pred_items, pred_boxes):
        candidates = [
            (intersection_over_union(prediction_box, truth_boxes[index]), index)
            for index, target in enumerate(truth_items)
            if index not in matched_truth and target.vehicle_class == prediction.vehicle_class
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_limit:
            matched_truth.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(truth_items) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else None
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else None
    if precision is not None and recall is not None and precision + recall > 0:
        f1: float | None = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    predicted_count = len(pred_items)
    ground_truth_count = len(truth_items)
    return ImageEvaluation(
        image_name=image_name,
        sample_size=1,
        confidence_threshold=threshold,
        iou_threshold=iou_limit,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_vehicle_count=predicted_count,
        ground_truth_vehicle_count=ground_truth_count,
        vehicle_count_mae=float(abs(predicted_count - ground_truth_count)),
        category_mapping=(
            "Model and annotation category names are normalized through explicit aliases to car, truck, bus, motorcycle; "
            "unmapped classes are excluded."
        ),
        matching_rule=f"Greedy confidence-sorted, class-aware one-to-one matching at IoU >= {iou_limit:.2f}.",
        notes=(
            "One matched image only. This is not a dataset-level metric or mAP. "
            "Count MAE is for the four mapped vehicle classes only."
        ),
    )


def aggregate_evaluations(evaluations):
    """Aggregate image evaluations into dataset-level detection metrics."""
    evaluations = list(evaluations)
    if not evaluations:
        raise ValueError("Cannot aggregate an empty evaluation list")

    tp = sum(item.true_positives for item in evaluations)
    fp = sum(item.false_positives for item in evaluations)
    fn = sum(item.false_negatives for item in evaluations)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall > 0
        else None
    )

    return {
        "sample_size": len(evaluations),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_vehicle_count": sum(
            item.predicted_vehicle_count for item in evaluations
        ),
        "ground_truth_vehicle_count": sum(
            item.ground_truth_vehicle_count for item in evaluations
        ),
        "vehicle_count_mae": sum(
            item.vehicle_count_mae for item in evaluations
        ) / len(evaluations),
    }
