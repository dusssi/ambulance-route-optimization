"""Typed results from real object-detector inference."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    model_class_name: str
    vehicle_class: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionRun:
    image_key: str
    image_name: str
    dataset_name: str
    model_source: str
    confidence_threshold: float
    detections: tuple[Detection, ...]
    counts: dict[str, int]
    model_class_names: dict[int, str]
    vehicle_class_ids: dict[int, str]
    inference_seconds: float

    @property
    def vehicle_count(self) -> int:
        return len(self.detections)

    def csv_rows(self) -> list[dict[str, object]]:
        return [
            {
                "dataset": self.dataset_name,
                "image_name": self.image_name,
                "model_source": self.model_source,
                "confidence_threshold": self.confidence_threshold,
                "class_id": item.class_id,
                "model_class_name": item.model_class_name,
                "vehicle_class": item.vehicle_class,
                "confidence": item.confidence,
                "x1": item.box_xyxy[0],
                "y1": item.box_xyxy[1],
                "x2": item.box_xyxy[2],
                "y2": item.box_xyxy[3],
            }
            for item in self.detections
        ]
