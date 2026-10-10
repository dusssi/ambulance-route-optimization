"""Ultralytics YOLO inference for the four explicitly supported vehicle classes."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

# Keep Ultralytics settings/checkpoints in an ignored local cache, not the
# repository index. Respect an explicit user override.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_ROOT = _PROJECT_ROOT / ".cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_CACHE_ROOT))
if os.environ.get("YOLO_CONFIG_DIR") == str(_DEFAULT_CACHE_ROOT):
    try:
        _DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Ultralytics will fall back to its writable per-user or temporary path.
        pass

from PIL import Image

from src.data.categories import VEHICLE_CLASSES, canonical_vehicle_class
from src.detection.models import Detection, DetectionRun

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEFAULT_NMS_IOU = 0.70
DEFAULT_MAX_DETECTIONS = 300


class DetectionError(RuntimeError):
    """Actionable inference, checkpoint, or model-mapping error."""


def verified_vehicle_mapping(raw_names: Any) -> tuple[dict[int, str], dict[int, str]]:
    """Normalize the actual checkpoint label map and select supported vehicles.

    The integer IDs in the returned map are taken from ``model.names``; they are
    never inferred from a dataset convention or assumed to match COCO IDs.
    """
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        raise DetectionError(
            "The loaded checkpoint does not expose class names, so vehicle class IDs cannot be verified. "
            "Use a supported Ultralytics object-detection checkpoint."
        )
    vehicle_ids = {
        class_id: canonical
        for class_id, name in names.items()
        if (canonical := canonical_vehicle_class(name)) in VEHICLE_CLASSES
    }
    if not vehicle_ids:
        available = ", ".join(f"{key}: {value}" for key, value in sorted(names.items()))
        raise DetectionError(
            "The loaded model has no class names matching car, truck, bus, or motorcycle. "
            f"Available names: {available or 'none'}. Choose compatible pretrained weights."
        )
    return names, vehicle_ids


class YOLOVehicleDetector:
    """Lazy-loading Ultralytics detector; class IDs come from the loaded model."""

    def __init__(self, model_source: str = DEFAULT_MODEL, device: str | None = None):
        self.model_source = str(model_source).strip()
        self.device = device
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        self._vehicle_ids: dict[int, str] = {}
        self.model_load_seconds: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_source:
            raise DetectionError("Enter a model name such as yolo11n.pt or a local checkpoint path.")
        local_path = Path(self.model_source).expanduser()
        if (local_path.is_absolute() or "/" in self.model_source or "\\" in self.model_source) and not local_path.is_file():
            raise DetectionError(
                f"Model checkpoint was not found: {local_path}. Provide a valid local .pt file or use the default "
                f"{DEFAULT_MODEL} and allow its first-run download."
            )
        start = time.perf_counter()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectionError(
                "Ultralytics or one of its native dependencies (PyTorch/OpenCV) could not be imported. "
                "Install the project stack with `./scripts/install_dependencies.sh` and check the reported system-library error. "
                f"Details: {exc}"
            ) from exc
        try:
            model = YOLO(self.model_source)
        except Exception as exc:
            raise DetectionError(
                f"Could not load model {self.model_source!r}. Check the checkpoint path, PyTorch installation, "
                f"and network access for the first-run pretrained-weight download. Details: {exc}"
            ) from exc
        raw_names = getattr(model, "names", None)
        if raw_names is None:
            raw_names = getattr(getattr(model, "model", None), "names", None)
        names, vehicle_ids = verified_vehicle_mapping(raw_names)
        self._model = model
        self._names = names
        self._vehicle_ids = vehicle_ids
        self.model_load_seconds = time.perf_counter() - start

    @property
    def model_class_names(self) -> dict[int, str]:
        self.load()
        return dict(self._names)

    @property
    def vehicle_class_ids(self) -> dict[int, str]:
        self.load()
        return dict(self._vehicle_ids)

    def detect(
        self,
        image: Image.Image,
        image_key: str,
        image_name: str,
        dataset_name: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
    ) -> DetectionRun:
        if not isinstance(confidence_threshold, (int, float)) or not math.isfinite(float(confidence_threshold)):
            raise ValueError("Confidence threshold must be a finite number from 0 to 1.")
        threshold = float(confidence_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be within [0, 1].")
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Inference requires a valid, non-empty Pillow image.")
        self.load()
        assert self._model is not None
        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "source": image.convert("RGB"),
                "conf": threshold,
                "classes": sorted(self._vehicle_ids),
                "imgsz": DEFAULT_IMAGE_SIZE,
                "iou": DEFAULT_NMS_IOU,
                "max_det": DEFAULT_MAX_DETECTIONS,
                "verbose": False,
            }
            if self.device:
                kwargs["device"] = self.device
            result_list = self._model.predict(**kwargs)
        except Exception as exc:
            raise DetectionError(
                f"YOLO inference failed for {image_name!r}. Check image decoding, model compatibility, "
                f"device availability, and memory. Details: {exc}"
            ) from exc
        elapsed = time.perf_counter() - start
        if not result_list:
            results = []
        else:
            results = result_list[0]
        detections: list[Detection] = []
        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy_values = boxes.xyxy.cpu().numpy()
            confidence_values = boxes.conf.cpu().numpy()
            class_values = boxes.cls.cpu().numpy()
            for coordinates, confidence, class_value in zip(xyxy_values, confidence_values, class_values):
                class_id = int(class_value)
                model_name = self._names.get(class_id)
                vehicle_class = self._vehicle_ids.get(class_id)
                if model_name is None or vehicle_class is None:
                    # Defensive check: prediction IDs must come from this model's
                    # verified label map, not an assumed COCO index.
                    continue
                bbox = tuple(float(value) for value in coordinates)
                score = float(confidence)
                if (
                    len(bbox) != 4
                    or not all(math.isfinite(value) for value in bbox)
                    or not math.isfinite(score)
                    or not 0.0 <= score <= 1.0
                ):
                    continue
                x1, y1, x2, y2 = bbox
                if (
                    x1 < 0.0
                    or y1 < 0.0
                    or x2 <= x1
                    or y2 <= y1
                    or x2 > image.width
                    or y2 > image.height
                ):
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        model_class_name=model_name,
                        vehicle_class=vehicle_class,
                        confidence=score,
                        box_xyxy=(x1, y1, x2, y2),
                    )
                )
        counts = {name: 0 for name in VEHICLE_CLASSES}
        for item in detections:
            counts[item.vehicle_class] += 1
        return DetectionRun(
            image_key=image_key,
            image_name=image_name,
            dataset_name=dataset_name,
            model_source=self.model_source,
            confidence_threshold=threshold,
            detections=tuple(detections),
            counts=counts,
            model_class_names=dict(self._names),
            vehicle_class_ids=dict(self._vehicle_ids),
            inference_seconds=elapsed,
            image_size=image.size,
        )
