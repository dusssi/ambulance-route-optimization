"""Draw actual YOLO detections on an RGB image with OpenCV."""
from __future__ import annotations

from PIL import Image

from src.detection.models import Detection

_COLORS_RGB = {
    "car": (52, 152, 219),
    "truck": (46, 204, 113),
    "bus": (241, 196, 15),
    "motorcycle": (231, 76, 60),
}


def annotate_image(image: Image.Image, detections: tuple[Detection, ...] | list[Detection]) -> Image.Image:
    """Return a copy with real detection boxes, labels, and confidence scores."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to draw detection overlays.") from exc

    rgb = np.asarray(image.convert("RGB")).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    thickness = max(1, round(min(image.size) / 450))
    font_scale = max(0.45, min(image.size) / 1100)
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item.box_xyxy)
        # OpenCV draws BGR; the palette is declared in RGB.
        red, green, blue = _COLORS_RGB.get(item.vehicle_class, (255, 255, 255))
        color = (blue, green, red)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
        label = f"{item.vehicle_class} {item.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(bgr, (x1, top), (x1 + text_width + 4, top + text_height + baseline + 4), color, -1)
        cv2.putText(
            bgr,
            label,
            (x1 + 2, top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated)
