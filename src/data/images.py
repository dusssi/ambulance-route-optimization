"""Image validation and RGB loading using Pillow."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

from PIL import Image, ImageOps

ImageSource = str | Path | bytes | bytearray | BinaryIO | Image.Image


def load_rgb_image(source: ImageSource) -> Image.Image:
    """Read and fully decode an image, apply EXIF orientation, and return RGB.

    A detached Pillow image is returned so callers do not retain open file handles.
    """
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(bytes(source)))
        elif hasattr(source, "read"):
            image = Image.open(source)
        else:
            image = Image.open(Path(source).expanduser())
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("The decoded image has invalid dimensions.")
        return image
    except Exception as exc:
        # Pillow raises several format-specific exceptions (including image-size
        # safety errors); normalize them so the dashboard can report the input.
        raise ValueError(f"Could not read a supported image: {exc}") from exc


def validate_image(path: str | Path) -> tuple[bool, tuple[int, int] | None, str | None]:
    """Validate one local image and return (valid, (width, height), error)."""
    try:
        image = load_rgb_image(path)
        return True, image.size, None
    except (ValueError, OSError) as exc:
        return False, None, str(exc)


@dataclass(frozen=True)
class ImageValidationReport:
    checked_count: int
    readable_count: int
    unreadable: tuple[tuple[str, str], ...]


def validate_image_paths(paths: Iterable[str | Path]) -> ImageValidationReport:
    """Decode each discovered image and retain errors instead of skipping them.

    This potentially expensive full-dataset check is explicit/on-demand in the
    dashboard; normal discovery does not eagerly decode an entire archive.
    """
    checked = 0
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        checked += 1
        valid, _, error = validate_image(path)
        if not valid:
            unreadable.append((str(path), error or "image could not be decoded"))
    return ImageValidationReport(
        checked_count=checked,
        readable_count=checked - len(unreadable),
        unreadable=tuple(unreadable),
    )
