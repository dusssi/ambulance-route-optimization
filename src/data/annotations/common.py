"""Shared annotation records, matching, and validation helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

from src.data.categories import canonical_vehicle_class


@dataclass(frozen=True, slots=True)
class RawAnnotation:
    image_name: str
    category_name: str
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ParsedAnnotationFile:
    recognized: bool
    format_name: str | None = None
    annotations: tuple[RawAnnotation, ...] = ()
    image_names: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    unmapped_categories: tuple[tuple[str, int], ...] = ()
    invalid_image_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """Compact supported-class annotation box; image path is the scan's map key."""
    category_name: str
    vehicle_class: str
    box_xyxy: tuple[float, float, float, float]


@dataclass
class AnnotationScan:
    dataset_name: str
    root: Path
    annotations_by_image: dict[str, list[GroundTruthBox]] = field(default_factory=dict)
    matched_image_paths: set[str] = field(default_factory=set)
    invalid_image_paths: set[str] = field(default_factory=set)
    parsed_formats: dict[str, int] = field(default_factory=dict)
    unsupported_files: list[str] = field(default_factory=list)
    unmatched_annotation_names: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    unmapped_categories: dict[str, int] = field(default_factory=dict)

    def is_matched(self, image_path: str | Path) -> bool:
        try:
            key = str(Path(image_path).resolve())
        except OSError:
            key = str(Path(image_path).absolute())
        return key in self.matched_image_paths

    def is_invalid(self, image_path: str | Path) -> bool:
        try:
            key = str(Path(image_path).resolve())
        except OSError:
            key = str(Path(image_path).absolute())
        return key in self.invalid_image_paths

    def has_valid_match(self, image_path: str | Path) -> bool:
        return self.is_matched(image_path) and not self.is_invalid(image_path)

    def boxes_for(self, image_path: str | Path) -> list[GroundTruthBox]:
        try:
            key = str(Path(image_path).resolve())
        except OSError:
            key = str(Path(image_path).absolute())
        return list(self.annotations_by_image.get(key, ()))


class ImageLookup:
    """Resolve annotation image identifiers without silently choosing duplicates."""

    def __init__(self, root: Path, image_paths: Iterable[Path]):
        self.root = root.resolve()
        self.exact: dict[str, list[Path]] = {}
        self.basename: dict[str, list[Path]] = {}
        for path in image_paths:
            try:
                relative = path.resolve().relative_to(self.root).as_posix()
            except (ValueError, OSError):
                relative = path.as_posix()
            self.exact.setdefault(_normalize_path(relative), []).append(path.resolve())
            self.basename.setdefault(path.name.casefold(), []).append(path.resolve())

    def resolve(self, image_name: str) -> tuple[Path | None, str | None]:
        raw = str(image_name or "").strip().replace("\\", "/")
        if not raw:
            return None, "empty image identifier"
        if raw.startswith("file://"):
            raw = raw[7:]
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
                if resolved.is_file() and resolved in {p for values in self.exact.values() for p in values}:
                    return resolved, None
            except OSError:
                pass
        parts = [part for part in PurePosixPath(raw).parts if part not in (".", "", "/")]
        normalized = _normalize_path("/".join(parts))
        # Full relative name first, then progressively shorter suffixes. This
        # supports archives whose annotation names include an extra root folder.
        for start in range(len(parts)):
            suffix = _normalize_path("/".join(parts[start:]))
            matches = self.exact.get(suffix, [])
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                return None, f"ambiguous relative image identifier: {image_name}"
        basename = PurePosixPath(raw).name.casefold()
        matches = self.basename.get(basename, [])
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"ambiguous image basename: {basename}"
        return None, f"no discovered image matches annotation identifier: {image_name}"


def _normalize_path(value: str) -> str:
    value = value.replace("\\", "/").strip("/")
    return re.sub(r"/+", "/", value).casefold()


def parse_xyxy(values: Iterable[object]) -> tuple[float, float, float, float]:
    items = tuple(float(value) for value in values)
    if len(items) != 4 or not all(math.isfinite(value) for value in items):
        raise ValueError("bounding box must contain four finite coordinates")
    x1, y1, x2, y2 = items
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box must have positive width and height")
    return x1, y1, x2, y2


def vehicle_class_for_category(name: object) -> str | None:
    return canonical_vehicle_class(name)
