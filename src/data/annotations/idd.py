"""Readers for common IDD object-detection annotation schemas.

The Kaggle mirror is not present in the current checkout, so these adapters
recognize explicit COCO JSON, BDD-style per-image JSON, Pascal VOC XML, and
YOLO text-label structures. A format is reported as supported only when its
schema is actually recognized in the local files.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from src.data.annotations.bdd100k import parse_bdd100k_json
from src.data.annotations.common import ParsedAnnotationFile, RawAnnotation, parse_xyxy
from src.data.categories import canonical_vehicle_class
from src.data.images import load_rgb_image


def _root_keys(path: Path) -> set[str]:
    try:
        import ijson

        keys: set[str] = set()
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix == "" and event == "map_key":
                    keys.add(str(value))
        return keys
    except (ImportError, OSError, ValueError):
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            return set(data) if isinstance(data, dict) else set()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return set()


def _json_array_items(path: Path, prefix: str) -> Iterator[Any]:
    try:
        import ijson

        with path.open("rb") as stream:
            yield from ijson.items(stream, prefix)
    except ImportError:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        current: Any = data
        for part in prefix.split("."):
            if part == "item":
                continue
            current = current.get(part, []) if isinstance(current, dict) else []
        if isinstance(current, list):
            yield from current


def parse_coco_json(path: str | Path) -> ParsedAnnotationFile:
    """Parse COCO-style ``images``, ``annotations`` and ``categories`` arrays."""
    path = Path(path)
    keys = _root_keys(path)
    if not {"images", "annotations"}.issubset(keys):
        return ParsedAnnotationFile(False)

    issues: list[str] = []
    invalid_image_names: list[str] = []
    try:
        images: dict[str, str] = {}
        for image in _json_array_items(path, "images.item"):
            if not isinstance(image, dict) or "id" not in image or not image.get("file_name"):
                issues.append(f"{path}: malformed COCO image entry")
                continue
            images[str(image["id"])] = str(image["file_name"])
        categories: dict[str, str] = {}
        for category in _json_array_items(path, "categories.item"):
            if isinstance(category, dict) and "id" in category and category.get("name"):
                categories[str(category["id"])] = str(category["name"])

        annotations: list[RawAnnotation] = []
        unmapped: dict[str, int] = {}
        for index, item in enumerate(_json_array_items(path, "annotations.item"), 1):
            if not isinstance(item, dict):
                issues.append(f"{path}: COCO annotation {index} is not an object")
                continue
            image_name = images.get(str(item.get("image_id")))
            if not image_name:
                issues.append(f"{path}: COCO annotation {index} references an unknown image_id")
                continue
            category = categories.get(str(item.get("category_id")), f"category_id:{item.get('category_id')}")
            if category.startswith("category_id:"):
                invalid_image_names.append(image_name)
            bbox = item.get("bbox")
            try:
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError("bbox must be [x, y, width, height]")
                x, y, width, height = (float(value) for value in bbox)
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    raise ValueError("bbox values must be finite")
                xyxy = parse_xyxy((x, y, x + width, y + height))
            except (TypeError, ValueError) as exc:
                invalid_image_names.append(image_name)
                issues.append(f"{path}: COCO annotation {index} has invalid bbox: {exc}")
                continue
            if canonical_vehicle_class(category) is None:
                unmapped[category] = unmapped.get(category, 0) + 1
            else:
                annotations.append(RawAnnotation(image_name, category, xyxy))
        return ParsedAnnotationFile(
            recognized=True,
            format_name="COCO JSON",
            annotations=tuple(annotations),
            image_names=tuple(images.values()),
            issues=tuple(issues),
            unmapped_categories=tuple(sorted(unmapped.items())),
            invalid_image_names=tuple(dict.fromkeys(invalid_image_names)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ParsedAnnotationFile(False, issues=(f"{path}: could not parse COCO JSON: {exc}",))


def parse_pascal_voc_xml(path: str | Path) -> ParsedAnnotationFile:
    """Parse one Pascal VOC XML file using its ``filename`` and ``bndbox``."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return ParsedAnnotationFile(False, issues=(f"{path}: invalid XML: {exc}",))
    if root.tag.casefold() != "annotation":
        return ParsedAnnotationFile(False)
    image_name = root.findtext("filename") or root.findtext("path") or f"{path.stem}.jpg"
    image_name = Path(image_name).name if root.findtext("filename") else image_name
    annotations: list[RawAnnotation] = []
    issues: list[str] = []
    unmapped: dict[str, int] = {}
    invalid_image_names: list[str] = []
    for index, obj in enumerate(root.findall("object"), 1):
        category = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not category or box is None:
            invalid_image_names.append(image_name)
            issues.append(f"{path}: object {index} lacks name or bndbox")
            continue
        try:
            # VOC coordinates are 1-based. Convert the top-left point to the
            # zero-based image convention used by Pillow/Ultralytics.
            xyxy = parse_xyxy((
                float(box.findtext("xmin")) - 1.0,
                float(box.findtext("ymin")) - 1.0,
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            ))
        except (TypeError, ValueError) as exc:
            invalid_image_names.append(image_name)
            issues.append(f"{path}: object {index} has invalid bndbox: {exc}")
            continue
        if canonical_vehicle_class(category) is None:
            unmapped[category] = unmapped.get(category, 0) + 1
        else:
            annotations.append(RawAnnotation(image_name, category, xyxy))
    return ParsedAnnotationFile(
        recognized=True,
        format_name="Pascal VOC XML",
        annotations=tuple(annotations),
        image_names=(image_name,),
        issues=tuple(issues),
        unmapped_categories=tuple(sorted(unmapped.items())),
        invalid_image_names=tuple(dict.fromkeys(invalid_image_names)),
    )


@lru_cache(maxsize=16)
def _load_class_names(root: Path) -> tuple[str, ...] | None:
    """Read a dataset class-name map once, not once per YOLO label file."""
    candidates = ("classes.txt", "classes.names", "obj.names", "labels.names")
    for name in candidates:
        for path in root.rglob(name):
            try:
                values = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                continue
            if values:
                return values
    return None


def _find_yolo_image(label_path: Path, root: Path, image_paths: tuple[Path, ...]) -> tuple[Path | None, str | None]:
    stem = label_path.stem.casefold()
    matches = [path for path in image_paths if path.stem.casefold() == stem]
    # Prefer matching split folders when labels/train mirrors images/train.
    label_parts = [part.casefold() for part in label_path.relative_to(root).parts[:-1]]
    split = next((part for part in label_parts if part in {"train", "val", "valid", "validation", "test"}), None)
    if split:
        split_matches = [path for path in matches if split in [part.casefold() for part in path.relative_to(root).parts[:-1]]]
        if split_matches:
            matches = split_matches
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"no image with stem {label_path.stem!r} matches YOLO label file"
    return None, f"ambiguous image stem {label_path.stem!r} for YOLO label file"


def parse_yolo_txt(
    path: str | Path,
    root: str | Path,
    image_paths: tuple[Path, ...],
) -> ParsedAnnotationFile:
    """Parse normalized YOLO ``class x_center y_center width height`` labels.

    Class IDs are mapped only when a sibling/ancestor classes file is available.
    Without one, they are reported as unmapped and that image is marked
    unevaluable; class-ID assumptions are deliberately not made.
    """
    path, root = Path(path), Path(root)
    if path.name.casefold() in {"classes.txt", "classes.names", "obj.names", "labels.names"}:
        return ParsedAnnotationFile(False)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return ParsedAnnotationFile(False, issues=(f"{path}: could not read YOLO labels: {exc}",))
    if not any(line.strip() for line in lines):
        # Empty per-image labels are valid only if a corresponding image exists.
        matched_image, match_error = _find_yolo_image(path, root, image_paths)
        if matched_image is None:
            return ParsedAnnotationFile(False, issues=(f"{path}: {match_error}",))
        try:
            image_name = matched_image.relative_to(root).as_posix()
        except ValueError:
            image_name = matched_image.name
        return ParsedAnnotationFile(True, "YOLO TXT", image_names=(image_name,))

    matched_image, match_error = _find_yolo_image(path, root, image_paths)
    if matched_image is None:
        return ParsedAnnotationFile(False, issues=(f"{path}: {match_error}",))
    try:
        image = load_rgb_image(matched_image)
    except ValueError as exc:
        return ParsedAnnotationFile(False, issues=(f"{path}: paired image is unreadable: {exc}",))
    try:
        image_name = matched_image.relative_to(root).as_posix()
    except ValueError:
        image_name = matched_image.name
    class_names = _load_class_names(root)
    annotations: list[RawAnnotation] = []
    issues: list[str] = []
    unmapped: dict[str, int] = {}
    invalid_image_names: list[str] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        pieces = line.split()
        if len(pieces) != 5:
            invalid_image_names.append(image_name)
            issues.append(f"{path}: line {line_number} must contain five YOLO fields")
            continue
        try:
            class_id = int(pieces[0])
            center_x, center_y, width, height = (float(value) for value in pieces[1:])
            if class_id < 0 or not all(math.isfinite(value) for value in (center_x, center_y, width, height)):
                raise ValueError("class ID and box values must be finite and nonnegative")
            if width <= 0 or height <= 0 or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and width <= 1 and height <= 1):
                raise ValueError("normalized center/size values must be within [0, 1]")
            left, top = center_x - width / 2, center_y - height / 2
            right, bottom = center_x + width / 2, center_y + height / 2
            if left < 0 or top < 0 or right > 1 or bottom > 1:
                raise ValueError("normalized box bounds must stay within [0, 1]")
            xyxy = parse_xyxy((left * image.width, top * image.height, right * image.width, bottom * image.height))
        except (TypeError, ValueError) as exc:
            invalid_image_names.append(image_name)
            issues.append(f"{path}: line {line_number} has invalid values: {exc}")
            continue
        category = class_names[class_id] if class_names and class_id < len(class_names) else f"class_id:{class_id}"
        canonical = canonical_vehicle_class(category)
        if canonical is None:
            unmapped[category] = unmapped.get(category, 0) + 1
            if category.startswith("class_id:"):
                invalid_image_names.append(image_name)
        else:
            annotations.append(RawAnnotation(image_name, category, xyxy))

    return ParsedAnnotationFile(
        recognized=True,
        format_name="YOLO TXT",
        annotations=tuple(annotations),
        image_names=(image_name,),
        issues=tuple(issues),
        unmapped_categories=tuple(sorted(unmapped.items())),
        invalid_image_names=tuple(dict.fromkeys(invalid_image_names)),
    )


def read_idd_annotation_file(
    path: str | Path,
    root: str | Path,
    image_paths: tuple[Path, ...],
) -> ParsedAnnotationFile:
    """Dispatch an IDD file to a known schema reader, without guessing labels."""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            coco = parse_coco_json(path)
            if coco.recognized:
                return coco
        return parse_bdd100k_json(path)
    if suffix == ".xml":
        return parse_pascal_voc_xml(path)
    if suffix == ".txt":
        return parse_yolo_txt(path, root, image_paths)
    return ParsedAnnotationFile(False)
