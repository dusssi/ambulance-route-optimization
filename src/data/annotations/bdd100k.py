"""Reader for the BDD100K object-detection JSON schema.

Supports the common BDD100K records containing ``name`` and ``labels`` with
``category`` and ``box2d`` fields. This is parser capability, not a claim that
the Kaggle archive has been downloaded or verified in this checkout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.data.annotations.common import ParsedAnnotationFile, RawAnnotation, parse_xyxy
from src.data.categories import canonical_vehicle_class


def _iter_json_records(path: Path) -> tuple[Iterator[Any], str | None]:
    """Create a streaming iterator for JSON arrays, objects, or JSONL records."""
    def json_lines() -> Iterator[Any]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

    try:
        if path.suffix.casefold() in {".jsonl", ".ndjson"}:
            return json_lines(), None
        with path.open("rb") as stream:
            first = b""
            while not first:
                byte = stream.read(1)
                if not byte:
                    return iter(()), "empty JSON file"
                if not byte.isspace():
                    first = byte
        if first == b"[":
            try:
                import ijson

                def stream_items() -> Iterator[Any]:
                    with path.open("rb") as handle:
                        yield from ijson.items(handle, "item")

                return stream_items(), None
            except ImportError:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return iter(data), None
                return iter(()), "JSON root was expected to be an array"
        # A single object is handled as a one-record list. If this is JSONL
        # stored with a .json extension, fall back to line-by-line parsing.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return iter(data), None
            if isinstance(data, dict):
                return iter((data,)), None
            return iter(()), "JSON root must be an object or array"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return json_lines(), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return iter(()), f"could not read JSON: {exc}"


def parse_bdd100k_json(
    path: str | Path,
    image_name_override: str | None = None,
) -> ParsedAnnotationFile:
    """Parse BDD100K object boxes from legacy labels or frames/objects JSON."""
    path = Path(path)
    records, read_error = _iter_json_records(path)
    if read_error:
        return ParsedAnnotationFile(False, issues=(f"{path}: {read_error}",))

    annotations: list[RawAnnotation] = []
    image_names: list[str] = []
    issues: list[str] = []
    unmapped: dict[str, int] = {}
    invalid_image_names: list[str] = []
    recognized = False

    try:
        for record_number, record in enumerate(records, 1):
            if not isinstance(record, dict):
                issues.append(f"{path}: record {record_number} is not an object")
                continue

            image_name = image_name_override or record.get("name") or record.get("image") or record.get("file_name")
            if not isinstance(image_name, str) or not image_name.strip():
                continue

            # Support BDD100K detection JSON: name + frames[0].objects.
            frames = record.get("frames")
            if isinstance(frames, list):
                recognized = True
                image_names.append(image_name)
                if not frames:
                    invalid_image_names.append(image_name)
                    issues.append(f"{path}: record {record_number} has an empty frames array")
                    continue

                frame = frames[0]
                objects = frame.get("objects") if isinstance(frame, dict) else None
                if not isinstance(objects, list):
                    invalid_image_names.append(image_name)
                    issues.append(f"{path}: record {record_number} has no valid objects array")
                    continue

                for object_number, obj in enumerate(objects, 1):
                    if not isinstance(obj, dict):
                        invalid_image_names.append(image_name)
                        issues.append(f"{path}: record {record_number}, object {object_number} is not an object")
                        continue

                    category = obj.get("category")
                    box = obj.get("box2d")
                    if not isinstance(category, str) or not isinstance(box, dict):
                        # Polygons/attributes are not bounding-box annotations.
                        continue

                    try:
                        xyxy = parse_xyxy((box["x1"], box["y1"], box["x2"], box["y2"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        invalid_image_names.append(image_name)
                        issues.append(
                            f"{path}: record {record_number}, object {object_number}: invalid box: {exc}"
                        )
                        continue

                    if canonical_vehicle_class(category) is None:
                        unmapped[category] = unmapped.get(category, 0) + 1
                    else:
                        annotations.append(RawAnnotation(image_name, category, xyxy))
                continue

            # Preserve legacy BDD-style name + labels + box2d records.
            labels = record.get("labels")
            if not isinstance(labels, list):
                continue

            recognized = True
            image_names.append(image_name)
            for object_number, label in enumerate(labels, 1):
                if not isinstance(label, dict):
                    invalid_image_names.append(image_name)
                    issues.append(f"{path}: record {record_number}, label {object_number} is not an object")
                    continue

                category = label.get("category") or label.get("name")
                box = label.get("box2d")
                if not isinstance(category, str) or not isinstance(box, dict):
                    invalid_image_names.append(image_name)
                    issues.append(
                        f"{path}: record {record_number}, label {object_number} lacks category/box2d"
                    )
                    continue

                try:
                    xyxy = parse_xyxy((box["x1"], box["y1"], box["x2"], box["y2"]))
                except (KeyError, TypeError, ValueError) as exc:
                    invalid_image_names.append(image_name)
                    issues.append(
                        f"{path}: record {record_number}, label {object_number}: invalid box: {exc}"
                    )
                    continue

                if canonical_vehicle_class(category) is None:
                    unmapped[category] = unmapped.get(category, 0) + 1
                else:
                    annotations.append(RawAnnotation(image_name, category, xyxy))

    except (ValueError, OSError, UnicodeDecodeError) as exc:
        return ParsedAnnotationFile(False, issues=(f"{path}: {exc}",))

    if not recognized:
        return ParsedAnnotationFile(
            False,
            issues=(f"{path}: no recognized BDD100K frames/objects or name/labels records",),
        )

    return ParsedAnnotationFile(
        recognized=True,
        format_name="BDD100K detection JSON",
        annotations=tuple(annotations),
        image_names=tuple(dict.fromkeys(image_names)),
        issues=tuple(issues),
        unmapped_categories=tuple(sorted(unmapped.items())),
        invalid_image_names=tuple(dict.fromkeys(invalid_image_names)),
    )

