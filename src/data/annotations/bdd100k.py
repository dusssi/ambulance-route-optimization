"""Reader for the BDD100K object-detection JSON schema.

Supports common BDD100K records containing ``name`` and ``labels`` or
``frames``/``objects`` with ``category`` and ``box2d`` fields. A caller may
provide a safe image-name fallback for a verified single-record per-image
file. This is parser capability, not a claim that the Kaggle archive has been
downloaded or verified in this checkout.
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
                from ijson.common import JSONError as IJSONError

                def stream_items() -> Iterator[Any]:
                    with path.open("rb") as handle:
                        try:
                            yield from ijson.items(handle, "item")
                        except IJSONError as exc:
                            raise ValueError(f"invalid JSON: {exc}") from exc

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
    """Parse BDD100K boxes; an explicit filename fallback requires one record."""
    path = Path(path)
    records, read_error = _iter_json_records(path)
    if read_error:
        return ParsedAnnotationFile(False, issues=(f"{path}: {read_error}",))

    if image_name_override is not None:
        # A filename fallback is safe only for a one-record per-image file.
        # Do not apply one filename to every record in a global label archive.
        sentinel = object()
        try:
            first_record = next(records, sentinel)
            second_record = next(records, sentinel)
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            return ParsedAnnotationFile(
                False,
                issues=(f"{path}: could not verify a single record for the filename fallback: {exc}",),
            )
        if first_record is sentinel:
            return ParsedAnnotationFile(False, issues=(f"{path}: no record to associate with the filename fallback",))
        if second_record is not sentinel:
            return ParsedAnnotationFile(
                False,
                issues=(f"{path}: filename fallback requires exactly one record; found multiple records",),
            )
        records = iter((first_record,))

    annotations: list[RawAnnotation] = []
    image_names: list[str] = []
    issues: list[str] = []
    unmapped: dict[str, int] = {}
    invalid_image_names: list[str] = []
    unattributed_invalid_record = False
    recognized = False

    try:
        for record_number, record in enumerate(records, 1):
            if not isinstance(record, dict):
                issues.append(f"{path}: record {record_number} is not an object")
                continue

            image_name = next(
                (
                    value
                    for key in ("name", "image", "file_name")
                    if isinstance((value := record.get(key)), str) and value.strip()
                ),
                image_name_override,
            )
            has_frames = "frames" in record
            has_labels = "labels" in record
            if not isinstance(image_name, str) or not image_name.strip():
                if has_frames or has_labels:
                    issues.append(f"{path}: record {record_number} has no valid image identifier")
                    unattributed_invalid_record = True
                continue

            # Support the per-image BDD100K shape: name + one frame + objects.
            # A multi-frame record cannot safely be matched to one still image;
            # withhold it rather than silently evaluating only frames[0].
            frames = record.get("frames")
            if has_frames:
                recognized = True
                image_names.append(image_name)
                if not isinstance(frames, list):
                    invalid_image_names.append(image_name)
                    issues.append(f"{path}: record {record_number} has no valid frames array")
                    continue
                if len(frames) != 1:
                    invalid_image_names.append(image_name)
                    issues.append(
                        f"{path}: record {record_number} must contain exactly one frame for a single-image annotation; "
                        f"found {len(frames)}"
                    )
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
                    if not isinstance(category, str) or not category.strip():
                        invalid_image_names.append(image_name)
                        issues.append(
                            f"{path}: record {record_number}, object {object_number} has no valid category"
                        )
                        continue
                    if not isinstance(box, dict):
                        # Some non-vehicle shapes (for example polygons) are
                        # not object-detection boxes. A missing box for a mapped
                        # vehicle, however, makes this image unevaluable.
                        if canonical_vehicle_class(category) is not None:
                            invalid_image_names.append(image_name)
                            issues.append(
                                f"{path}: record {record_number}, object {object_number} ({category}) has no box2d"
                            )
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
                if has_labels:
                    recognized = True
                    image_names.append(image_name)
                    invalid_image_names.append(image_name)
                    issues.append(f"{path}: record {record_number} has no valid labels array")
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
        issues.append(f"{path}: no recognized BDD100K frames/objects or name/labels records")
        return ParsedAnnotationFile(False, issues=tuple(issues))
    if unattributed_invalid_record:
        # Without an image identifier the malformed record cannot be matched
        # safely; withhold any named records from this same annotation file.
        invalid_image_names.extend(image_names)

    return ParsedAnnotationFile(
        recognized=True,
        format_name="BDD100K detection JSON",
        annotations=tuple(annotations),
        image_names=tuple(dict.fromkeys(image_names)),
        issues=tuple(issues),
        unmapped_categories=tuple(sorted(unmapped.items())),
        invalid_image_names=tuple(dict.fromkeys(invalid_image_names)),
    )

