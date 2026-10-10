"""Dataset-specific annotation reader dispatch and image matching."""
from __future__ import annotations

from pathlib import Path

from src.data.annotations.bdd100k import parse_bdd100k_json
from src.data.annotations.common import AnnotationScan, GroundTruthBox, ImageLookup, ParsedAnnotationFile
from src.data.annotations.idd import parse_coco_json, parse_pascal_voc_xml, parse_yolo_txt
from src.data.discovery import DatasetDiscovery
from src.data.categories import canonical_vehicle_class


def _parse_bdd_file(
    path: Path,
    discovery: DatasetDiscovery,
    image_paths_by_stem: dict[str, list[Path]],
) -> ParsedAnnotationFile:
    if path.suffix.casefold() in {".json", ".jsonl"}:
        bdd = parse_bdd100k_json(path)
        if not bdd.recognized:
            same_stem_images = image_paths_by_stem.get(path.stem.casefold(), [])
            if len(same_stem_images) == 1:
                filename_matched = parse_bdd100k_json(
                    path,
                    image_name_override=same_stem_images[0].name,
                )
                if filename_matched.recognized:
                    bdd = filename_matched
                else:
                    bdd = ParsedAnnotationFile(
                        False,
                        issues=tuple(dict.fromkeys(bdd.issues + filename_matched.issues)),
                    )
        if bdd.recognized or path.suffix.casefold() == ".jsonl":
            return bdd
        # Some repackaged archives use COCO JSON instead of the official
        # BDD per-image JSON. The actual parser name is reported when detected.
        coco = parse_coco_json(path)
        if coco.recognized:
            return coco
        return ParsedAnnotationFile(
            False,
            issues=tuple(bdd.issues) + tuple(coco.issues) + (f"{path}: unsupported BDD100K annotation schema",),
        )
    if path.suffix.casefold() == ".xml":
        return parse_pascal_voc_xml(path)
    if path.suffix.casefold() == ".txt":
        return parse_yolo_txt(path, discovery.root, discovery.image_paths)
    return ParsedAnnotationFile(False)


def _parse_idd_file(path: Path, discovery: DatasetDiscovery) -> ParsedAnnotationFile:
    from src.data.annotations.idd import read_idd_annotation_file

    return read_idd_annotation_file(path, discovery.root, discovery.image_paths)


def scan_annotations(discovery: DatasetDiscovery) -> AnnotationScan:
    """Parse discovered annotations and explicitly match their image identifiers.

    Non-vehicle categories are retained as ground truth with ``vehicle_class``
    set to ``None``; evaluation uses only the supported four vehicle classes.
    Every unreadable, malformed, unsupported, unmatched, or unmapped item is
    represented in the returned scan report.
    """
    scan = AnnotationScan(dataset_name=discovery.dataset_name, root=discovery.root)
    if not discovery.root_exists:
        scan.issues.extend(discovery.scan_errors)
        return scan
    lookup = ImageLookup(discovery.root, discovery.image_paths)
    image_paths_by_stem: dict[str, list[Path]] = {}
    if discovery.dataset_name.casefold() in {"bdd100k", "bdd"}:
        for image_path in discovery.image_paths:
            image_paths_by_stem.setdefault(image_path.stem.casefold(), []).append(image_path)

    for annotation_path in discovery.annotation_paths:
        try:
            if discovery.dataset_name.casefold() in {"bdd100k", "bdd"}:
                parsed = _parse_bdd_file(annotation_path, discovery, image_paths_by_stem)
            else:
                parsed = _parse_idd_file(annotation_path, discovery)
        except Exception as exc:  # A bad file must not abort the remaining scan.
            scan.issues.append(f"{annotation_path}: parser failed: {type(exc).__name__}: {exc}")
            scan.unsupported_files.append(str(annotation_path))
            continue

        scan.issues.extend(parsed.issues)
        if not parsed.recognized:
            scan.unsupported_files.append(str(annotation_path))
            if annotation_path.suffix.casefold() not in {".json", ".jsonl", ".xml", ".txt"}:
                scan.issues.append(f"{annotation_path}: unsupported annotation extension")
            continue
        format_name = parsed.format_name or "recognized schema"
        scan.parsed_formats[format_name] = scan.parsed_formats.get(format_name, 0) + 1

        for image_name in parsed.image_names:
            image_path, match_error = lookup.resolve(image_name)
            if image_path is None:
                scan.unmatched_annotation_names.append(image_name)
                if match_error:
                    scan.issues.append(f"{annotation_path}: {match_error}")
                continue
            image_key = str(image_path.resolve())
            scan.matched_image_paths.add(image_key)

        for image_name in parsed.invalid_image_names:
            image_path, match_error = lookup.resolve(image_name)
            if image_path is None:
                if image_name not in scan.unmatched_annotation_names:
                    scan.unmatched_annotation_names.append(image_name)
                if match_error:
                    scan.issues.append(f"{annotation_path}: invalid-label image match failed: {match_error}")
                continue
            scan.invalid_image_paths.add(str(image_path.resolve()))

        for raw in parsed.annotations:
            image_path, match_error = lookup.resolve(raw.image_name)
            if image_path is None:
                if raw.image_name not in scan.unmatched_annotation_names:
                    scan.unmatched_annotation_names.append(raw.image_name)
                if match_error:
                    scan.issues.append(f"{annotation_path}: {match_error}")
                continue
            image_key = str(image_path.resolve())
            scan.matched_image_paths.add(image_key)
            vehicle_class = canonical_vehicle_class(raw.category_name)
            if vehicle_class is None:
                # Defensive fallback; schema readers normally keep only mapped
                # vehicle boxes and report other categories as compact counts.
                scan.unmapped_categories[raw.category_name] = scan.unmapped_categories.get(raw.category_name, 0) + 1
                continue
            box = GroundTruthBox(
                category_name=raw.category_name,
                vehicle_class=vehicle_class,
                box_xyxy=raw.box_xyxy,
            )
            scan.annotations_by_image.setdefault(image_key, []).append(box)

        for category, count in parsed.unmapped_categories:
            scan.unmapped_categories[category] = scan.unmapped_categories.get(category, 0) + count

    # Stable order makes reports/tests reproducible across filesystem iteration.
    for boxes in scan.annotations_by_image.values():
        boxes.sort(key=lambda item: (item.vehicle_class or "", item.box_xyxy, item.category_name))
    scan.unmatched_annotation_names = list(dict.fromkeys(scan.unmatched_annotation_names))
    scan.issues = list(dict.fromkeys(scan.issues))
    return scan
