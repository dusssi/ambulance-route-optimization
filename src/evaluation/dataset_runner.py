"""Reproducible subset evaluation for BDD100K's per-image JSON annotations.

The batch runner deliberately pairs one image with one same-stem JSON/JSONL file.
It does not download data, build a full-dataset annotation cache, or infer that a
single-frame label represents video/live traffic. Unsupported/global annotation
files are rejected or skipped with diagnostics rather than guessed.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from src.data.annotations.bdd100k import parse_bdd100k_json
from src.data.annotations.common import GroundTruthBox, ImageLookup
from src.data.categories import canonical_vehicle_class
from src.data.discovery import DatasetDiscovery, discover_dataset
from src.data.images import load_rgb_image
from src.detection.models import DetectionRun
from src.evaluation.detection_metrics import (
    DEFAULT_IOU_THRESHOLD,
    ImageEvaluation,
    aggregate_evaluations,
    evaluate_image,
)

DEFAULT_CONFIDENCE_THRESHOLDS = (0.25,)


class VehicleDetector(Protocol):
    """Minimal inference interface used by the dataset runner and its tests."""

    def detect(
        self,
        image,
        image_key: str,
        image_name: str,
        dataset_name: str,
        confidence_threshold: float,
    ) -> DetectionRun: ...


@dataclass(frozen=True, slots=True)
class ImageAnnotationPair:
    image_path: Path
    annotation_path: Path


@dataclass(frozen=True, slots=True)
class EvaluatedImage:
    image_name: str
    annotation_file: str
    inference_seconds: float
    evaluations: tuple[ImageEvaluation, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSubsetResult:
    dataset_root: Path
    requested_sample_size: int
    sample_size: int
    seed: int
    iou_threshold: float
    confidence_thresholds: tuple[float, ...]
    candidate_pair_count: int
    attempted_pair_count: int
    pair_diagnostics: dict[str, int]
    evaluated_images: tuple[EvaluatedImage, ...]
    skipped_by_reason: dict[str, int]
    skipped_examples: tuple[str, ...]
    aggregates: dict[str, dict[str, object]]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def discover_bdd100k_image_annotation_pairs(
    discovery: DatasetDiscovery,
) -> tuple[tuple[ImageAnnotationPair, ...], dict[str, int]]:
    """Pair unique image/JSON basenames without parsing the entire archive."""
    if discovery.dataset_name.casefold() not in {"bdd100k", "bdd"}:
        raise ValueError("The subset runner currently supports the BDD100K dataset name only.")
    if not discovery.root_exists:
        raise ValueError(f"BDD100K dataset directory is unavailable: {discovery.root}")

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in discovery.image_paths:
        images_by_stem[image_path.stem.casefold()].append(image_path)
    annotations_by_stem: dict[str, list[Path]] = defaultdict(list)
    for annotation_path in discovery.annotation_paths:
        if annotation_path.suffix.casefold() in {".json", ".jsonl"}:
            annotations_by_stem[annotation_path.stem.casefold()].append(annotation_path)

    pairs: list[ImageAnnotationPair] = []
    ambiguous_stems = 0
    annotations_without_images = 0
    for stem, annotation_paths in sorted(annotations_by_stem.items()):
        image_paths = images_by_stem.get(stem, [])
        if not image_paths:
            annotations_without_images += len(annotation_paths)
        elif len(image_paths) != 1 or len(annotation_paths) != 1:
            ambiguous_stems += 1
        else:
            pairs.append(ImageAnnotationPair(image_paths[0], annotation_paths[0]))

    images_without_annotations = sum(
        len(image_paths)
        for stem, image_paths in images_by_stem.items()
        if stem not in annotations_by_stem
    )
    pairs.sort(key=lambda pair: (pair.image_path.as_posix().casefold(), pair.annotation_path.as_posix().casefold()))
    diagnostics = {
        "unique_same_stem_pairs": len(pairs),
        "images_without_matching_json": images_without_annotations,
        "json_files_without_matching_image": annotations_without_images,
        "ambiguous_basename_groups": ambiguous_stems,
    }
    return tuple(pairs), diagnostics


def _thresholds(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("Provide at least one confidence threshold.")
    result: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("Every confidence threshold must be a finite value within [0, 1].")
        threshold = float(value)
        if threshold not in result:
            result.append(threshold)
    return tuple(result)


def _skip(
    reason: str,
    detail: str,
    counts: Counter[str],
    examples: list[str],
    image_name: str,
    example_limit: int,
) -> None:
    counts[reason] += 1
    if len(examples) < example_limit:
        examples.append(f"{image_name}: {reason}: {detail}")


def run_bdd100k_subset(
    root_path: str | Path,
    detector: VehicleDetector,
    sample_size: int = 100,
    seed: int = 42,
    confidence_thresholds: tuple[float, ...] | list[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    progress: Callable[[int, int, int], None] | None = None,
    max_annotation_bytes: int = 8 * 1024 * 1024,
    skip_example_limit: int = 20,
) -> EvaluationSubsetResult:
    """Evaluate a deterministic subset of valid matched BDD100K still images.

    The sampler shuffles sorted, unique same-stem image/annotation pairs with a
    local ``random.Random(seed)`` instance, then accepts valid pairs until the
    requested number is reached. Confidence thresholds share one detector call
    per image by running inference at the lowest threshold and filtering the
    returned boxes during metric calculation.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not math.isfinite(float(iou_threshold))
        or not 0.0 < float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be a finite value within (0, 1].")
    if (
        isinstance(max_annotation_bytes, bool)
        or not isinstance(max_annotation_bytes, int)
        or max_annotation_bytes <= 0
    ):
        raise ValueError("max_annotation_bytes must be a positive integer")
    if (
        isinstance(skip_example_limit, bool)
        or not isinstance(skip_example_limit, int)
        or skip_example_limit < 0
    ):
        raise ValueError("skip_example_limit must be a nonnegative integer")

    root = Path(root_path).expanduser().resolve()
    discovery = discover_dataset("BDD100K", root)
    pairs, pair_diagnostics = discover_bdd100k_image_annotation_pairs(discovery)
    if not pairs:
        raise ValueError(
            "No unique same-stem BDD100K image/JSON pairs were discovered. "
            f"Images: {discovery.image_count}; candidate annotations: {discovery.annotation_file_count}; "
            f"pair diagnostics: {pair_diagnostics}. This runner expects one per-image JSON/JSONL file "
            "with the same basename as its image."
        )

    selected_order = list(pairs)
    random.Random(seed).shuffle(selected_order)
    lookup = ImageLookup(discovery.root, discovery.image_paths)
    thresholds = _thresholds(confidence_thresholds)
    iou_limit = float(iou_threshold)
    minimum_confidence = min(thresholds)
    evaluated: list[EvaluatedImage] = []
    skipped: Counter[str] = Counter()
    skip_examples: list[str] = []
    attempted = 0

    for pair in selected_order:
        if len(evaluated) >= sample_size:
            break
        attempted += 1
        relative_image = _relative(pair.image_path, root)
        relative_annotation = _relative(pair.annotation_path, root)

        try:
            if pair.annotation_path.stat().st_size > max_annotation_bytes:
                _skip(
                    "annotation_file_too_large",
                    f"file exceeds the per-image limit of {max_annotation_bytes} bytes",
                    skipped,
                    skip_examples,
                    relative_image,
                    skip_example_limit,
                )
                continue
        except OSError as exc:
            _skip("annotation_file_unreadable", str(exc), skipped, skip_examples, relative_image, skip_example_limit)
            continue

        parsed = parse_bdd100k_json(pair.annotation_path)
        if not parsed.recognized:
            detail = "; ".join(parsed.issues[:2]) or "unsupported JSON structure"
            _skip("annotation_unrecognized", detail, skipped, skip_examples, relative_image, skip_example_limit)
            continue

        matching_names: list[str] = []
        for image_name in parsed.image_names:
            resolved, _ = lookup.resolve(image_name)
            if resolved is not None and resolved.resolve() == pair.image_path.resolve():
                matching_names.append(image_name)
        if len(parsed.image_names) != 1 or len(matching_names) != 1:
            _skip(
                "annotation_record_mismatch",
                f"expected one record resolving to the paired image; found {len(parsed.image_names)} record name(s), "
                f"{len(matching_names)} matching",
                skipped,
                skip_examples,
                relative_image,
                skip_example_limit,
            )
            continue

        invalid_names: list[str] = []
        for image_name in parsed.invalid_image_names:
            resolved, _ = lookup.resolve(image_name)
            if resolved is not None and resolved.resolve() == pair.image_path.resolve():
                invalid_names.append(image_name)
        if invalid_names or parsed.issues:
            detail = "; ".join(parsed.issues[:2]) or "annotation record has malformed labels"
            _skip("malformed_annotation", detail, skipped, skip_examples, relative_image, skip_example_limit)
            continue

        try:
            image = load_rgb_image(pair.image_path)
        except ValueError as exc:
            _skip("corrupt_image", str(exc), skipped, skip_examples, relative_image, skip_example_limit)
            continue

        ground_truth: list[GroundTruthBox] = []
        invalid_box: tuple[float, float, float, float] | None = None
        for annotation in parsed.annotations:
            resolved, _ = lookup.resolve(annotation.image_name)
            if resolved is None or resolved.resolve() != pair.image_path.resolve():
                continue
            vehicle_class = canonical_vehicle_class(annotation.category_name)
            if vehicle_class is None:
                continue
            x1, y1, x2, y2 = annotation.box_xyxy
            if x2 > image.width or y2 > image.height:
                invalid_box = annotation.box_xyxy
                break
            ground_truth.append(
                GroundTruthBox(
                    category_name=annotation.category_name,
                    vehicle_class=vehicle_class,
                    box_xyxy=annotation.box_xyxy,
                )
            )
        if invalid_box is not None:
            _skip(
                "ground_truth_box_out_of_bounds",
                f"box {invalid_box!r} exceeds image dimensions {image.width}x{image.height}",
                skipped,
                skip_examples,
                relative_image,
                skip_example_limit,
            )
            continue

        detection_run = detector.detect(
            image=image,
            image_key=str(pair.image_path.resolve()),
            image_name=relative_image,
            dataset_name="BDD100K",
            confidence_threshold=minimum_confidence,
        )
        inference_seconds = float(detection_run.inference_seconds)
        if not math.isfinite(inference_seconds) or inference_seconds < 0.0:
            raise ValueError("detector returned an invalid inference duration")
        per_threshold = tuple(
            evaluate_image(
                predictions=detection_run.detections,
                ground_truth=ground_truth,
                image_name=relative_image,
                confidence_threshold=threshold,
                iou_threshold=iou_limit,
                image_size=image.size,
            )
            for threshold in thresholds
        )
        evaluated.append(
            EvaluatedImage(
                image_name=relative_image,
                annotation_file=relative_annotation,
                inference_seconds=inference_seconds,
                evaluations=per_threshold,
            )
        )
        if progress is not None:
            progress(attempted, len(selected_order), len(evaluated))

    aggregates = {
        repr(threshold): aggregate_evaluations(
            [item.evaluations[index] for item in evaluated]
        )
        for index, threshold in enumerate(thresholds)
        if evaluated
    }
    return EvaluationSubsetResult(
        dataset_root=root,
        requested_sample_size=sample_size,
        sample_size=len(evaluated),
        seed=seed,
        iou_threshold=iou_limit,
        confidence_thresholds=thresholds,
        candidate_pair_count=len(pairs),
        attempted_pair_count=attempted,
        pair_diagnostics=pair_diagnostics,
        evaluated_images=tuple(evaluated),
        skipped_by_reason=dict(sorted(skipped.items())),
        skipped_examples=tuple(skip_examples),
        aggregates=aggregates,
    )
