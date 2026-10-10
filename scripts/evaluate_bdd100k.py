#!/usr/bin/env python3
"""Run a reproducible BDD100K detection subset evaluation (no dataset download)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.categories import VEHICLE_CLASSES
from src.data.discovery import discover_dataset
from src.detection.yolo import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MODEL,
    DEFAULT_NMS_IOU,
    YOLOVehicleDetector,
)
from src.evaluation.dataset_runner import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    discover_bdd100k_image_annotation_pairs,
    run_bdd100k_subset,
)
from src.evaluation.detection_metrics import DEFAULT_IOU_THRESHOLD
from src.exports.csv_export import rows_to_csv


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ram_status() -> dict[str, int] | None:
    """Best-effort RAM preflight without adding a psutil dependency."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            fields = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                name, value, *_ = line.split()
                if name in {"MemTotal:", "MemAvailable:"}:
                    fields[name.rstrip(":")] = int(value) * 1024
            if fields:
                return fields
        except (OSError, ValueError):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return {"MemTotal": int(page_size * pages)}
    except (AttributeError, OSError, ValueError):
        return None


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    status: dict[str, Any] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_names": [],
        "device_memory_bytes": [],
    }
    if status["cuda_available"]:
        try:
            status["device_names"] = [
                torch.cuda.get_device_name(index) for index in range(status["device_count"])
            ]
            status["device_memory_bytes"] = [
                int(torch.cuda.get_device_properties(index).total_memory)
                for index in range(status["device_count"])
            ]
        except Exception as exc:
            status["device_names_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _sha256_if_file(path_text: str) -> str | None:
    path = Path(path_text).expanduser()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_root() -> Path:
    configured = os.environ.get("AMBULANCE_BDD100K_DIR")
    return Path(configured).expanduser() if configured else ROOT / "data" / "bdd100k"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a deterministic subset of local BDD100K per-image JSON annotations. "
            "This command never downloads a dataset."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_default_data_root(), help="Local extracted BDD100K folder (default: data/bdd100k or AMBULANCE_BDD100K_DIR).")
    parser.add_argument("--sample-size", type=int, default=100, help="Target number of valid matched images (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to shuffle the sorted image/annotation pairs (default: 42).")
    parser.add_argument("--iou", "--iou-threshold", dest="iou_threshold", type=float, default=DEFAULT_IOU_THRESHOLD, help="Class-aware matching IoU threshold (default: 0.50).")
    parser.add_argument(
        "--confidence",
        dest="confidence_thresholds",
        action="append",
        type=float,
        default=None,
        help="Confidence threshold; repeat to compare several thresholds (default: 0.25).",
    )
    parser.add_argument("--model", default=os.environ.get("AMBULANCE_YOLO_MODEL", DEFAULT_MODEL), help="Ultralytics model name or local checkpoint path.")
    parser.add_argument("--device", default=None, help="Inference device accepted by Ultralytics (for example: 0, cuda:0, or cpu). Default: automatic.")
    parser.add_argument("--output-dir", type=Path, default=None, help="New or empty directory for CSV/JSON outputs; defaults to a timestamped ignored outputs/ subfolder.")
    parser.add_argument("--max-annotation-mb", type=float, default=8.0, help="Skip unusually large same-stem files rather than treating a global label archive as per-image JSON (default: 8 MiB).")
    parser.add_argument("--preflight-only", action="store_true", help="Check runtime resources and image/JSON basename pairing; do not parse labels or load a model.")
    return parser


def _output_directory(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = ROOT / "outputs" / f"bdd100k-evaluation-{stamp}"
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new --output-dir to preserve previous results."
        )
    return path


def _print_preflight(data_root: Path, candidate_pairs: int, diagnostics: dict[str, int], torch_info: dict[str, Any]) -> dict[str, Any]:
    disk_path = data_root if data_root.exists() else ROOT
    try:
        disk = shutil.disk_usage(disk_path)
    except OSError:
        disk_path = ROOT
        disk = shutil.disk_usage(ROOT)
    ram = _ram_status()
    print("Runtime preflight")
    print(f"  Python: {platform.python_version()} ({platform.platform()})")
    print(f"  RAM: {ram if ram is not None else 'unavailable'}")
    print(f"  Free disk on {disk_path}: {disk.free / (1024 ** 3):.2f} GiB")
    print(f"  PyTorch/CUDA: {torch_info}")
    print(f"  Ultralytics: {_version('ultralytics') or 'not installed'}")
    print(f"  Pillow/OpenCV: {'available' if importlib.util.find_spec('PIL') and importlib.util.find_spec('cv2') else 'one or both modules missing'}")
    print(f"  Unique same-stem image/JSON pairs: {candidate_pairs:,}")
    print(f"  Pair diagnostics: {diagnostics}")
    if not torch_info.get("cuda_available"):
        print("  Note: GPU not detected; begin with --sample-size 10 or use a GPU runtime if available.")
    if ram is not None and ram.get("MemAvailable", ram.get("MemTotal", 0)) < 2 * 1024 ** 3:
        print("  Warning: less than 2 GiB RAM appears available; keep the sample small and close other workloads.")
    if disk.free < 2 * 1024 ** 3:
        print("  Warning: less than 2 GiB free disk. Check model/output space before inference; the script will not download dataset files.")
    return {"ram_bytes": ram, "free_disk_bytes": disk.free, "disk_probe_path": str(disk_path), "torch": torch_info}


def _write_results(
    output_dir: Path,
    result,
    model_source: str,
    model_class_names: dict[int, str],
    vehicle_class_ids: dict[int, str],
    device: str | None,
    runtime: dict[str, Any],
    max_annotation_bytes: int,
) -> tuple[Path, Path, Path]:
    csv_columns = (
        "dataset",
        "image_name",
        "annotation_file",
        "model_source",
        "requested_sample_size",
        "evaluated_sample_size",
        "seed",
        "inference_seconds",
        "sample_size",
        "confidence_threshold",
        "iou_threshold",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "predicted_vehicle_count",
        "ground_truth_vehicle_count",
        "vehicle_count_mae",
        "category_mapping",
        "matching_rule",
        "notes",
    )
    rows: list[dict[str, object]] = []
    for image in result.evaluated_images:
        for evaluation in image.evaluations:
            row = asdict(evaluation)
            row.update({
                "dataset": "BDD100K",
                "annotation_file": image.annotation_file,
                "model_source": model_source,
                "requested_sample_size": result.requested_sample_size,
                "evaluated_sample_size": result.sample_size,
                "seed": result.seed,
                "inference_seconds": image.inference_seconds,
            })
            rows.append(row)
    csv_path = output_dir / "per_image_metrics.csv"
    csv_path.write_text(rows_to_csv(rows, csv_columns), encoding="utf-8", newline="")

    summary = {
        "dataset": "BDD100K",
        "dataset_root": str(result.dataset_root),
        "requested_sample_size": result.requested_sample_size,
        "evaluated_sample_size": result.sample_size,
        "seed": result.seed,
        "iou_threshold": result.iou_threshold,
        "confidence_thresholds": list(result.confidence_thresholds),
        "candidate_pair_count": result.candidate_pair_count,
        "attempted_pair_count": result.attempted_pair_count,
        "pair_diagnostics": result.pair_diagnostics,
        "metrics_by_confidence": result.aggregates,
        "skipped_by_reason": result.skipped_by_reason,
        "metric_scope": (
            "Micro precision/recall/F1 pooled over evaluated matched images; count MAE is the mean absolute "
            "per-image error for the mapped car/truck/bus/motorcycle classes. Not mAP."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "BDD100K local subset detection evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "BDD100K",
            "root": str(result.dataset_root),
            "pairing_rule": "unique case-insensitive image/JSON-or-JSONL basename (same stem)",
            "annotation_parser": "BDD100K per-image JSON with exactly one record; a missing image name is inferred only from a unique same-stem image pair; frames must contain exactly one frame",
            "max_per_image_annotation_bytes": max_annotation_bytes,
        },
        "sampling": {
            "requested_images": result.requested_sample_size,
            "evaluated_images": result.sample_size,
            "seed": result.seed,
            "method": "shuffle sorted unique basename pairs with random.Random(seed); skip invalid pairs until target or exhaustion",
            "candidate_pairs": result.candidate_pair_count,
            "attempted_pairs": result.attempted_pair_count,
            "pair_diagnostics": result.pair_diagnostics,
            "evaluated_image_names": [image.image_name for image in result.evaluated_images],
            "skipped_by_reason": result.skipped_by_reason,
            "skipped_examples": list(result.skipped_examples),
        },
        "model": {
            "source": model_source,
            "sha256_if_local_file": _sha256_if_file(model_source),
            "ultralytics_version": _version("ultralytics"),
            "device_requested": device or "Ultralytics automatic selection",
            "inference_defaults": {
                "image_size": DEFAULT_IMAGE_SIZE,
                "nms_iou_threshold": DEFAULT_NMS_IOU,
                "maximum_detections_per_image": DEFAULT_MAX_DETECTIONS,
                "minimum_confidence_for_shared_inference": min(result.confidence_thresholds),
            },
            "torch_version": runtime.get("torch", {}).get("version"),
            "class_names_from_loaded_checkpoint": {str(key): value for key, value in model_class_names.items()},
            "supported_vehicle_ids_from_loaded_checkpoint": {str(key): value for key, value in vehicle_class_ids.items()},
        },
        "evaluation": {
            "confidence_thresholds": list(result.confidence_thresholds),
            "iou_threshold": result.iou_threshold,
            "matching": "confidence-descending greedy, class-aware, one-to-one matching at IoU >= threshold",
            "supported_vehicle_classes": list(VEHICLE_CLASSES),
            "unmapped_categories": "excluded; annotation parser maps class names explicitly, never assumes shared model IDs",
            "metrics_by_confidence": result.aggregates,
            "limitations": [
                "Subset metrics are not full-dataset metrics or mAP.",
                "Images with corrupt files, malformed/unsupported labels, ambiguous matches, or out-of-bounds boxes are skipped and counted.",
                "A single still-image count is not physical density, traffic speed, queue length, or travel time.",
            ],
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "streamlit_version": _version("streamlit"),
            "ultralytics_version": _version("ultralytics"),
            "torch_version": runtime.get("torch", {}).get("version"),
            "cuda_available": runtime.get("torch", {}).get("cuda_available", False),
            "cuda_device_names": runtime.get("torch", {}).get("device_names", []),
            "cuda_device_memory_bytes": runtime.get("torch", {}).get("device_memory_bytes", []),
            "free_disk_bytes_before_inference": runtime.get("free_disk_bytes"),
            "disk_probe_path": runtime.get("disk_probe_path"),
            "ram_bytes_before_inference": runtime.get("ram_bytes"),
        },
        "artifacts": ["per_image_metrics.csv", "summary.json", "manifest.json"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, summary_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if args.sample_size <= 0:
        print("Error: --sample-size must be positive.", file=sys.stderr)
        return 2
    if not math.isfinite(args.iou_threshold) or not 0.0 < args.iou_threshold <= 1.0:
        print("Error: --iou must be finite and within (0, 1].", file=sys.stderr)
        return 2
    if (
        not math.isfinite(args.max_annotation_mb)
        or args.max_annotation_mb <= 0
        or int(args.max_annotation_mb * 1024 * 1024) <= 0
    ):
        print("Error: --max-annotation-mb must be finite and at least 1 byte.", file=sys.stderr)
        return 2

    confidence_thresholds = tuple(
        args.confidence_thresholds if args.confidence_thresholds is not None else DEFAULT_CONFIDENCE_THRESHOLDS
    )
    if not confidence_thresholds or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in confidence_thresholds
    ):
        print("Error: every --confidence value must be finite and within [0, 1].", file=sys.stderr)
        return 2

    discovery = discover_dataset("BDD100K", data_root)
    if not discovery.root_exists:
        print(f"Error: local BDD100K directory does not exist: {data_root}", file=sys.stderr)
        print("No dataset download was started. Mount/copy the already-downloaded data and pass --data-root.", file=sys.stderr)
        return 2
    pairs, pair_diagnostics = discover_bdd100k_image_annotation_pairs(discovery)
    torch_info = _torch_status()
    runtime = _print_preflight(data_root, len(pairs), pair_diagnostics, torch_info)
    print(f"  Discovered images: {discovery.image_count:,}")
    print(f"  Candidate annotation files: {discovery.annotation_file_count:,}")
    if args.preflight_only:
        print("Preflight only: annotation JSON and model weights were not loaded; no inference was run.")
        return 0 if pairs else 2
    if not pairs:
        print("Error: no unique same-stem per-image annotation pairs; inspect the extracted folder layout.", file=sys.stderr)
        return 2
    if not torch_info.get("installed"):
        print("Error: PyTorch is not available. Install/enable a suitable runtime before inference.", file=sys.stderr)
        return 2
    if _version("ultralytics") is None:
        print("Error: Ultralytics is not installed. Install the project's pinned version before inference.", file=sys.stderr)
        return 2
    if importlib.util.find_spec("cv2") is None:
        print("Error: OpenCV is not installed. Install a compatible OpenCV wheel before inference.", file=sys.stderr)
        return 2
    if args.device and (args.device.casefold().startswith("cuda") or args.device.isdigit()) and not torch_info.get("cuda_available"):
        print(f"Error: requested device {args.device!r}, but CUDA is not available in this runtime.", file=sys.stderr)
        return 2

    try:
        max_annotation_bytes = int(args.max_annotation_mb * 1024 * 1024)
        output_dir = _output_directory(args.output_dir)
        detector = YOLOVehicleDetector(args.model, device=args.device)
        print(
            f"Starting evaluation: target={args.sample_size}, seed={args.seed}, IoU={args.iou_threshold:.2f}, "
            f"confidence={list(confidence_thresholds)}, model={args.model!r}."
        )

        def report_progress(attempted: int, candidates: int, evaluated: int) -> None:
            if evaluated == 1 or evaluated % 10 == 0:
                print(f"  Progress: {evaluated}/{args.sample_size} valid images; {attempted}/{candidates} pairs attempted.")

        result = run_bdd100k_subset(
            root_path=data_root,
            detector=detector,
            sample_size=args.sample_size,
            seed=args.seed,
            confidence_thresholds=confidence_thresholds,
            iou_threshold=args.iou_threshold,
            progress=report_progress,
            max_annotation_bytes=max_annotation_bytes,
        )
        model_class_names = detector.model_class_names if result.sample_size else {}
        vehicle_class_ids = detector.vehicle_class_ids if result.sample_size else {}
        csv_path, summary_path, manifest_path = _write_results(
            output_dir=output_dir,
            result=result,
            model_source=args.model,
            model_class_names=model_class_names,
            vehicle_class_ids=vehicle_class_ids,
            device=args.device,
            runtime=runtime,
            max_annotation_bytes=max_annotation_bytes,
        )
    except Exception as exc:
        print(f"Evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"Evaluated: {result.sample_size}/{result.requested_sample_size} matched images")
    print(f"Skipped: {result.skipped_by_reason or 'none'}")
    for threshold, metrics in result.aggregates.items():
        print(
            f"  conf={threshold}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} precision={metrics['precision']} recall={metrics['recall']} "
            f"F1={metrics['f1']} count_MAE={metrics['vehicle_count_mae']}"
        )
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    if result.sample_size < result.requested_sample_size:
        print("Warning: fewer valid matched images were found than requested; see manifest skip diagnostics.")
    return 0 if result.sample_size else 2


if __name__ == "__main__":
    raise SystemExit(main())
