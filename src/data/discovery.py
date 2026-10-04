"""Filesystem-only discovery of local driving datasets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".json", ".jsonl", ".xml", ".txt"}
_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache",
    ".ultralytics", "runs", "outputs", "node_modules",
}


@dataclass(frozen=True)
class DatasetDiscovery:
    dataset_name: str
    root: Path
    root_exists: bool
    image_paths: tuple[Path, ...] = ()
    annotation_paths: tuple[Path, ...] = ()
    annotation_format_counts: dict[str, int] = field(default_factory=dict)
    scan_errors: tuple[str, ...] = ()

    @property
    def image_count(self) -> int:
        return len(self.image_paths)

    @property
    def annotation_file_count(self) -> int:
        return len(self.annotation_paths)

    @property
    def ready_for_image_inference(self) -> bool:
        return self.root_exists and self.image_count > 0


def discover_dataset(dataset_name: str, root_path: str | Path) -> DatasetDiscovery:
    """Discover images and candidate JSON/XML/TXT annotation files recursively.

    Discovery does not infer that a file is a valid annotation. Schema parsing,
    image matching, and malformed-file reporting are handled by the readers.
    """
    root = Path(root_path).expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = root.absolute()
    if not root.is_dir():
        return DatasetDiscovery(
            dataset_name=dataset_name,
            root=root,
            root_exists=False,
            scan_errors=(f"Dataset directory does not exist or is not a directory: {root}",),
        )

    image_paths: list[Path] = []
    annotation_paths: list[Path] = []
    errors: list[str] = []
    format_counts: dict[str, int] = {}

    def on_walk_error(exc: OSError) -> None:
        errors.append(f"Could not inspect {getattr(exc, 'filename', root)}: {exc}")

    for current, directories, filenames in os.walk(root, onerror=on_walk_error):
        directories[:] = sorted(name for name in directories if name not in _IGNORED_DIRS)
        for filename in filenames:
            path = Path(current) / filename
            suffix = path.suffix.casefold()
            if suffix in IMAGE_SUFFIXES:
                image_paths.append(path)
            elif suffix in ANNOTATION_SUFFIXES:
                # YOLO class-name files are parser metadata, not per-image
                # annotation files and should not inflate the annotation count.
                if suffix == ".txt" and filename.casefold() in {
                    "classes.txt", "classes.names", "obj.names", "labels.names"
                }:
                    continue
                annotation_paths.append(path)
                format_name = suffix.lstrip(".").upper()
                format_counts[format_name] = format_counts.get(format_name, 0) + 1

    image_paths.sort(key=lambda item: item.as_posix().casefold())
    annotation_paths.sort(key=lambda item: item.as_posix().casefold())
    return DatasetDiscovery(
        dataset_name=dataset_name,
        root=root,
        root_exists=True,
        image_paths=tuple(image_paths),
        annotation_paths=tuple(annotation_paths),
        annotation_format_counts=dict(sorted(format_counts.items())),
        scan_errors=tuple(errors),
    )
