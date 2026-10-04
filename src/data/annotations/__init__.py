"""Supported dataset annotation readers."""
from src.data.annotations.common import AnnotationScan, GroundTruthBox
from src.data.annotations.scan import scan_annotations

__all__ = ["AnnotationScan", "GroundTruthBox", "scan_annotations"]
