"""Ground-truth-dependent evaluation utilities."""
from src.evaluation.detection_metrics import ImageEvaluation, evaluate_image, intersection_over_union

__all__ = ["ImageEvaluation", "evaluate_image", "intersection_over_union"]
