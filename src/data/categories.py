"""Explicit aliases for the four vehicle classes used by this prototype.

Annotation category strings and model class names are resolved by name, never by
assuming that class IDs are shared between datasets and model checkpoints.
"""
from __future__ import annotations

import re

VEHICLE_CLASSES: tuple[str, ...] = ("car", "truck", "bus", "motorcycle")

_ALIASES = {
    "car": "car",
    "automobile": "car",
    "truck": "truck",
    "lorry": "truck",
    "bus": "bus",
    "motor": "motorcycle",  # BDD100K's detection category name
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motorized two wheeler": "motorcycle",
    "motorized two-wheeler": "motorcycle",
    "two wheeler": "motorcycle",
    "two-wheeler": "motorcycle",
}


def normalize_category_name(value: object) -> str:
    """Normalize whitespace/case while retaining useful word separators."""
    text = str(value or "").strip().casefold()
    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_vehicle_class(value: object) -> str | None:
    """Return a supported canonical class for a recognized category name."""
    name = normalize_category_name(value)
    if name in _ALIASES:
        return _ALIASES[name]
    # Normalize punctuation variants without making fuzzy guesses.
    compact = re.sub(r"[^a-z0-9]+", " ", name).strip()
    return _ALIASES.get(compact)
