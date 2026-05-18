"""
Shared transformation utilities for the analogy dataset.

Provides:
  - ROTATION_ANGLES, TRANSFORM_DESCRIPTIONS: shared constants
  - _apply_offset, _pick_similar_shape: helper functions

These are imported by analogy_simulator.py.
"""

from __future__ import annotations

import random

from .pieces import SHAPES, SHAPES_BY_FAMILY

# ---------------------------------------------------------------------------
# Transform constants
# ---------------------------------------------------------------------------

TRANSFORM_DESCRIPTIONS = {
    "rotation":    "{angle}° clockwise rotation",
    "translation": "translation of {dr:+d} row(s) and {dc:+d} column(s)",
    "combined":    "{angle}° clockwise rotation and translation of {dr:+d} row(s) and {dc:+d} column(s)",
}

ROTATION_ANGLES = {1: 90, 2: 180, 3: 270}   # rotation index offset → degrees


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_offset(cells, offset):
    dr, dc = offset
    return [(r + dr, c + dc) for r, c in cells]


def _pick_similar_shape(shape: dict, rng: random.Random) -> dict:
    """Pick a different shape from the same family, or any other shape as fallback."""
    same_family = [s for s in SHAPES_BY_FAMILY.get(shape["family"], []) if s["name"] != shape["name"]]
    if same_family:
        return rng.choice(same_family)
    return rng.choice([s for s in SHAPES if s["name"] != shape["name"]])
