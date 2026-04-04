"""
Template-based reasoning trace generation for the analogy dataset.

Two phrasings (variant 0 and variant 1) are defined for each text field.
Generating both variants per sample doubles training variety.

Reasoning flow:
  1. pre_visual_text_think  — observe source→target rotation, apply to query
                              [intermediate image shown after this text]
  2. post_visual_text_think — one positive statement for the correct option,
                              empty strings for the other three
  3. text_think             — restatement of the conclusion (similar to pre,
                              different phrasing)
"""

from __future__ import annotations
import re
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_angle(transform_description: str) -> str:
    """
    Pull the rotation degrees out of a transform description string.
    Works for 'rotation' and 'combined' types; returns '' if not found.
    Examples:
        '90° clockwise rotation'                           → '90°'
        '180° clockwise rotation and translation of …'    → '180°'
    """
    m = re.search(r'(\d+)°', transform_description)
    return f"{m.group(1)}°" if m else ""


# ---------------------------------------------------------------------------
# Text templates  (two variants per field)
# ---------------------------------------------------------------------------

_PRE_TEMPLATES = [
    # variant 0
    (
        "Looking at the reference transformation, the source object appears to have been "
        "rotated {angle} clockwise to produce the target object. "
        "Applying this same {angle} clockwise rotation to the query object gives us "
        "the expected result shown above."
    ),
    # variant 1
    (
        "Observing the example, the source object has undergone a {angle} clockwise rotation "
        "to become the target object. "
        "If we apply this same rotation to the query object, we obtain the shape illustrated above."
    ),
]

_CORRECT_OPT_TEMPLATES = [
    # variant 0 — placed in post_visual_text_think[correct_idx]
    "This option resembles the query object after a {angle} clockwise rotation — consistent with the reference transformation.",
    # variant 1
    "The shape here matches the query object rotated {angle} clockwise, in line with the source-to-target transformation.",
]

_TEXT_THINK_TEMPLATES = [
    # variant 0
    (
        "The reference transformation is a {angle} clockwise rotation. "
        "The query object with the same rotation applied matches option ({answer})."
    ),
    # variant 1
    (
        "A {angle} clockwise rotation maps the source object to the target object. "
        "Applying this rotation to the query object yields a result that corresponds to option ({answer})."
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fill_reasoning_traces(
    sample: Dict[str, Any],
    variant: int = 0,
) -> Dict[str, Any]:
    """
    Return a filled reasoning_traces dict for one analogy sample.

    Args:
        sample:  Output dict from generate_analogy_sample (must contain
                 'answer', 'transform_description', 'transform_type').
        variant: 0 or 1 — which phrasing template to use.

    Returns:
        reasoning_traces dict ready to embed in the SFT JSON record.
        'intermediate_img_path' is left as "" — caller must set it after
        rendering and saving the intermediate image.
    """
    v = variant % 2
    answer = sample["answer"]                            # e.g. "b"
    desc   = sample.get("transform_description", "")
    angle  = _extract_angle(desc) or "some degrees"     # e.g. "90°"

    fmt = dict(angle=angle, answer=answer)

    pre   = _PRE_TEMPLATES[v].format(**fmt)
    c_opt = _CORRECT_OPT_TEMPLATES[v].format(**fmt)
    think = _TEXT_THINK_TEMPLATES[v].format(**fmt)

    # post_visual_text_think: one entry per option (a/b/c/d).
    # Only the correct option slot is filled; others are empty.
    correct_idx = "abcd".index(answer)
    post = ["", "", "", ""]
    post[correct_idx] = c_opt

    return {
        "pre_visual_text_think":    pre,
        "intermediate_img_path":    "",   # filled by create_dataset.py
        "post_visual_text_think":   post,
        "text_think":               think,
        "answer":                   answer,
    }


def fill_both_variants(sample: Dict[str, Any]):
    """Convenience: return (trace_v0, trace_v1) for the same sample."""
    return fill_reasoning_traces(sample, 0), fill_reasoning_traces(sample, 1)
