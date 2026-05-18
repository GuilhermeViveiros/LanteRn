"""
Template-based reasoning trace generation for the analogy dataset.

15 phrasings are defined for each text field; one is chosen randomly per
sample so the final JSON has exactly one record per sample with diverse
language variety across the dataset.

Reasoning flow:
  1. pre_visual_text_think  — observe source→target rotation, apply to query
                              [intermediate image shown after this text]
  2. post_visual_text_think — one positive statement for the correct option,
                              empty strings for the other three
  3. text_think             — restatement of the conclusion
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_angle(transform_description: str) -> str:
    m = re.search(r'(\d+)°', transform_description)
    return f"{m.group(1)}°" if m else ""


# ---------------------------------------------------------------------------
# Text templates  (15 variants per field)
# ---------------------------------------------------------------------------

_PRE_TEMPLATES = [
    "Looking at the reference transformation, the source object appears to have been "
    "rotated {angle} clockwise to produce the target object. "
    "If we imagine applying the same {angle} clockwise rotation to the query object,",

    "Observing the example, the source object has undergone a {angle} clockwise rotation "
    "to become the target object. "
    "If we mentally apply this same {angle} clockwise rotation to the query object,",

    "The reference pair shows a {angle} clockwise rotation from source to target. "
    "Applying that same rotation to the query object in my mind,",

    "From the example, the transformation is a {angle} clockwise turn. "
    "Rotating the query object {angle} clockwise in imagination,",

    "The source object was rotated {angle} clockwise to get the target. "
    "Doing the same to the query object mentally,",

    "I can see that the reference transformation is a {angle} clockwise rotation. "
    "Visualising this applied to the query shape,",

    "The example demonstrates a {angle} clockwise rotation. "
    "If I rotate the query object by {angle} in the same direction,",

    "By examining the A→B pair, I identify a {angle} clockwise rotation. "
    "Applying this mentally to the query object,",

    "The transformation from source to target is a {angle} clockwise rotation. "
    "Performing this rotation on the query object in my head,",

    "Looking at how A maps to B, I see a {angle} clockwise rotation. "
    "Imagining the query object undergoing the same rotation,",

    "The source-to-target change is a {angle} clockwise turn. "
    "Mentally spinning the query object by {angle} clockwise,",

    "A {angle} clockwise rotation takes the source to the target. "
    "If I apply that same {angle} clockwise spin to the query,",

    "The reference shows the shape being rotated {angle} to the right. "
    "Turning the query object {angle} to the right in my mind,",

    "Analysing the example, the transformation is a {angle} clockwise rotation. "
    "Imagining the same rotation applied to the query shape,",

    "The pattern shows a {angle} clockwise rotation from source to target. "
    "Applying this transformation to the query object mentally,",
]

_CORRECT_OPT_TEMPLATES = [
    "The mentally rotated shape resembles option ({answer}).",
    "The result of this mental rotation corresponds to option ({answer}).",
    "After rotating the query object, the resulting shape lines up with option ({answer}).",
    "The rotated query shape matches option ({answer}).",
    "Option ({answer}) matches the mentally rotated query object.",
    "The query object after rotation looks like option ({answer}).",
    "This mental image corresponds to option ({answer}).",
    "The shape produced by this rotation matches option ({answer}).",
    "After the rotation, the query object resembles option ({answer}).",
    "The mentally rotated query object lines up with option ({answer}).",
    "Option ({answer}) is consistent with the rotated query shape.",
    "The result of rotating the query matches option ({answer}).",
    "Option ({answer}) corresponds to the query object after the mental rotation.",
    "The rotated query shape and option ({answer}) are the same.",
    "Option ({answer}) is what the query object looks like after the rotation.",
]

_TEXT_THINK_TEMPLATES = [
    "The reference transformation is a {angle} clockwise rotation. "
    "The imagined rotation of the query object most closely matches option ({answer}).",

    "A {angle} clockwise rotation maps the source object to the target object. "
    "Imagining this same rotation applied to the query object, the result corresponds to option ({answer}).",

    "The transformation is a {angle} clockwise rotation, and the query object rotated accordingly matches option ({answer}).",

    "Given the {angle} clockwise rotation in the example, the correct answer for the query object is option ({answer}).",

    "Rotating the query object {angle} clockwise — same as the reference — gives the shape in option ({answer}).",

    "The {angle} clockwise rotation applied to the query object produces the shape shown in option ({answer}).",

    "Since the transformation is a {angle} clockwise rotation, the query object should look like option ({answer}).",

    "After a {angle} clockwise rotation, the query object matches option ({answer}), consistent with the example.",

    "The reference rotation is {angle} clockwise; applying it to the query yields option ({answer}).",

    "Applying the {angle} clockwise rotation to the query object results in option ({answer}).",

    "The {angle} clockwise rotation seen in A→B, when applied to the query, gives option ({answer}).",

    "Option ({answer}) is the result of rotating the query object {angle} clockwise, matching the pattern.",

    "The query object rotated {angle} clockwise matches option ({answer}), as expected from the reference.",

    "Following the {angle} clockwise rotation pattern, the query object transformed is option ({answer}).",

    "The correct answer is option ({answer}): the query object after a {angle} clockwise rotation.",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fill_reasoning_traces(
    sample: dict[str, Any],
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """
    Return a filled reasoning_traces dict for one analogy sample,
    picking templates randomly from the 15 available phrasings.

    Args:
        sample: Output dict from generate_analogy_sample.
        rng:    Optional seeded random.Random for reproducibility.

    Returns:
        reasoning_traces dict. 'intermediate_img_path' is left as ""
        — caller must set it after saving the intermediate image.
    """
    if rng is None:
        rng = random.Random()

    answer = sample["answer"]
    desc   = sample.get("transform_description", "")
    angle  = _extract_angle(desc) or "some degrees"

    fmt = dict(angle=angle, answer=answer)

    pre   = rng.choice(_PRE_TEMPLATES).format(**fmt)
    c_opt = rng.choice(_CORRECT_OPT_TEMPLATES).format(**fmt)
    think = rng.choice(_TEXT_THINK_TEMPLATES).format(**fmt)

    correct_idx = "abcd".index(answer)
    post = ["", "", "", ""]
    post[correct_idx] = c_opt

    return {
        "pre_visual_text_think":  pre,
        "intermediate_img_path":  "",
        "post_visual_text_think": post,
        "text_think":             think,
        "answer":                 answer,
    }
