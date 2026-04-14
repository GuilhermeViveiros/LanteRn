"""
Sample generator for the visual analogy dataset:

    "Image A is to image B as image C is to which of the following options?"

The transformation that maps A → B (rotation, translation, or combined) is the
same transformation that must be applied to C to produce the correct answer.
A and C are always different shapes, making the analogy non-trivial.

Composite image layout:

    +-------+  →  +-------+       +-------+
    |  (A)  |      |  (B)  |  ::   |  (C)  |
    +-------+      +-------+       +-------+
               Options for  C → ?
         (a)       (b)       (c)       (d)
"""

from __future__ import annotations

import random
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageDraw

from .pieces import (
    SHAPES, SHAPES_BY_FAMILY,
    fits_in_grid, valid_offsets,
    _normalize, _rotate_90cw,
)

# Palette of visually distinct colors used to randomize option panel colors.
# Each sample independently picks 4 of these without replacement so no two
# options share a color, and the correct answer has no systematic color bias.
_OPTION_COLOR_PALETTE = [
    (220,  50,  50),   # red
    ( 50, 110, 230),   # blue
    ( 50, 200,  70),   # green
    (230, 180,  35),   # amber
    (175,  50, 225),   # violet
    ( 35, 205, 205),   # cyan
    (230,  95,  35),   # orange
    (205,  50, 165),   # magenta
    ( 50, 180, 130),   # teal
    (240, 230,  60),   # yellow
    (130,  80, 230),   # indigo
    (230, 130,  80),   # salmon
    ( 80, 230, 180),   # mint
    (230,  80, 120),   # rose
    (100, 200, 240),   # sky blue
    (200, 230,  80),   # lime
    (240, 160,  60),   # tangerine
    (160,  60, 200),   # purple
    ( 60, 160, 200),   # steel blue
    (200,  90,  90),   # brick
    ( 90, 200, 130),   # seafoam
    (230, 200, 100),   # gold
    (100, 130, 230),   # periwinkle
    (230, 100, 180),   # hot pink
]
from .renderer import draw_piece_on_grid, draw_piece_standalone, render_rotation_strip, _get_font, _PAD, _LABEL_HEIGHT, _SECTION_LABEL_HEIGHT
from .simulator import (
    ROTATION_ANGLES, TRANSFORM_DESCRIPTIONS,
    _apply_offset,
    _pick_similar_shape,
)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _adjacent_cells(cells: list) -> List[Tuple[int, int]]:
    """Return all orthogonally adjacent cells that are NOT already in the shape."""
    occupied = set(cells)
    adj = set()
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb not in occupied:
                adj.add(nb)
    return list(adj)


def _is_connected(cells: list) -> bool:
    """BFS connectivity check."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    visited = {cells[0]}
    queue = [cells[0]]
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(cells)


def _is_filled_rectangle(cells: list) -> bool:
    """Return True if cells fully fill an m×n bounding box (m,n ≥ 2) — i.e. a solid block."""
    norm = list(_normalize(cells))
    rows = [r for r, c in norm]
    cols = [c for r, c in norm]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return h >= 2 and w >= 2 and len(norm) == h * w


def _perturb_cells(cells: list, rng: random.Random, mode: str = "random") -> Optional[list]:
    """
    Return a perturbed version of cells by adding or removing exactly one cell.

    mode: "add" | "remove" | "random"
    Returns None if no valid perturbation exists.
    """
    if mode == "random":
        choices = []
        if len(cells) > 2:
            choices.append("remove")
        choices.append("add")
        mode = rng.choice(choices)

    if mode == "add":
        adj = _adjacent_cells(cells)
        if not adj:
            return None
        new_cell = rng.choice(adj)
        return list(cells) + [new_cell]

    elif mode == "remove":
        if len(cells) <= 1:
            return None
        removable = [c for c in cells if _is_connected([x for x in cells if x != c])]
        if not removable:
            return None
        removed = rng.choice(removable)
        return [c for c in cells if c != removed]

    return None


# ---------------------------------------------------------------------------
# Analogy composite image builder
# ---------------------------------------------------------------------------

def build_analogy_composite(
    img_A: Image.Image,
    img_B: Image.Image,
    img_C: Image.Image,
    options: List[Image.Image],
    labels: List[str] = None,
) -> Tuple[Image.Image, List[List[int]]]:
    """
    Layout:

        [A]  "is rotated to"  [B]      ← standalone objects, no grid
        ──────────── separator ─────────────
        "As (C) is rotated to"
             [ C on grid, centred ]
        (a)      (b)      (c)      (d)
        [ a ]    [ b ]    [ c ]    [ d ]

    Returns (composite_image, bboxes) where bboxes = [[x1,y1,x2,y2], ...]
    for each of the 4 option panels in label order.
    """
    assert len(options) == 4
    if labels is None:
        labels = ["a", "b", "c", "d"]

    sa_w, sa_h = img_A.size       # standalone size — A and B
    c_w,  c_h  = img_C.size       # C on grid
    op_w, op_h = options[0].size  # option panel size

    font_lbl  = _get_font(26)
    font_text = _get_font(32)     # large text throughout

    bg         = (18, 18, 22)
    text_color = (210, 210, 220)
    sep_color  = (55, 55, 70)

    GAP = 20   # horizontal gap between elements in top row / options row
    PAD = 16   # outer padding
    SEP = 2    # separator height
    LBL = 42   # label height above options

    # ── measure text ─────────────────────────────────────────────────────────
    rot_text = "is rotated to"
    tw  = int(font_text.getlength(rot_text)) if hasattr(font_text, "getlength") else 120
    th  = int(font_text.size) if hasattr(font_text, "size") else 20

    # Inline row: "As" + C + "is rotated to"
    pre_w_est  = int(font_text.getlength("As"))           if hasattr(font_text, "getlength") else 30
    post_w_est = int(font_text.getlength("is rotated to")) if hasattr(font_text, "getlength") else 130
    inline_w   = pre_w_est + 14 + c_w + 14 + post_w_est

    # ── canvas width ──────────────────────────────────────────────────────────
    top_w      = sa_w + GAP + tw + GAP + sa_w
    opts_row_w = 4 * op_w + 3 * GAP
    inner_w    = max(top_w, inline_w, opts_row_w)
    total_w    = inner_w + 2 * PAD

    # ── canvas height ─────────────────────────────────────────────────────────
    inline_row_h = max(th, c_h)
    total_h = (PAD
               + sa_h + PAD              # top row (A → B)
               + SEP + PAD               # separator
               + inline_row_h + PAD      # inline "As [C] is rotated to"
               + LBL + op_h + PAD)       # options row

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw   = ImageDraw.Draw(canvas)

    # ── Top row: A  "is rotated to"  B ───────────────────────────────────────
    row_x = PAD + (inner_w - top_w) // 2
    cy    = PAD

    canvas.paste(img_A, (row_x, cy))
    tx = row_x + sa_w + GAP
    draw.text((tx, cy + (sa_h - th) // 2), rot_text, fill=text_color, font=font_text)
    canvas.paste(img_B, (tx + tw + GAP, cy))
    cy += sa_h + PAD

    # ── Separator ────────────────────────────────────────────────────────────
    draw.rectangle([PAD, cy, total_w - PAD, cy + SEP - 1], fill=sep_color)
    cy += SEP + PAD

    # ── Inline row: "As"  [C image]  "is rotated to" ─────────────────────────
    pre_text  = "As"
    post_text = "is rotated to"
    pre_w  = int(font_text.getlength(pre_text))  if hasattr(font_text, "getlength") else 30
    post_w = int(font_text.getlength(post_text)) if hasattr(font_text, "getlength") else 130
    img_gap = 14   # gap between text and image

    inline_row_w = pre_w + img_gap + c_w + img_gap + post_w
    row_x3 = PAD + (inner_w - inline_row_w) // 2
    row_h3 = max(th, c_h)   # row height = tallest element

    # "As"
    draw.text((row_x3, cy + (row_h3 - th) // 2), pre_text, fill=text_color, font=font_text)
    # C image centred vertically
    canvas.paste(img_C, (row_x3 + pre_w + img_gap, cy + (row_h3 - c_h) // 2))
    # "is rotated to"
    draw.text((row_x3 + pre_w + img_gap + c_w + img_gap, cy + (row_h3 - th) // 2),
              post_text, fill=text_color, font=font_text)
    cy += row_h3 + PAD

    # ── Options row: (a) (b) (c) (d) with labels above ───────────────────────
    opts_x = PAD + (inner_w - opts_row_w) // 2
    bboxes: List[List[int]] = []
    for i, (opt_img, lbl) in enumerate(zip(options, labels)):
        ox = opts_x + i * (op_w + GAP)
        lt = f"({lbl})"
        lw = int(font_lbl.getlength(lt)) if hasattr(font_lbl, "getlength") else 20
        draw.text((ox + (op_w - lw) // 2, cy + 4), lt, fill=text_color, font=font_lbl)
        oy = cy + LBL
        canvas.paste(opt_img, (ox, oy))
        bboxes.append([ox, oy, ox + op_w - 1, oy + op_h - 1])

    return canvas, bboxes


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------

def generate_analogy_sample(
    shape_A: dict,
    shape_C: dict,
    transform_type: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    cell_size: int = 32,
    rng: Optional[random.Random] = None,
    force_correct_pos: Optional[int] = None,
    ref_rot_A_idx: Optional[int] = None,
    ref_rot_C_idx: Optional[int] = None,
    rot_step_override: Optional[int] = None,
    bw_intermediate: bool = False,
) -> Dict[str, Any]:
    """
    Generate one analogy sample:  A : B :: C : ?

    Args:
        shape_A:          Shape used for the A/B pair.
        shape_C:          Shape used for the C/? pair (must differ from shape_A).
        transform_type:   "rotation" | "translation" | "combined".
        cell_size:        Pixel size of each grid cell.
        rng:              Seeded random.Random.
        ref_rot_A_idx:    Pin A's starting rotation index (for enumeration-based generation).
        ref_rot_C_idx:    Pin C's starting rotation index (for enumeration-based generation).
        rot_step_override: Pin the rotation step (for enumeration-based generation).

    Returns a dict with composite_img, bboxes, answer, metadata.
    """
    if rng is None:
        rng = random.Random()

    rots_A = shape_A["rotations"]
    rots_C = shape_C["rotations"]
    n_rots_A = len(rots_A)
    n_rots_C = len(rots_C)

    # Fall back if rotation is impossible for either shape
    effective_type = transform_type
    if effective_type in ("rotation", "combined") and (n_rots_A == 1 or n_rots_C == 1):
        effective_type = "translation"

    # --- 1. Pick reference rotation + placement for A ---
    ref_rot_A = ref_rot_A_idx if ref_rot_A_idx is not None else rng.randrange(n_rots_A)
    cells_A = rots_A[ref_rot_A]
    off_A = rng.choice(valid_offsets(cells_A, grid_rows, grid_cols))

    # --- 2. Determine the transformation (same will be applied to C) ---
    gt_rot_A = ref_rot_A
    delta = (0, 0)
    angle = 0
    transform_desc = ""

    if effective_type == "rotation":
        rot_step = rot_step_override if rot_step_override is not None else rng.choice(list(range(1, n_rots_A)))
        gt_rot_A = (ref_rot_A + rot_step) % n_rots_A
        angle = ROTATION_ANGLES.get(rot_step, rot_step * 90)
        transform_desc = TRANSFORM_DESCRIPTIONS["rotation"].format(angle=angle)

    elif effective_type == "translation":
        cands = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if (dr, dc) == (0, 0):
                    continue
                if fits_in_grid(cells_A, (off_A[0] + dr, off_A[1] + dc), grid_rows, grid_cols):
                    cands.append((dr, dc))
        if not cands:
            # fallback: any different offset
            alt = [o for o in valid_offsets(cells_A, grid_rows, grid_cols) if o != off_A]
            if not alt:
                return generate_analogy_sample(shape_A, shape_C, "rotation",
                                               grid_rows, grid_cols,
                                               panel_cell_size, opt_cell_size, rng)
            delta = (alt[0][0] - off_A[0], alt[0][1] - off_A[1])
        else:
            delta = rng.choice(cands)
        transform_desc = TRANSFORM_DESCRIPTIONS["translation"].format(dr=delta[0], dc=delta[1])

    elif effective_type == "combined":
        rot_step = rng.choice(list(range(1, n_rots_A)))
        gt_rot_A = (ref_rot_A + rot_step) % n_rots_A
        angle = ROTATION_ANGLES.get(rot_step, rot_step * 90)
        cands = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if fits_in_grid(rots_A[gt_rot_A],
                                (off_A[0] + dr, off_A[1] + dc), grid_rows, grid_cols):
                    cands.append((dr, dc))
        delta = rng.choice(cands) if cands else (0, 0)
        transform_desc = TRANSFORM_DESCRIPTIONS["combined"].format(
            angle=angle, dr=delta[0], dc=delta[1]
        )

    # B = A after transformation
    cells_B = rots_A[gt_rot_A]
    off_B_base = valid_offsets(cells_B, grid_rows, grid_cols)
    if delta != (0, 0):
        new_off_B = (off_A[0] + delta[0], off_A[1] + delta[1])
        off_B = new_off_B if fits_in_grid(cells_B, new_off_B, grid_rows, grid_cols) \
                else rng.choice(off_B_base)
    else:
        off_B = rng.choice(off_B_base)

    # --- 3. Apply the SAME transformation to C ---
    # Rotation step is relative; we apply the same rot_step to shape_C's rotations.
    # For translation, we apply the same (dr, dc) delta.
    ref_rot_C = ref_rot_C_idx if ref_rot_C_idx is not None else rng.randrange(n_rots_C)
    cells_C = rots_C[ref_rot_C]
    off_C = rng.choice(valid_offsets(cells_C, grid_rows, grid_cols))

    # Map rot_step onto shape_C's rotation space
    rot_step_for_C = (gt_rot_A - ref_rot_A) % n_rots_A  # steps taken for A
    # Normalise to C's rotation count
    gt_rot_C = (ref_rot_C + rot_step_for_C) % n_rots_C
    cells_gt_C = rots_C[gt_rot_C]

    # Degenerate: shape_C looks identical before and after (symmetric + wrap).
    # The correct answer would be visually indistinguishable from C itself.
    # Retry with a rotation step that actually changes C's appearance.
    if _normalize(cells_gt_C) == _normalize(cells_C) and effective_type in ("rotation", "combined"):
        # Force a rotation step that produces a visually different result for C
        valid_steps = [i for i in range(1, n_rots_C)
                       if _normalize(rots_C[(ref_rot_C + i) % n_rots_C]) != _normalize(cells_C)]
        if not valid_steps:
            # shape_C has only one unique visual state — skip entirely
            raise ValueError(f"shape_C '{shape_C['name']}' has no visually distinct rotation; skip sample")
        new_step = rng.choice(valid_steps)
        gt_rot_C  = (ref_rot_C + new_step) % n_rots_C
        cells_gt_C = rots_C[gt_rot_C]
        # Recompute angle / transform_desc for the adjusted step
        angle = ROTATION_ANGLES.get(new_step, new_step * 90)
        transform_desc = TRANSFORM_DESCRIPTIONS["rotation"].format(angle=angle)

    if delta != (0, 0):
        new_off_gt_C = (off_C[0] + delta[0], off_C[1] + delta[1])
        off_gt_C = new_off_gt_C if fits_in_grid(cells_gt_C, new_off_gt_C, grid_rows, grid_cols) \
                   else rng.choice(valid_offsets(cells_gt_C, grid_rows, grid_cols))
    else:
        off_gt_C = rng.choice(valid_offsets(cells_gt_C, grid_rows, grid_cols))

    # --- 4. Generate 3 distractors for the ? slot ---
    def _also_correct(cells_d, offset_d, shape_d) -> bool:
        """True if this option is visually equivalent to the correct answer."""
        if effective_type == "rotation":
            return (shape_d["name"] == shape_C["name"] and
                    _normalize(cells_d) == _normalize(cells_gt_C))
        else:
            abs_d  = frozenset(_apply_offset(cells_d, offset_d))
            abs_gt = frozenset(_apply_offset(cells_gt_C, off_gt_C))
            return abs_d == abs_gt

    def _vkey(cells, offset, shape_obj):
        return (frozenset(_apply_offset(cells, offset)), shape_obj["name"])

    seen = {_vkey(cells_gt_C, off_gt_C, shape_C)}
    distractors = []
    distractor_transforms = []  # parallel to distractors: transform description for each

    def _add(cells, offset, shape_obj, desc: str = "unknown") -> bool:
        if not fits_in_grid(cells, offset, grid_rows, grid_cols):
            return False
        if _also_correct(cells, offset, shape_obj):
            return False
        if _is_filled_rectangle(cells):
            return False
        vk = _vkey(cells, offset, shape_obj)
        if vk in seen:
            return False
        seen.add(vk)
        distractors.append((cells, offset, shape_obj["color"]))
        distractor_transforms.append(desc)
        return True

    # Distractors must ALWAYS show shape_C at a different rotation (or a clearly
    # different shape). We never use same-rotation-different-position distractors:
    # at small panel sizes they look visually identical to the correct answer and
    # create ambiguous / multiple-correct samples.

    # Fake shape-obj for perturbed shapes (cells differ from any named shape)
    _perturb_obj = {"name": "__perturbed__", "color": shape_C["color"]}

    # --- Priority 1: every other unique rotation of shape C ---
    # Shuffled so we don't always pick in the same order.
    other_rot_indices = [i for i in range(n_rots_C) if i != gt_rot_C]
    rng.shuffle(other_rot_indices)
    for ri in other_rot_indices:
        if len(distractors) >= 3:
            break
        alt_cells = rots_C[ri]
        alt_offs = valid_offsets(alt_cells, grid_rows, grid_cols)
        if alt_offs:
            d_step = (ri - ref_rot_C) % n_rots_C
            if d_step == 0:
                d_desc = "identity"
            else:
                d_angle = ROTATION_ANGLES.get(d_step, d_step * 90)
                d_desc = TRANSFORM_DESCRIPTIONS["rotation"].format(angle=d_angle)
            _add(alt_cells, rng.choice(alt_offs), shape_C, desc=d_desc)

    # --- Priority 1.5: structurally perturbed shape_C (add or remove one cell) ---
    # These near-miss distractors are shown in shape_C's color and look very similar
    # to the correct answer, making the task harder. We try one "add" and one
    # "remove" perturbation to avoid burning all three distractor slots on this.
    for mode in rng.sample(["add", "remove"], 2):
        if len(distractors) >= 3:
            break
        perturbed = _perturb_cells(list(cells_gt_C), rng, mode=mode)
        if perturbed is None:
            continue
        perturbed_norm = list(_normalize(perturbed))
        offs = valid_offsets(perturbed_norm, grid_rows, grid_cols)
        if offs:
            _add(perturbed_norm, rng.choice(offs), _perturb_obj, desc="perturbed")

    # --- Priority 2: visually similar shapes (same family), different rotation ---
    # Fill remaining slots (happens when shape_C has ≤ 2 unique rotations).
    while len(distractors) < 3:
        sim = _pick_similar_shape(shape_C, rng)
        # Pick a rotation for the similar shape — prefer one that doesn't match gt_rot_C's look
        sim_rot_indices = list(range(len(sim["rotations"])))
        rng.shuffle(sim_rot_indices)
        added = False
        for ri in sim_rot_indices:
            sim_cells = sim["rotations"][ri]
            sim_offs = valid_offsets(sim_cells, grid_rows, grid_cols)
            if sim_offs and _add(sim_cells, rng.choice(sim_offs), sim, desc="other_shape"):
                added = True
                break
        if not added:
            break  # give up if no similar shape works

    # --- Fallback: any random shape ---
    attempts = 0
    while len(distractors) < 3 and attempts < 80:
        attempts += 1
        rs = rng.choice(SHAPES)
        ri = rng.randrange(len(rs["rotations"]))
        rc = rs["rotations"][ri]
        ro = valid_offsets(rc, grid_rows, grid_cols)
        if ro:
            _add(rc, rng.choice(ro), rs, desc="other_shape")

    distractors = distractors[:3]
    distractor_transforms = distractor_transforms[:3]

    # --- 5. Shuffle options ---
    all_opts = [(cells_gt_C, off_gt_C, shape_C["color"])] + distractors
    all_opt_transforms = [transform_desc] + distractor_transforms
    if force_correct_pos is not None:
        # Place correct answer at the requested slot; shuffle distractors among the rest.
        pos = force_correct_pos % 4
        dist_indices = list(range(1, 4))
        rng.shuffle(dist_indices)
        ordered = [None] * 4
        ordered[pos] = all_opts[0]
        remaining_slots = [i for i in range(4) if i != pos]
        for slot, di in zip(remaining_slots, dist_indices):
            ordered[slot] = all_opts[di]
        shuffled = ordered
        correct_pos = pos
        slot_to_src = {pos: 0}
        slot_to_src.update(zip(remaining_slots, dist_indices))
    else:
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        shuffled = [all_opts[i] for i in indices]
        slot_to_src = {slot: src for slot, src in enumerate(indices)}
    answer = "abcd"[correct_pos]
    option_transforms = {"abcd"[slot]: all_opt_transforms[src]
                         for slot, src in slot_to_src.items()}

    # --- 6. Render ---
    # A, B, C all at the same cell_size so both rows are structurally identical.
    # Options are rendered smaller; build_analogy_composite rescales them to fill
    # the 2×2 grid which is exactly the same footprint as one panel.
    opt_cell_size = max(8, cell_size // 2)
    img_A = draw_piece_standalone(cells_A, shape_A["color"], cell_size)
    img_B = draw_piece_standalone(cells_B, shape_A["color"], cell_size)
    img_C = draw_piece_on_grid(cells_C, shape_C["color"], off_C, grid_rows, grid_cols, cell_size)

    # Assign a unique random color to each option so the model cannot use color
    # as a shortcut and must rely purely on shape geometry.
    option_colors = rng.sample(_OPTION_COLOR_PALETTE, 4)
    opt_imgs = [
        draw_piece_on_grid(cells, option_colors[i], off, grid_rows, grid_cols, opt_cell_size)
        for i, (cells, off, _col) in enumerate(shuffled)
    ]

    composite_img, bboxes = build_analogy_composite(img_A, img_B, img_C, opt_imgs)

    # Mental-rotation strip following the circular rotation path.
    # 270° is shown as a single −90° (CCW) step instead of three CW steps.
    _rot_step = (gt_rot_C - ref_rot_C) % n_rots_C
    if _rot_step == 3:   # 270° CW  =  −90° CCW
        rotation_steps = [rots_C[ref_rot_C], rots_C[gt_rot_C]]
        _clockwise = False
    else:
        rotation_steps = [rots_C[(ref_rot_C + i) % n_rots_C]
                          for i in range(_rot_step + 1)]
        _clockwise = True
    intermediate_img = render_rotation_strip(
        rotation_steps, shape_C["color"],
        clockwise=_clockwise,
        angle=angle if _clockwise else 90,
        cell_size=cell_size,
        bw=bw_intermediate,
    )

    return {
        "composite_img":         composite_img,
        "intermediate_img":      intermediate_img,
        "bboxes":                bboxes,
        "answer":                answer,
        "transform_description": transform_desc,
        "transform_type":        effective_type,
        "shape_A_name":          shape_A["name"],
        "shape_C_name":          shape_C["name"],
        "shape_A_family":        shape_A["family"],
        "shape_C_family":        shape_C["family"],
        "option_transforms":     option_transforms,
    }


# ---------------------------------------------------------------------------
# Batch convenience
# ---------------------------------------------------------------------------

def generate_analogy_batch(
    n: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    cell_size: int = 32,
    seed: Optional[int] = None,
    transform_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if transform_types is None:
        transform_types = ["rotation", "translation", "combined"]
    # Exclude shapes with only 1 rotation from being shape_A: their A→B pair
    # looks identical (symmetric shape translated = same visual), making the
    # transformation hint trivial or invisible.
    eligible_A = [s for s in SHAPES if len(s["rotations"]) > 1]
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        shape_A = rng.choice(eligible_A)
        shape_C = rng.choice([s for s in SHAPES
                               if s["name"] != shape_A["name"]
                               and s["family"] != shape_A["family"]])
        ttype = rng.choice(transform_types)
        s = generate_analogy_sample(shape_A, shape_C, ttype,
                                    grid_rows, grid_cols, cell_size, rng)
        samples.append(s)
    return samples
