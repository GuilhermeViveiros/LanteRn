"""
Sample generator for the Tetris/polyomino transformation dataset.

For each sample:
  - Picks a shape (must have >1 rotation) and a reference rotation + placement.
  - Applies a ground-truth transformation (rotation, translation, or combined).
  - Generates 3 distractors — all rendered in the same color as the reference shape.
  - Builds the composite image and returns the LantErn-ready sample dict.

Distractor priority:
  1. All other unique rotations of the same shape (90°/180°/270° mix)
  2. Structurally perturbed version of the correct answer (±1 cell)
  3. Same-family different shape
  4. Random shape (fallback)

Never uses same-rotation-different-position as a distractor: at small panel sizes
these are visually indistinguishable from the correct answer.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

from .pieces import (
    SHAPES, SHAPES_BY_FAMILY,
    fits_in_grid, valid_offsets,
    get_unique_rotations, _normalize,
)
from .renderer import draw_piece_on_grid, build_composite_image


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

TRANSFORM_DESCRIPTIONS = {
    "rotation":    "{angle}° clockwise rotation",
    "translation": "translation of {dr:+d} row(s) and {dc:+d} column(s)",
    "combined":    "{angle}° clockwise rotation and translation of {dr:+d} row(s) and {dc:+d} column(s)",
}

ROTATION_ANGLES = {1: 90, 2: 180, 3: 270}   # rotation index offset → degrees


def _apply_offset(cells, offset):
    dr, dc = offset
    return [(r + dr, c + dc) for r, c in cells]


def _pick_similar_shape(shape: dict, rng: random.Random) -> dict:
    """Pick a different shape from the same family, or any other shape as fallback."""
    same_family = [s for s in SHAPES_BY_FAMILY.get(shape["family"], []) if s["name"] != shape["name"]]
    if same_family:
        return rng.choice(same_family)
    return rng.choice([s for s in SHAPES if s["name"] != shape["name"]])


# ---------------------------------------------------------------------------
# Perturbation helpers (for near-miss distractors)
# ---------------------------------------------------------------------------

def _adjacent_cells(cells: list) -> list:
    occupied = set(cells)
    adj = set()
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb not in occupied:
                adj.add(nb)
    return list(adj)


def _is_connected(cells: list) -> bool:
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
    """True if cells fully fill an m×n bounding box (m,n ≥ 2) — i.e. a solid block."""
    norm = list(_normalize(cells))
    rows = [r for r, c in norm]
    cols = [c for r, c in norm]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return h >= 2 and w >= 2 and len(norm) == h * w


def _perturb_cells(cells: list, rng: random.Random, mode: str = "random") -> Optional[list]:
    """Return cells with one cell added or removed, or None if impossible."""
    if mode == "random":
        options = ["add"]
        if len(cells) > 2:
            options.append("remove")
        mode = rng.choice(options)

    if mode == "add":
        adj = _adjacent_cells(cells)
        if not adj:
            return None
        return list(cells) + [rng.choice(adj)]

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
# Core sample generation
# ---------------------------------------------------------------------------

def generate_sample(
    shape: dict,
    transform_type: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    ref_cell_size: int = 52,
    opt_cell_size: int = 28,
    rng: Optional[random.Random] = None,
    force_correct_pos: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate one multiple-choice sample.

    All 4 option panels are rendered in the reference shape's color.
    shape must have len(rotations) > 1 for rotation/combined tasks to be meaningful.
    """
    if rng is None:
        rng = random.Random()

    rotations = shape["rotations"]
    n_rots = len(rotations)
    color = shape["color"]

    # --- 1. Pick reference rotation and placement ---
    ref_rot_idx = rng.randrange(n_rots)
    ref_cells = rotations[ref_rot_idx]
    ref_offsets = valid_offsets(ref_cells, grid_rows, grid_cols)
    ref_offset = rng.choice(ref_offsets)

    # --- 2. Determine ground-truth transformation ---
    effective_type = transform_type
    if n_rots == 1 and effective_type in ("rotation", "combined"):
        effective_type = "translation"

    gt_rot_idx = ref_rot_idx
    gt_offset = ref_offset
    transform_desc = ""
    angle = 0
    delta = (0, 0)

    if effective_type == "rotation":
        rot_step = rng.choice(list(range(1, n_rots)))
        gt_rot_idx = (ref_rot_idx + rot_step) % n_rots
        gt_cells = rotations[gt_rot_idx]
        gt_offset = rng.choice(valid_offsets(gt_cells, grid_rows, grid_cols))
        angle = ROTATION_ANGLES.get(rot_step, rot_step * 90)
        transform_desc = TRANSFORM_DESCRIPTIONS["rotation"].format(angle=angle)

    elif effective_type == "translation":
        gt_cells = ref_cells
        candidates = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if (dr, dc) == (0, 0):
                    continue
                new_off = (ref_offset[0] + dr, ref_offset[1] + dc)
                if fits_in_grid(ref_cells, new_off, grid_rows, grid_cols):
                    candidates.append((dr, dc))
        if not candidates:
            alt = [o for o in valid_offsets(ref_cells, grid_rows, grid_cols) if o != ref_offset]
            if not alt:
                return generate_sample(shape, "rotation", grid_rows, grid_cols,
                                       ref_cell_size, opt_cell_size, rng, force_correct_pos)
            delta = (alt[0][0] - ref_offset[0], alt[0][1] - ref_offset[1])
        else:
            delta = rng.choice(candidates)
        gt_offset = (ref_offset[0] + delta[0], ref_offset[1] + delta[1])
        transform_desc = TRANSFORM_DESCRIPTIONS["translation"].format(dr=delta[0], dc=delta[1])

    elif effective_type == "combined":
        rot_step = rng.choice(list(range(1, n_rots)))
        gt_rot_idx = (ref_rot_idx + rot_step) % n_rots
        gt_cells = rotations[gt_rot_idx]
        angle = ROTATION_ANGLES.get(rot_step, rot_step * 90)
        cands = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                base_off = valid_offsets(gt_cells, grid_rows, grid_cols)
                if base_off:
                    new_off = (base_off[0][0] + dr, base_off[0][1] + dc)
                    if fits_in_grid(gt_cells, new_off, grid_rows, grid_cols):
                        cands.append((dr, dc))
        gt_offset = rng.choice(valid_offsets(gt_cells, grid_rows, grid_cols))
        if cands:
            base_off = valid_offsets(gt_cells, grid_rows, grid_cols)[0]
            delta = rng.choice(cands)
            new_off = (base_off[0] + delta[0], base_off[1] + delta[1])
            if fits_in_grid(gt_cells, new_off, grid_rows, grid_cols):
                gt_offset = new_off
            else:
                delta = (0, 0)
        transform_desc = TRANSFORM_DESCRIPTIONS["combined"].format(
            angle=angle, dr=gt_offset[0] - ref_offset[0], dc=gt_offset[1] - ref_offset[1]
        )
    else:
        raise ValueError(f"Unknown transform_type: {transform_type!r}")

    gt_cells = rotations[gt_rot_idx]

    # --- 3. Generate 3 distractors ---
    # All distractors are rendered in the same color as the reference shape.
    # We never use same-rotation-different-position: visually identical to correct answer.

    def _also_correct(cells_d, offset_d) -> bool:
        if effective_type == "rotation":
            # Position is irrelevant for rotation tasks — orientation is what matters.
            return _normalize(cells_d) == _normalize(gt_cells)
        else:
            return frozenset(_apply_offset(cells_d, offset_d)) == frozenset(_apply_offset(gt_cells, gt_offset))

    def _vkey(cells, offset):
        return frozenset(_apply_offset(cells, offset))

    seen = {_vkey(gt_cells, gt_offset)}
    distractors: List[Tuple] = []

    def _add(cells, offset) -> bool:
        if not fits_in_grid(cells, offset, grid_rows, grid_cols):
            return False
        if _also_correct(cells, offset):
            return False
        if _is_filled_rectangle(cells):
            return False
        vk = _vkey(cells, offset)
        if vk in seen:
            return False
        seen.add(vk)
        distractors.append((cells, offset))
        return True

    # Priority 1: all other unique rotations of the same shape (shuffled for variety)
    other_rots = [i for i in range(n_rots) if i != gt_rot_idx]
    rng.shuffle(other_rots)
    for ri in other_rots:
        if len(distractors) >= 3:
            break
        alt_cells = rotations[ri]
        alt_offs = valid_offsets(alt_cells, grid_rows, grid_cols)
        if alt_offs:
            _add(alt_cells, rng.choice(alt_offs))

    # Priority 2: structurally perturbed correct answer (±1 cell, same color)
    for mode in rng.sample(["add", "remove"], 2):
        if len(distractors) >= 3:
            break
        perturbed = _perturb_cells(list(gt_cells), rng, mode=mode)
        if perturbed is None:
            continue
        perturbed_norm = list(_normalize(perturbed))
        offs = valid_offsets(perturbed_norm, grid_rows, grid_cols)
        if offs:
            _add(perturbed_norm, rng.choice(offs))

    # Priority 3: same-family different shape
    attempts = 0
    while len(distractors) < 3 and attempts < 20:
        attempts += 1
        sim = _pick_similar_shape(shape, rng)
        sim_rots = list(range(len(sim["rotations"])))
        rng.shuffle(sim_rots)
        for ri in sim_rots:
            sim_cells = sim["rotations"][ri]
            sim_offs = valid_offsets(sim_cells, grid_rows, grid_cols)
            if sim_offs and _add(sim_cells, rng.choice(sim_offs)):
                break

    # Fallback: any random shape
    attempts = 0
    while len(distractors) < 3 and attempts < 50:
        attempts += 1
        rs = rng.choice(SHAPES)
        ri = rng.randrange(len(rs["rotations"]))
        rc = rs["rotations"][ri]
        ro = valid_offsets(rc, grid_rows, grid_cols)
        if ro:
            _add(rc, rng.choice(ro))

    distractors = distractors[:3]

    # --- 4. Shuffle options and record correct label ---
    # correct answer is index 0; distractors are 1-3
    all_opts = [(gt_cells, gt_offset)] + distractors

    if force_correct_pos is not None:
        pos = force_correct_pos % 4
        dist_indices = list(range(1, 4))
        rng.shuffle(dist_indices)
        ordered = [None] * 4
        ordered[pos] = all_opts[0]
        for slot, di in zip([i for i in range(4) if i != pos], dist_indices):
            ordered[slot] = all_opts[di]
        shuffled_options = ordered
        correct_shuffled_idx = pos
    else:
        indices = list(range(4))
        rng.shuffle(indices)
        correct_shuffled_idx = indices.index(0)
        shuffled_options = [all_opts[i] for i in indices]

    answer = "abcd"[correct_shuffled_idx]

    # --- 5. Render — all options in the reference shape's color ---
    ref_img = draw_piece_on_grid(
        ref_cells, color, ref_offset, grid_rows, grid_cols, ref_cell_size
    )
    opt_imgs = [
        draw_piece_on_grid(cells, color, off, grid_rows, grid_cols, opt_cell_size)
        for cells, off in shuffled_options
    ]

    composite_img, bboxes = build_composite_image(ref_img, opt_imgs)

    return {
        "composite_img": composite_img,
        "bboxes": bboxes,
        "answer": answer,
        "transform_description": transform_desc,
        "transform_type": effective_type,
        "shape_name": shape["name"],
        "shape_family": shape["family"],
    }


# ---------------------------------------------------------------------------
# Batch convenience
# ---------------------------------------------------------------------------

def generate_batch(
    n: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    ref_cell_size: int = 52,
    opt_cell_size: int = 28,
    seed: Optional[int] = None,
    transform_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate n samples with a uniform mix of transform types."""
    if transform_types is None:
        transform_types = ["rotation", "translation", "combined"]
    eligible = [s for s in SHAPES if len(s["rotations"]) > 1]
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        shape = rng.choice(eligible)
        ttype = rng.choice(transform_types)
        sample = generate_sample(shape, ttype, grid_rows, grid_cols,
                                 ref_cell_size, opt_cell_size, rng)
        samples.append(sample)
    return samples
