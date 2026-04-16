"""
Polyomino shape library for the Tetris synthetic dataset.

Includes all 7 tetrominoes, all 12 pentominoes, and 10 selected hexominoes.
Each shape has its unique rotations pre-computed (duplicates removed via canonical form).
"""

from typing import List, Tuple

Cells = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def _normalize(cells: Cells) -> tuple:
    """Translate so that min(row) == 0 and min(col) == 0, return sorted tuple."""
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_90cw(cells: Cells) -> Cells:
    """90° clockwise rotation: (r, c) -> (c, -r), then normalize."""
    return list(_normalize([(c, -r) for r, c in cells]))


def get_unique_rotations(base_cells: Cells) -> List[Cells]:
    """Return all unique rotations of a shape (1, 2, or 4)."""
    rotations = []
    seen: set = set()
    current = list(_normalize(base_cells))
    for _ in range(4):
        key = _normalize(current)
        if key not in seen:
            seen.add(key)
            rotations.append(list(key))
        current = _rotate_90cw(current)
    return rotations


# ---------------------------------------------------------------------------
# Shape definitions  (base orientation only; rotations computed automatically)
# ---------------------------------------------------------------------------

# Each entry: (name, family, base_cells, color_rgb)
# Mini shape set for ablation experiments (~2k configs).
# To restore the full dataset, uncomment all shapes below.
_SHAPE_DEFS = [
    # ---- Tetrominoes (4 cells) — only fully asymmetric (4 unique rotations) ----
    ("T4",  "tetromino", [(0,0),(0,1),(0,2),(1,1)],          (160,   0, 240)),  # purple
    ("J",   "tetromino", [(0,0),(1,0),(1,1),(1,2)],          (0,     0, 240)),  # blue
    ("L4",  "tetromino", [(0,2),(1,0),(1,1),(1,2)],          (240, 160,   0)),  # orange

    # ---- Pentominoes (5 cells) — only fully asymmetric (4 unique rotations) ----
    ("F5",  "pentomino", [(0,1),(0,2),(1,0),(1,1),(2,1)],    (220,  80,  80)),  # rose
    ("L5",  "pentomino", [(0,0),(1,0),(2,0),(3,0),(3,1)],    (240, 180,  80)),  # gold
    ("N5",  "pentomino", [(0,1),(1,0),(1,1),(2,0),(3,0)],    (180, 240, 100)),  # yellow-green
    ("P5",  "pentomino", [(0,0),(0,1),(1,0),(1,1),(2,0)],    (240, 100, 180)),  # pink
    ("T5",  "pentomino", [(0,0),(0,1),(0,2),(1,1),(2,1)],    (140, 100, 240)),  # violet
    # ("U5",  "pentomino", [(0,0),(0,2),(1,0),(1,1),(1,2)],    (80,  200, 200)),  # teal
    # ("V5",  "pentomino", [(0,0),(1,0),(2,0),(2,1),(2,2)],    (240, 140, 140)),  # salmon
    # ("W5",  "pentomino", [(0,0),(1,0),(1,1),(2,1),(2,2)],    (140, 240, 160)),  # mint
    # ("Y5",  "pentomino", [(0,1),(1,0),(1,1),(2,1),(3,1)],    (180, 120, 240)),  # lavender

    # ---- Hexominoes (6 cells) — only fully asymmetric ----
    ("H_Lbig",  "hexomino", [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1)],       (200,  60, 120)),  # crimson
    ("H_Tbig",  "hexomino", [(0,0),(0,1),(0,2),(0,3),(1,1),(2,1)],       (60,  200, 160)),  # aquamarine
    # ("H_Sbig",  "hexomino", [(0,1),(0,2),(1,0),(1,1),(2,0),(2,1)],       (120,  80, 200)),  # indigo
    # ("H_cross", "hexomino", [(0,1),(1,0),(1,1),(1,2),(2,1),(3,1)],       (200, 100,  60)),  # burnt-orange
    # ("H_Ubig",  "hexomino", [(0,0),(0,3),(1,0),(1,1),(1,2),(1,3)],       (80,  160, 240)),  # cornflower
    # Held-out hexominoes — kept out of SHAPES (see HELD_OUT_SHAPES below)
    # ("H_hook",  "hexomino", [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1)],       (160, 200,  80)),
    # ("H_F",     "hexomino", [(0,1),(0,2),(1,0),(1,1),(2,1),(3,1)],       (240,  80, 200)),
    # ("H_G",     "hexomino", [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1)],       (100, 240, 120)),
    # ("H_Y",     "hexomino", [(0,1),(1,0),(1,1),(2,1),(3,1),(4,1)],       (240, 200,  60)),
    # ("H_R",     "hexomino", [(0,0),(0,1),(1,0),(1,1),(2,1),(3,1)],       (80,  120, 240)),

    # ---- Heptominoes (7 cells) — commented out ----
    # ("G_Lbig",  "heptomino", [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(5,1)],         (220,  80,  80)),
    # ("G_Ttall", "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)],         (80,  220, 140)),
    # ("G_stair", "heptomino", [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3)],         (220, 180,  60)),
    # ("G_plus",  "heptomino", [(0,2),(1,0),(1,1),(1,2),(1,3),(2,2),(3,2)],         (180,  60, 220)),
    # ("G_Uwide", "heptomino", [(0,0),(0,1),(0,2),(0,3),(1,0),(1,3),(2,0)],         (60,  200, 220)),
    # ("G_zig",   "heptomino", [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)],         (220, 120,  60)),
    # ("G_C",     "heptomino", [(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1)],         (100, 220, 100)),
    # ("G_F",     "heptomino", [(0,1),(0,2),(1,0),(1,1),(2,1),(2,2),(3,2)],         (220,  80, 160)),
    # ("G_sq",    "heptomino", [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)],         (100, 160, 220)),
    # ("G_Slong", "heptomino", [(0,2),(0,3),(1,1),(1,2),(2,0),(2,1),(3,0)],         (200, 200,  80)),
    # ("G_Twide", "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)],         (220, 140, 100)),
    # ("G_R",     "heptomino", [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,1)],         (180, 240, 180)),
    # ("G_hook2", "heptomino", [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(2,2)],         (240, 160, 200)),
    # ("G_E",     "heptomino", [(0,0),(0,1),(0,2),(1,0),(1,1),(2,0),(2,1)],         (160, 100, 200)),
    # ("G_spiral","heptomino", [(0,0),(0,1),(0,2),(1,2),(1,1),(2,1),(2,0)],         (100, 200, 160)),

    # ---- Octominoes (8 cells) — commented out ----
    # ("K_Lbig",  "octomino",  [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(6,1)],   (60,  180,  80)),
    # ("K_T",     "octomino",  [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2),(3,2)],   (180, 100, 180)),
    # ("K_plus",  "octomino",  [(0,2),(1,2),(2,0),(2,1),(2,2),(2,3),(2,4),(3,2)],   (80,  220, 180)),
    # ("K_F",     "octomino",  [(0,1),(0,2),(1,0),(1,1),(2,1),(3,1),(4,1),(4,2)],   (200,  80,  60)),
    # ("K_G",     "octomino",  [(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1),(2,2)],   (80,  140, 240)),
    # ("K_N",     "octomino",  [(0,1),(1,0),(1,1),(2,0),(3,0),(4,0),(5,0),(5,1)],   (240, 200, 120)),
    # ("K_C",     "octomino",  [(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(3,0),(3,1)],   (200, 120, 240)),
    # ("K_E",     "octomino",  [(0,0),(0,1),(0,2),(1,0),(1,1),(2,0),(3,0),(3,1)],   (120, 240, 200)),
    # ("K_spiral","octomino",  [(0,0),(0,1),(0,2),(1,2),(1,1),(2,1),(2,0),(3,0)],   (240, 140,  80)),
]


# ---------------------------------------------------------------------------
# Build the public shape library
# ---------------------------------------------------------------------------

SHAPES: List[dict] = []

for name, family, base_cells, color in _SHAPE_DEFS:
    rotations = get_unique_rotations(base_cells)
    SHAPES.append({
        "name": name,
        "family": family,
        "rotations": rotations,   # list of lists of (r,c) tuples
        "color": color,
        "n_cells": len(base_cells),
    })

# Lookup by name
SHAPE_BY_NAME: dict = {s["name"]: s for s in SHAPES}

# Group by family
SHAPES_BY_FAMILY: dict = {}
for s in SHAPES:
    SHAPES_BY_FAMILY.setdefault(s["family"], []).append(s)


# ---------------------------------------------------------------------------
# Held-out shapes — excluded from SHAPES/training; used for OOD eval only
# ---------------------------------------------------------------------------

_HELD_OUT_DEFS = [
    ("H_hook", "hexomino", [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1)], (160, 200,  80)),
    ("H_F",    "hexomino", [(0,1),(0,2),(1,0),(1,1),(2,1),(3,1)], (240,  80, 200)),
    ("H_G",    "hexomino", [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1)], (100, 240, 120)),
    ("H_Y",    "hexomino", [(0,1),(1,0),(1,1),(2,1),(3,1),(4,1)], (240, 200,  60)),
    ("H_R",    "hexomino", [(0,0),(0,1),(1,0),(1,1),(2,1),(3,1)], ( 80, 120, 240)),
]

HELD_OUT_SHAPES: List[dict] = []
for _name, _family, _base, _color in _HELD_OUT_DEFS:
    _rots = get_unique_rotations(_base)
    HELD_OUT_SHAPES.append({
        "name": _name, "family": _family, "rotations": _rots,
        "color": _color, "n_cells": len(_base),
    })

HELD_OUT_SHAPE_NAMES: set = {s["name"] for s in HELD_OUT_SHAPES}


# ---------------------------------------------------------------------------
# Harder C shapes — heptominoes (7 cells), fully asymmetric (4 unique rotations)
# Used as shape_C in the harder held-out OOD eval; never seen during training.
# ---------------------------------------------------------------------------

_HARDER_C_DEFS = [
    ("G_Lbig",  "heptomino", [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(5,1)],         (220,  80,  80)),
    ("G_Ttall", "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)],         ( 80, 220, 140)),
    ("G_stair", "heptomino", [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3)],         (220, 180,  60)),
    ("G_plus",  "heptomino", [(0,2),(1,0),(1,1),(1,2),(1,3),(2,2),(3,2)],         (180,  60, 220)),
    ("G_F",     "heptomino", [(0,1),(0,2),(1,0),(1,1),(2,1),(2,2),(3,2)],         (220,  80, 160)),
]

HARDER_C_SHAPES: List[dict] = []
for _name, _family, _base, _color in _HARDER_C_DEFS:
    _rots = get_unique_rotations(_base)
    # Only keep shapes with ≥2 unique rotations (needed as valid shape_C targets)
    HARDER_C_SHAPES.append({
        "name": _name, "family": _family, "rotations": _rots,
        "color": _color, "n_cells": len(_base),
    })

HARDER_C_SHAPE_NAMES: set = {s["name"] for s in HARDER_C_SHAPES}


def get_bounding_box(cells: Cells) -> Tuple[int, int, int, int]:
    """Return (min_r, min_c, max_r, max_c) for a cell list."""
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def fits_in_grid(cells: Cells, offset: Tuple[int, int], grid_rows: int, grid_cols: int) -> bool:
    """Check whether cells + offset stays within the grid."""
    dr, dc = offset
    for r, c in cells:
        if not (0 <= r + dr < grid_rows and 0 <= c + dc < grid_cols):
            return False
    return True


def valid_offsets(cells: Cells, grid_rows: int, grid_cols: int) -> List[Tuple[int, int]]:
    """Return all (dr, dc) offsets that keep the piece fully inside the grid."""
    min_r, min_c, max_r, max_c = get_bounding_box(cells)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    offsets = []
    for dr in range(grid_rows - h + 1):
        for dc in range(grid_cols - w + 1):
            offsets.append((dr, dc))
    return offsets
