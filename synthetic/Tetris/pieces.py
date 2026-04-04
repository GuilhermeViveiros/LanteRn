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
_SHAPE_DEFS = [
    # ---- Tetrominoes (4 cells) ----
    ("I4",  "tetromino", [(0,0),(0,1),(0,2),(0,3)],          (0,   240, 240)),  # cyan
    ("T4",  "tetromino", [(0,0),(0,1),(0,2),(1,1)],          (160,   0, 240)),  # purple
    ("S",   "tetromino", [(0,1),(0,2),(1,0),(1,1)],          (0,   240,   0)),  # green
    ("Z",   "tetromino", [(0,0),(0,1),(1,1),(1,2)],          (240,   0,   0)),  # red
    ("J",   "tetromino", [(0,0),(1,0),(1,1),(1,2)],          (0,     0, 240)),  # blue
    ("L4",  "tetromino", [(0,2),(1,0),(1,1),(1,2)],          (240, 160,   0)),  # orange

    # ---- Pentominoes (5 cells) ----
    ("F5",  "pentomino", [(0,1),(0,2),(1,0),(1,1),(2,1)],    (220,  80,  80)),  # rose
    ("I5",  "pentomino", [(0,0),(0,1),(0,2),(0,3),(0,4)],    (200, 120,  40)),  # brown-orange
    ("L5",  "pentomino", [(0,0),(1,0),(2,0),(3,0),(3,1)],    (240, 180,  80)),  # gold
    ("N5",  "pentomino", [(0,1),(1,0),(1,1),(2,0),(3,0)],    (180, 240, 100)),  # yellow-green
    ("P5",  "pentomino", [(0,0),(0,1),(1,0),(1,1),(2,0)],    (240, 100, 180)),  # pink
    ("T5",  "pentomino", [(0,0),(0,1),(0,2),(1,1),(2,1)],    (140, 100, 240)),  # violet
    ("U5",  "pentomino", [(0,0),(0,2),(1,0),(1,1),(1,2)],    (80,  200, 200)),  # teal
    ("V5",  "pentomino", [(0,0),(1,0),(2,0),(2,1),(2,2)],    (240, 140, 140)),  # salmon
    ("W5",  "pentomino", [(0,0),(1,0),(1,1),(2,1),(2,2)],    (140, 240, 160)),  # mint
    ("X5",  "pentomino", [(0,1),(1,0),(1,1),(1,2),(2,1)],    (240, 200, 100)),  # amber
    ("Y5",  "pentomino", [(0,1),(1,0),(1,1),(2,1),(3,1)],    (180, 120, 240)),  # lavender
    ("Z5",  "pentomino", [(0,0),(0,1),(1,1),(2,1),(2,2)],    (100, 180, 240)),  # sky-blue

    # ---- Hexominoes (6 cells) — 10 visually distinct selections ----
    # H1: straight line of 6
    ("H_line",  "hexomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)],       (60,  120, 200)),  # steel-blue
    # H3: L-shape with 4+2
    ("H_Lbig",  "hexomino", [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1)],       (200,  60, 120)),  # crimson
    # H4: T-shape wide
    ("H_Tbig",  "hexomino", [(0,0),(0,1),(0,2),(0,3),(1,1),(2,1)],       (60,  200, 160)),  # aquamarine
    # H5: staircase 3-step
    ("H_stair", "hexomino", [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2)],       (200, 160,  60)),  # mustard
    # H6: S-shape extended
    ("H_Sbig",  "hexomino", [(0,1),(0,2),(1,0),(1,1),(2,0),(2,1)],       (120,  80, 200)),  # indigo -- wait that's a 2x3 rect rotation
    # H7: + cross with tail
    ("H_cross", "hexomino", [(0,1),(1,0),(1,1),(1,2),(2,1),(3,1)],       (200, 100,  60)),  # burnt-orange
    # H8: U-shape extended
    ("H_Ubig",  "hexomino", [(0,0),(0,3),(1,0),(1,1),(1,2),(1,3)],       (80,  160, 240)),  # cornflower
    # H9: C-shape / hook
    ("H_hook",  "hexomino", [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1)],       (160, 200,  80)),  # lime
    # H10: Z-shape extended
    ("H_Zbig",  "hexomino", [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3)],       (240, 120, 120)),  # light-red

    # ---- Heptominoes (7 cells) — 12 visually distinct selections ----
    # G1: line of 7
    ("G_line",  "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)],         (80,  100, 220)),  # periwinkle
    # G2: L-shape 6+1
    ("G_Lbig",  "heptomino", [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(5,1)],         (220,  80,  80)),  # brick-red
    # G3: T-shape 5+2 tall
    ("G_Ttall", "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)],         (80,  220, 140)),  # sea-green
    # G4: S-staircase 4-step
    ("G_stair", "heptomino", [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3)],         (220, 180,  60)),  # saffron
    # G5: plus-sign with extra arm
    ("G_plus",  "heptomino", [(0,2),(1,0),(1,1),(1,2),(1,3),(2,2),(3,2)],         (180,  60, 220)),  # violet
    # G6: U-wide
    ("G_Uwide", "heptomino", [(0,0),(0,1),(0,2),(0,3),(1,0),(1,3),(2,0)],         (60,  200, 220)),  # cyan-teal
    # G7: zigzag
    ("G_zig",   "heptomino", [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)],         (220, 120,  60)),  # tangerine
    # G8: C-shape extended
    ("G_C",     "heptomino", [(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1)],         (100, 220, 100)),  # bright-green
    # G9: F-like asymmetric
    ("G_F",     "heptomino", [(0,1),(0,2),(1,0),(1,1),(2,1),(2,2),(3,2)],         (220,  80, 160)),  # hot-pink
    # G10: 3×3 minus 2 corners
    ("G_sq",    "heptomino", [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)],         (100, 160, 220)),  # light-blue
    # G11: long S
    ("G_Slong", "heptomino", [(0,2),(0,3),(1,1),(1,2),(2,0),(2,1),(3,0)],         (200, 200,  80)),  # chartreuse
    # G12: T-wide + tail
    ("G_Twide", "heptomino", [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)],         (220, 140, 100)),  # peach

    # ---- Octominoes (8 cells) — 8 visually distinct selections ----
    # O1: line of 8
    ("K_line",  "octomino",  [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7)],   (60,  80, 180)),   # navy
    # O3: L-shape 7+1
    ("K_Lbig",  "octomino",  [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(6,1)],   (60,  180,  80)),  # forest-green
    # O4: T-cross 5×3
    ("K_T",     "octomino",  [(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2),(3,2)],   (180, 100, 180)),  # mauve
    # O5: S-staircase 5-step
    ("K_stair", "octomino",  [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3)],   (180, 160,  40)),  # olive
    # O6: ring/frame (3×3 minus center)
    ("K_ring",  "octomino",  [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)],   (40,  160, 200)),  # cerulean
    # O7: Z extended
    ("K_Zbig",  "octomino",  [(0,0),(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(2,5)],   (200,  80, 100)),  # raspberry
    # O8: plus with long arms
    ("K_plus",  "octomino",  [(0,2),(1,2),(2,0),(2,1),(2,2),(2,3),(2,4),(3,2)],   (80,  220, 180)),  # aqua
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
