"""
PIL-based renderer for polyomino pieces on a grid canvas.
Builds individual piece images and composite (reference + options) images.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Single-panel renderer
# ---------------------------------------------------------------------------

def draw_piece_on_grid(
    cells: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    offset: Tuple[int, int] = (0, 0),
    grid_rows: int = 8,
    grid_cols: int = 8,
    cell_size: int = 40,
    bg_color: Tuple[int, int, int] = (28, 28, 32),
    grid_line_color: Tuple[int, int, int] = (55, 55, 65),
    border_color: Tuple[int, int, int] = (80, 80, 95),
    show_grid: bool = True,
) -> Image.Image:
    """
    Render a polyomino piece on a dark grid canvas.

    Args:
        cells:    List of (row, col) cell coordinates (0-indexed, normalized).
        color:    RGB fill color for the piece cells.
        offset:   (row_offset, col_offset) to shift the piece on the grid.
        grid_rows, grid_cols: Grid dimensions.
        cell_size: Pixel size of each grid cell (square).
        bg_color: Background color.
        grid_line_color: Interior grid line color.
        border_color: Outer border color.

    Returns:
        PIL Image of size (grid_cols*cell_size, grid_rows*cell_size).
    """
    w = grid_cols * cell_size
    h = grid_rows * cell_size
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw grid lines (optional)
    if show_grid:
        for r in range(grid_rows + 1):
            y = r * cell_size
            draw.line([(0, y), (w, y)], fill=grid_line_color, width=1)
        for c in range(grid_cols + 1):
            x = c * cell_size
            draw.line([(x, 0), (x, h)], fill=grid_line_color, width=1)

    # Draw piece cells
    dr, dc = offset
    highlight = tuple(min(255, v + 60) for v in color)   # lighter top-left edge
    shadow    = tuple(max(0,   v - 60) for v in color)   # darker bottom-right edge

    for r, c in cells:
        pr = r + dr
        pc = c + dc
        x0 = pc * cell_size + 1
        y0 = pr * cell_size + 1
        x1 = x0 + cell_size - 2
        y1 = y0 + cell_size - 2

        # Fill
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # 3-D bevel: highlight top/left, shadow bottom/right
        bevel = max(2, cell_size // 10)
        draw.rectangle([x0, y0, x1, y0 + bevel], fill=highlight)
        draw.rectangle([x0, y0, x0 + bevel, y1], fill=highlight)
        draw.rectangle([x0, y1 - bevel, x1, y1], fill=shadow)
        draw.rectangle([x1 - bevel, y0, x1, y1], fill=shadow)

    # Outer border
    draw.rectangle([0, 0, w - 1, h - 1], outline=border_color, width=2)

    return img


# ---------------------------------------------------------------------------
# Intermediate transformation image
# ---------------------------------------------------------------------------

def render_intermediate(
    cells_C: List[Tuple[int, int]],
    cells_gt: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    angle_text: str,
    cell_size: int = 40,
    grid_rows: int = 8,
    grid_cols: int = 8,
    bg_color: Tuple[int, int, int] = (18, 18, 22),
) -> Image.Image:
    """
    Render a three-panel image illustrating the transformation applied to C:

        [ query object (C) ]  →  "<angle_text>"  →  [ C after rotation ]

    Both panels use the same grid size and color. The arrow and label are
    centred between them.

    Args:
        cells_C:    Query object cells (normalized).
        cells_gt:   Query object after the ground-truth rotation (normalized).
        color:      RGB fill color for both panels.
        angle_text: Human-readable rotation label, e.g. "90° clockwise rotation".
        cell_size:  Pixels per grid cell.
        grid_rows, grid_cols: Grid dimensions (same for both panels).

    Returns:
        PIL Image.
    """
    panel_w = grid_cols * cell_size
    panel_h = grid_rows * cell_size

    font_arrow = _get_font(26)
    font_label = _get_font(18)

    text_color  = (210, 210, 220)
    arrow_color = (160, 160, 180)

    PAD       = 14
    ARROW_GAP = 50   # total width reserved for the arrow + label column

    total_w = panel_w + ARROW_GAP + panel_w + 2 * PAD
    total_h = panel_h + 2 * PAD

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw   = ImageDraw.Draw(canvas)

    # ── Left panel: query object C ─────────────────────────────────────────
    # Centre the piece in the grid
    from .pieces import valid_offsets as _valid_offsets
    offs_C = _valid_offsets(cells_C, grid_rows, grid_cols)
    off_C  = offs_C[len(offs_C) // 2] if offs_C else (0, 0)
    img_C  = draw_piece_on_grid(cells_C, color, off_C, grid_rows, grid_cols, cell_size)
    canvas.paste(img_C, (PAD, PAD))

    # ── Right panel: C rotated ────────────────────────────────────────────
    offs_gt = _valid_offsets(cells_gt, grid_rows, grid_cols)
    off_gt  = offs_gt[len(offs_gt) // 2] if offs_gt else (0, 0)
    img_gt  = draw_piece_on_grid(cells_gt, color, off_gt, grid_rows, grid_cols, cell_size)
    canvas.paste(img_gt, (PAD + panel_w + ARROW_GAP, PAD))

    # ── Arrow + label in the middle column ───────────────────────────────
    mid_x = PAD + panel_w + ARROW_GAP // 2
    mid_y = PAD + panel_h // 2

    # "→" arrow
    arrow = "→"
    aw = int(font_arrow.getlength(arrow)) if hasattr(font_arrow, "getlength") else 20
    ah = int(font_arrow.size)             if hasattr(font_arrow, "size")       else 26
    draw.text((mid_x - aw // 2, mid_y - ah - 4), arrow, fill=arrow_color, font=font_arrow)

    # rotation label (split into two lines if long)
    parts = angle_text.split(" rotation")[0].split(" clockwise")
    line1 = parts[0].strip() + "°" if "°" not in parts[0] else parts[0].strip()
    line2 = "clockwise" if len(parts) > 1 else ""

    lw1 = int(font_label.getlength(line1)) if hasattr(font_label, "getlength") else 60
    draw.text((mid_x - lw1 // 2, mid_y + 6), line1, fill=text_color, font=font_label)
    if line2:
        lw2 = int(font_label.getlength(line2)) if hasattr(font_label, "getlength") else 60
        draw.text((mid_x - lw2 // 2, mid_y + 6 + 22), line2, fill=text_color, font=font_label)

    return canvas


# ---------------------------------------------------------------------------
# Standalone piece renderer (no grid, no border — tight crop centred)
# ---------------------------------------------------------------------------

def draw_piece_standalone(
    cells: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    cell_size: int = 40,
    canvas_cells: int = 5,
    bg_color: Tuple[int, int, int] = (18, 18, 22),
) -> Image.Image:
    """
    Render a piece centred in a fixed square canvas with no grid or border.

    Args:
        cells:        Normalized (row, col) coordinates.
        color:        RGB fill color.
        cell_size:    Pixels per cell.
        canvas_cells: Canvas side length in cells (piece is centred within).
        bg_color:     Background color.
    """
    size = canvas_cells * cell_size
    img  = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    span_r = max(rows) - min(rows) + 1
    span_c = max(cols) - min(cols) + 1

    start_r = (canvas_cells - span_r) // 2 - min(rows)
    start_c = (canvas_cells - span_c) // 2 - min(cols)

    highlight = tuple(min(255, v + 60) for v in color)
    shadow    = tuple(max(0,   v - 60) for v in color)

    for r, c in cells:
        pr = r + start_r
        pc = c + start_c
        x0 = pc * cell_size + 1
        y0 = pr * cell_size + 1
        x1 = x0 + cell_size - 2
        y1 = y0 + cell_size - 2
        bevel = max(2, cell_size // 10)
        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.rectangle([x0, y0, x1, y0 + bevel], fill=highlight)
        draw.rectangle([x0, y0, x0 + bevel, y1], fill=highlight)
        draw.rectangle([x0, y1 - bevel, x1, y1], fill=shadow)
        draw.rectangle([x1 - bevel, y0, x1, y1], fill=shadow)

    return img


# ---------------------------------------------------------------------------
# Composite image builder
# ---------------------------------------------------------------------------

_LABEL_HEIGHT = 28   # pixels for the "(A)" header above each option panel
_SECTION_LABEL_HEIGHT = 22  # pixels for "Reference" / "Options" section header
_PAD = 8             # outer / inter-panel padding


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Try to load a truetype font; fall back to default."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def build_composite_image(
    reference_img: Image.Image,
    options: List[Image.Image],
    labels: List[str] = None,
) -> Tuple[Image.Image, List[List[int]]]:
    """
    Build a composite image with the reference on top (large) and the four
    options in a single row below it:

        +------------------------------------------+
        |              Reference                   |
        |              [big piece]                 |
        +------------------------------------------+
        |  (A)    |   (B)    |   (C)   |   (D)    |
        | [piece] |  [piece] | [piece] |  [piece] |
        +------------------------------------------+

    Args:
        reference_img: Rendered reference piece image (typically larger).
        options:       Exactly 4 option images [A, B, C, D].
        labels:        Option labels, default ['A','B','C','D'].

    Returns:
        (composite_image, bboxes) where bboxes is a list of four
        [x1, y1, x2, y2] pixel rectangles (one per option, in label order).
    """
    assert len(options) == 4, "Expected exactly 4 option images."
    if labels is None:
        labels = ["A", "B", "C", "D"]

    ref_w, ref_h = reference_img.size
    opt_w, opt_h = options[0].size   # assume all options same size

    font_label    = _get_font(15)
    font_section  = _get_font(13)
    text_color    = (220, 220, 220)
    section_color = (150, 150, 170)
    sep_color     = (55, 55, 70)
    bg            = (18, 18, 22)

    # Options row: 4 panels side by side
    options_row_w = 4 * opt_w + 5 * _PAD   # _PAD on each side + between
    options_row_h = _LABEL_HEIGHT + opt_h + _PAD

    # Total canvas width: wide enough for both reference and options row
    total_w = max(ref_w + 2 * _PAD, options_row_w + 2 * _PAD)

    # Total canvas height: section label + ref + separator + options row + padding
    sep_h = 2
    total_h = (
        _PAD
        + _SECTION_LABEL_HEIGHT + _PAD
        + ref_h
        + _PAD + sep_h + _PAD
        + _SECTION_LABEL_HEIGHT + _PAD
        + options_row_h
        + _PAD
    )

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    # --- Top section: Reference ---
    cy = _PAD
    draw.text((_PAD, cy), "Reference", fill=section_color, font=font_section)
    cy += _SECTION_LABEL_HEIGHT + _PAD

    # Center reference horizontally
    ref_x = (total_w - ref_w) // 2
    canvas.paste(reference_img, (ref_x, cy))
    cy += ref_h + _PAD

    # Horizontal separator
    draw.rectangle([_PAD, cy, total_w - _PAD, cy + sep_h - 1], fill=sep_color)
    cy += sep_h + _PAD

    # --- Bottom section: Options ---
    draw.text((_PAD, cy), "Options", fill=section_color, font=font_section)
    cy += _SECTION_LABEL_HEIGHT + _PAD

    # Center the 4 options row horizontally
    row_start_x = (total_w - (4 * opt_w + 3 * _PAD)) // 2

    bboxes: List[List[int]] = []
    for i, (opt_img, label) in enumerate(zip(options, labels)):
        ox = row_start_x + i * (opt_w + _PAD)

        # Label above option
        label_text = f"({label})"
        lw = font_label.getlength(label_text) if hasattr(font_label, "getlength") else 20
        draw.text((ox + (opt_w - lw) // 2, cy + 4), label_text,
                  fill=text_color, font=font_label)

        # Option image
        img_y = cy + _LABEL_HEIGHT
        canvas.paste(opt_img, (ox, img_y))

        bboxes.append([ox, img_y, ox + opt_w - 1, img_y + opt_h - 1])

    return canvas, bboxes


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pieces import SHAPES
    import random

    random.seed(0)
    shape = SHAPES[2]  # T-tetromino
    cells = shape["rotations"][0]
    color = shape["color"]

    ref  = draw_piece_on_grid(cells, color, offset=(2, 2))
    opts = [draw_piece_on_grid(cells, color, offset=(r, c))
            for r, c in [(1,1),(3,1),(1,3),(3,3)]]

    img, bboxes = build_composite_image(ref, opts)
    img.save("/tmp/composite_test.png")
    print("Saved /tmp/composite_test.png")
    print("Bboxes:", bboxes)
