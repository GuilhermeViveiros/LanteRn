"""
PIL-based renderer for polyomino pieces on a grid canvas.
Builds individual piece images and composite (reference + options) images.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Single-panel renderer
# ---------------------------------------------------------------------------

def draw_piece_on_grid(
    cells: list[tuple[int, int]],
    color: tuple[int, int, int],
    offset: tuple[int, int] = (0, 0),
    grid_rows: int = 8,
    grid_cols: int = 8,
    cell_size: int = 40,
    bg_color: tuple[int, int, int] = (28, 28, 32),
    grid_line_color: tuple[int, int, int] = (55, 55, 65),
    border_color: tuple[int, int, int] = (80, 80, 95),
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
# Mental-rotation strip  (circular layout)
# ---------------------------------------------------------------------------

def render_rotation_strip(
    rotation_steps: list[list[tuple[int, int]]],
    color: tuple[int, int, int],
    clockwise: bool = True,
    angle: int = 90,
    cell_size: int = 40,
    bg_color: tuple[int, int, int] = (18, 18, 22),
    bw: bool = False,
) -> Image.Image:
    """
    Mental-rotation image. Three layouts:

        90°  CW  (2 steps): [C] →"90° right"→ [C@90°]
        180° CW  (3 steps): [C] →"90° right"→ [C@90°]
                                                    ↓"90° right"↓
                                               [C@180°]            (L-shape)
        270° CCW (2 steps): [C@left] ←"90° left"← [C]
    """
    piece_color = (0, 0, 0)       if bw else color
    strip_bg    = (255, 255, 255) if bw else bg_color
    imgs = [draw_piece_standalone(c, piece_color, cell_size, bg_color=strip_bg) for c in rotation_steps]
    if len(imgs) < 2:
        imgs = imgs * 2

    PAD         = 20
    GAP         = 70
    BEND        = 36
    arrow_color = (80, 80, 80)    if bw else (140, 170, 200)
    text_color  = (40, 40, 40)    if bw else (190, 200, 215)
    font        = _get_font(15)
    # Use max dimensions across all panels so no piece gets clipped
    pw = max(im.width  for im in imgs)
    ph = max(im.height for im in imgs)

    def _bez(p0, p1, p2, n=70):
        pts = []
        for i in range(n + 1):
            t = i / n
            mt = 1 - t
            pts.append((int(mt*mt*p0[0]+2*mt*t*p1[0]+t*t*p2[0]),
                        int(mt*mt*p0[1]+2*mt*t*p1[1]+t*t*p2[1])))
        return pts

    def _draw_arrow(draw_obj, pts, lbl, label_x, label_y):
        for j in range(len(pts) - 2):
            draw_obj.line([pts[j], pts[j+1]], fill=arrow_color, width=2)
        if len(pts) >= 2:
            dx, dy = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
            L = math.sqrt(dx*dx+dy*dy)
            if L > 1e-6:
                dx, dy = dx/L, dy/L
                px, py = -dy, dx
                s = 14
                bx, by = pts[-1][0]-dx*s, pts[-1][1]-dy*s
                q1 = (int(bx+px*s*.45), int(by+py*s*.45))
                q2 = (int(bx-px*s*.45), int(by-py*s*.45))
                draw_obj.polygon([pts[-1], q1, q2], fill=arrow_color)
        lw = int(font.getlength(lbl)) if hasattr(font, "getlength") else 50
        draw_obj.text((label_x - lw//2, label_y), lbl, fill=text_color, font=font)

    # ── single horizontal row for all cases ────────────────────────────────
    if True:  # always
        n_panels = len(imgs)
        total_w  = PAD + n_panels * pw + (n_panels - 1) * GAP + PAD
        total_h  = ph + 2*PAD + BEND

        canvas = Image.new("RGB", (total_w, total_h), strip_bg)
        draw   = ImageDraw.Draw(canvas)

        y0 = PAD + BEND
        cy = y0 + ph // 2

        def _paste(im, x):
            # vertically centre each panel within the tallest slot
            canvas.paste(im, (x, y0 + (ph - im.height) // 2))

        if clockwise:
            # panels left to right: imgs[0], imgs[1], imgs[2] (if 180°)
            for i, im in enumerate(imgs):
                _paste(im, PAD + i * (pw + GAP))
            for i in range(n_panels - 1):
                x_left  = PAD + i * (pw + GAP) + pw
                x_right = PAD + (i + 1) * (pw + GAP)
                p0 = (x_left,  cy)
                p2 = (x_right, cy)
                p1 = ((p0[0]+p2[0])//2, y0 - BEND//2)
                pts = _bez(p0, p1, p2)
                _draw_arrow(draw, pts, "90\u00b0 right", p1[0], PAD + 2)
        else:
            # CCW: result on left, source on right
            _paste(imgs[1], PAD)
            _paste(imgs[0], PAD + pw + GAP)
            p0 = (PAD + pw + GAP, cy)
            p2 = (PAD + pw,       cy)
            p1 = ((p0[0]+p2[0])//2, y0 - BEND//2)
            pts = _bez(p0, p1, p2)
            _draw_arrow(draw, pts, "90\u00b0 left", p1[0], PAD + 2)

    return canvas


# ---------------------------------------------------------------------------
# Standalone piece renderer (no grid, no border — tight crop centred)
# ---------------------------------------------------------------------------

def draw_piece_standalone(
    cells: list[tuple[int, int]],
    color: tuple[int, int, int],
    cell_size: int = 40,
    canvas_cells: int = None,
    bg_color: tuple[int, int, int] = (18, 18, 22),
) -> Image.Image:
    """
    Render a piece centred in a canvas with no grid or border.

    Args:
        cells:        Normalized (row, col) coordinates.
        color:        RGB fill color.
        cell_size:    Pixels per cell.
        canvas_cells: Canvas side length in cells. If None (default), the canvas
                      is sized dynamically to fit the piece with 2 cells of padding
                      on each side — prevents clipping for large/wide pieces.
        bg_color:     Background color.
    """
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    span_r = max(rows) - min(rows) + 1
    span_c = max(cols) - min(cols) + 1

    if canvas_cells is None:
        pad  = 2
        side = max(span_r, span_c) + 2 * pad   # square: same for all rotations of a piece
        canvas_w = side * cell_size
        canvas_h = side * cell_size
        start_r  = (side - span_r) // 2 - min(rows)
        start_c  = (side - span_c) // 2 - min(cols)
    else:
        canvas_w = canvas_cells * cell_size
        canvas_h = canvas_cells * cell_size
        start_r  = (canvas_cells - span_r) // 2 - min(rows)
        start_c  = (canvas_cells - span_c) // 2 - min(cols)

    img  = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)

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
        except OSError:
            pass
    return ImageFont.load_default()


def build_composite_image(
    reference_img: Image.Image,
    options: list[Image.Image],
    labels: list[str] = None,
) -> tuple[Image.Image, list[list[int]]]:
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

    bboxes: list[list[int]] = []
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
    import random

    from pieces import SHAPES

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
