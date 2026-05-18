"""
Simplified composite image builder: shows only C + 4 option panels.

A and B are omitted because the rotation is stated explicitly in the text
prompt — there is no need to show the reference pair visually.

Layout:
    "Apply the rotation to (C):"
         [ C on grid, centred ]
    (a)      (b)      (c)      (d)
    [ a ]    [ b ]    [ c ]    [ d ]
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from .renderer import _get_font


def build_query_composite(
    img_C: Image.Image,
    options: list[Image.Image],
    labels: list[str] = None,
) -> tuple[Image.Image, list[list[int]]]:
    """
    Build composite showing only the query shape C and the 4 answer options.

    Args:
        img_C:   C rendered on its grid (draw_piece_on_grid).
        options: 4 option images in label order [a, b, c, d].
        labels:  Option labels (default ["a","b","c","d"]).

    Returns:
        (composite_image, bboxes) where bboxes[i] = [x1, y1, x2, y2]
        for option panel i in label order.
    """
    assert len(options) == 4
    if labels is None:
        labels = ["a", "b", "c", "d"]

    c_w, c_h = img_C.size
    op_w, op_h = options[0].size

    font_lbl = _get_font(26)
    font_text = _get_font(32)

    bg = (18, 18, 22)
    text_color = (210, 210, 220)

    GAP = 20  # gap between option panels
    PAD = 16  # outer padding
    LBL = 42  # label row height above each option panel

    th = int(font_text.size) if hasattr(font_text, "size") else 20

    header = "Apply the rotation to (C):"
    header_w = int(font_text.getlength(header)) if hasattr(font_text, "getlength") else 220
    opts_row_w = 4 * op_w + 3 * GAP
    inner_w = max(header_w, c_w, opts_row_w)
    total_w = inner_w + 2 * PAD
    total_h = PAD + th + PAD + c_h + PAD + LBL + op_h + PAD

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    cy = PAD

    # ── Header ────────────────────────────────────────────────────────────────
    hx = PAD + (inner_w - header_w) // 2
    draw.text((hx, cy), header, fill=text_color, font=font_text)
    cy += th + PAD

    # ── C on grid, centred ────────────────────────────────────────────────────
    canvas.paste(img_C, (PAD + (inner_w - c_w) // 2, cy))
    cy += c_h + PAD

    # ── Options row ───────────────────────────────────────────────────────────
    opts_x = PAD + (inner_w - opts_row_w) // 2
    bboxes: list[list[int]] = []
    for i, (opt_img, lbl) in enumerate(zip(options, labels)):
        ox = opts_x + i * (op_w + GAP)
        lt = f"({lbl})"
        lw = int(font_lbl.getlength(lt)) if hasattr(font_lbl, "getlength") else 20
        draw.text((ox + (op_w - lw) // 2, cy + 4), lt, fill=text_color, font=font_lbl)
        oy = cy + LBL
        canvas.paste(opt_img, (ox, oy))
        bboxes.append([ox, oy, ox + op_w - 1, oy + op_h - 1])

    return canvas, bboxes
