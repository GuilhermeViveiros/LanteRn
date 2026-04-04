"""
Unified visualization tool for both Tetris problem types.

Usage:
    python -m synthetic.Tetris.visualize --task tetris  --n 9 --cols 3
    python -m synthetic.Tetris.visualize --task analogy --n 9 --cols 3 --cell_size 52
"""

import argparse
import random

from PIL import Image, ImageDraw

from .pieces import SHAPES
from .simulator import generate_sample
from .analogy_simulator import generate_analogy_sample
from .renderer import _get_font


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

def make_grid(images, n_cols: int, padding: int = 6, bg=(12, 12, 16)) -> Image.Image:
    n_rows = (len(images) + n_cols - 1) // n_cols
    cell_w = max(im.width  for im in images)
    cell_h = max(im.height for im in images)
    canvas = Image.new("RGB",
                       (n_cols * cell_w + (n_cols + 1) * padding,
                        n_rows * cell_h + (n_rows + 1) * padding),
                       bg)
    for idx, im in enumerate(images):
        col = idx % n_cols
        row = idx // n_cols
        canvas.paste(im, (padding + col * (cell_w + padding),
                          padding + row * (cell_h + padding)))
    return canvas


def annotate(img: Image.Image, sample: dict) -> Image.Image:
    cap_h = 24
    font = _get_font(14)
    out = Image.new("RGB", (img.width, img.height + cap_h), (18, 18, 22))
    out.paste(img, (0, 0))
    answer = sample.get("answer", "?")
    transform = sample.get("transform_description", sample.get("transform_type", ""))
    text = f"Answer: ({answer})   |   {transform}"
    ImageDraw.Draw(out).text(
        (6, img.height + 4),
        text,
        fill=(160, 210, 160), font=font,
    )
    return out


# ---------------------------------------------------------------------------
# Per-task sample generators
# ---------------------------------------------------------------------------

_ELIGIBLE_A = [s for s in SHAPES if len(s["rotations"]) > 1]
_TETRIS_TRANSFORM_TYPES  = ["rotation", "combined"]   # no translation: same orientation = looks identical to reference
_ANALOGY_TRANSFORM_TYPES = ["rotation", "combined"]


def generate_samples_tetris(n: int, cell_size: int, seed: int) -> list:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        shape = rng.choice(_ELIGIBLE_A)
        ttype = rng.choice(_TETRIS_TRANSFORM_TYPES)
        s = generate_sample(
            shape=shape,
            transform_type=ttype,
            ref_cell_size=cell_size,
            opt_cell_size=max(16, cell_size // 2),
            rng=rng,
        )
        samples.append(s)
    return samples


def generate_samples_analogy(n: int, cell_size: int, seed: int) -> list:
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        shape_A = rng.choice(_ELIGIBLE_A)
        shape_C = rng.choice([
            s for s in SHAPES
            if s["name"] != shape_A["name"] and s["family"] != shape_A["family"]
        ])
        ttype = rng.choice(_ANALOGY_TRANSFORM_TYPES)
        s = generate_analogy_sample(
            shape_A=shape_A,
            shape_C=shape_C,
            transform_type=ttype,
            cell_size=cell_size,
            rng=rng,
            force_correct_pos=i % 4,
        )
        samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize Tetris synthetic dataset samples.")
    p.add_argument("--task",      type=str, required=True, choices=["tetris", "analogy"])
    p.add_argument("--n",         type=int, default=9)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--cell_size", type=int, default=48)
    p.add_argument("--cols",      type=int, default=3)
    p.add_argument("--output",    type=str, default="/tmp/tetris_vis.png")
    p.add_argument("--show",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Generating {args.n} {args.task} samples (seed={args.seed}, cell_size={args.cell_size})...")

    if args.task == "tetris":
        samples = generate_samples_tetris(args.n, args.cell_size, args.seed)
    else:
        samples = generate_samples_analogy(args.n, args.cell_size, args.seed)

    annotated = [annotate(s["composite_img"], s) for s in samples]
    grid = make_grid(annotated, n_cols=args.cols)
    grid.save(args.output)
    print(f"Saved → {args.output}")
    if args.show:
        grid.show()


if __name__ == "__main__":
    main()
