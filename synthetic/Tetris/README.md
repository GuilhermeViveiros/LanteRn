# Tetris Synthetic Dataset

Two multiple-choice visual reasoning tasks built on a polyomino shape library. Both are designed as LVR (Latent Visual Reasoning) training data for LantErn: the model inspects each option via latent tokens and selects the correct one.

---

## Problem types

### 1. Tetris transformation (`--task tetris`)
A reference shape is shown. Pick which of 4 options shows a valid rotation/translation/combined transformation of it.

```
Reference (large)
┌──────────────┐
│   [shape]    │
└──────────────┘
Options (A) (B) (C) (D) — one row, smaller panels
```

### 2. Visual analogy (`--task analogy`)
"Image A is to image B as image C is to which option?"  
The same transformation that maps A→B must be applied to C.

```
(A) → (B)          top row: transformation example
──────────────
(C) → (a)(b)       bottom row: C + 2×2 options grid
      (c)(d)
```

---

## Shape library (`pieces.py`)

**49 shapes** across 5 families:

| Family | Count | Cells |
|---|---|---|
| tetromino | 6 | 4 (O/square removed) |
| pentomino | 12 | 5 |
| hexomino | 8 | 6 (H_rect removed) |
| heptomino | 12 | 7 |
| octomino | 7 | 8 (K_rect removed) |

**Removed shapes:** O-tetromino (square), H_rect (2×3 block), K_rect (2×4 block) — solid rectangular blocks are excluded because they are visually ambiguous after rotation.

Each shape stores: `name`, `family`, `rotations` (unique, pre-computed), `color` (RGB), `n_cells`.

---

## Design decisions

### Shape selection
- **shape_A** (analogy): must have `len(rotations) > 1` — symmetric shapes excluded since their A→B pair looks identical after rotation
- **shape_C** (analogy): must be from a **different family** than shape_A — ensures A and C look visually distinct
- **No filled rectangles**: `_is_filled_rectangle()` filter in `analogy_simulator.py` rejects any distractor that forms a solid m×n block (m,n ≥ 2)

### Answer distribution
`force_correct_pos = sample_id % 4` cycles a→b→c→d deterministically → exactly 25% per answer across any complete multiple-of-4 dataset. Distractors are still randomly shuffled among the other 3 slots.

### Distractor strategies (analogy task, in priority order)
1. **Other rotations of shape_C** — hardest to dismiss, same color and shape
2. **Structurally perturbed shape_C** — add or remove one cell (near-miss)
3. **Same-family different shape** — similar visual style
4. **Random shape** — fallback

Same-rotation-different-position distractors are explicitly **banned**: at small panel sizes they are visually indistinguishable from the correct answer, creating multiple-correct samples.

### Reasoning traces (stage 1 — empty placeholders)
```python
"reasoning_traces": {
    "pre_visual_text_think": "",           # text before any bbox inspection
    "post_visual_text_think": ["","","",""],  # one entry per bbox (a/b/c/d)
    "text_think": None,                    # final consolidation text
    "answer": "a",
}
```
Stage 2 will fill these via a VLM pass. Intended stage-2 structure (not yet written):
```python
"steps": [
    {"type": "text",    "content": "...pre-inspection reasoning..."},
    {"type": "inspect", "option": "a", "content": "...reasoning about option a..."},
    {"type": "inspect", "option": "b", "content": "..."},
    {"type": "inspect", "option": "c", "content": "..."},
    {"type": "inspect", "option": "d", "content": "..."},
    {"type": "text",    "content": "...final conclusion..."},
]
```

---

## Usage

### Generate datasets
```bash
# Tetris transformation task
python -m synthetic.Tetris.create_dataset \
    --task tetris \
    --output_dir /path/to/tetris_data \
    --n_samples 50000 \
    --seed 42

# Visual analogy task
python -m synthetic.Tetris.create_dataset \
    --task analogy \
    --output_dir /path/to/analogy_data \
    --n_samples 50000 \
    --seed 42
```

**Output:**
```
{output_dir}/
├── images/{sample_id:06d}.png
└── tetris_data.json   (or analogy_data.json)
```

**All args:** `--task`, `--output_dir`, `--n_samples` (10k), `--grid_size` (8), `--cell_size` (auto: 40/32), `--seed` (42), `--save_every` (500), `--split` (train)

Checkpoint recovery is built-in: re-running the same command resumes from the existing JSON.

### Visualize (debug)
```bash
python -m synthetic.Tetris.visualize --task tetris  --n 9 --cols 3 --cell_size 52
python -m synthetic.Tetris.visualize --task analogy --n 9 --cols 3 --cell_size 52
```

**All args:** `--task`, `--n` (9), `--seed` (0), `--cell_size` (48), `--cols` (3), `--output` (/tmp/tetris_vis.png), `--show`

---

## File structure

```
synthetic/Tetris/
├── README.md               ← this file
├── __init__.py
├── pieces.py               # shape library: SHAPES, SHAPES_BY_FAMILY, get_unique_rotations()
├── renderer.py             # PIL renderer: draw_piece_on_grid(), build_composite_image()
├── simulator.py            # tetris task generator: generate_sample()
├── analogy_simulator.py    # analogy task generator: generate_analogy_sample()
├── create_dataset.py       # unified CLI entry point (--task tetris|analogy)
└── visualize.py            # unified visualizer (--task tetris|analogy)
```

---

## LantErn SFT format notes

- `bboxs`: 4 entries, one per option region in the composite image
- `len(bboxs)` must equal `len(post_visual_text_think)` — both are 4
- `src/datasets/sft_data.py` currently skips samples with `len(bboxs) != 1`; a new loader `src/datasets/sft_tetris_data.py` supporting 4-bbox samples is needed before training
