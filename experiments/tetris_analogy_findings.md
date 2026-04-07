# Tetris Analogy Task — Findings & Design Decisions

## Task Overview

Visual analogy task: **"A is rotated to B — as C is rotated to ?"**

The model sees:
- A and B as standalone floating pieces (no grid)
- C on a grid with 4 option panels (a/b/c/d)
- Must select which option is C after the same rotation applied to A→B

---

## Shape Library

**46 shapes total**, all with exactly **4 unique rotations** (fully asymmetric).

| Family | Count |
|---|---|
| Tetromino (4 cells) | 3 |
| Pentomino (5 cells) | 9 |
| Hexomino (6 cells) | 10 |
| Heptomino (7 cells) | 15 |
| Octomino (8 cells) | 9 |

### Why only fully asymmetric shapes?

Shapes with < 4 unique rotations (e.g., S, Z tetrominoes with 2 rotations) create **ambiguous analogy pairs**: the rotation transform is not uniquely determined from A→B, making the correct answer for C undefined. All 14 original ambiguous shapes were replaced with new fully asymmetric ones.

Removed: I4, S, Z, I5, X5, Z5, H_line, H_stair, H_Zbig, G_line, K_line, K_ring, K_stair, K_Zbig

Added: H_F, H_G, H_Y, H_R, G_R, G_hook2, G_E, G_spiral, K_F, K_G, K_N, K_C, K_E, K_spiral

---

## Configuration Space

Each analogy sample is defined by:

| Parameter | Count | Reason |
|---|---|---|
| A/B shape | 46 | which shape demonstrates the transform |
| A starting rotation | 4 | which rotation of A is shown |
| Transform (+90°/+180°/+270°) | 3 | what rotation is applied |
| C shape (different family) | ~36–43 | query shape (family constraint) |
| C starting rotation | 4 | which rotation of C is the query |

**Total unique configurations: ~77,760** (exact count varies due to family constraint on C).

A 100k dataset gives ~1.3× coverage with replacement — sufficient diversity without excessive repetition.

### Answer balance

Correct answer position is enforced via `force_correct_pos = n_success % 4`, guaranteeing exactly **25% per option (a/b/c/d)**.

---

## Dataset Split

Train/eval split is **baked into the dataset at generation time** (`create_dataset.py`):
- `train.json` — 90% of samples (shuffled with seed=42)
- `eval.json` — 10% of samples (same seed, no overlap)

Same objects appear in both splits — the split is over **configurations**, not shapes. This cleanly captures overfitting without the confound of unseen shapes.

---

## Latent Utility Results — Step 800

Training: Qwen2.5-VL-3B, SFT on 10k Tetris analogy samples, eval on 50 samples.

| Condition | Accuracy | Interpretation |
|---|---|---|
| GT latent tokens | **92.5%** | Correct visual embeddings → near-perfect performance |
| Own generated tokens | 30.0% | Model's own latents not yet useful |
| Random tokens | 25.0% | Chance level (4 options) |
| Zero tokens | 30.0% | Slightly above chance |

### Key observations

1. **GT vs random gap (92.5% vs 25%)**: The latent tokens carry strong visual information. The model has learned to consume them correctly.

2. **Own ≈ random ≈ chance**: The model has not yet learned to *generate* useful latent tokens — it can use them but can't produce them. This is expected SFT phase 1 behavior.

3. **Not overfitting**: The 92.5% GT accuracy on the eval set reflects that the visual reasoning signal is strong. The gap between GT and own is the target for GRPO phase 2.

4. **lvr_rate = 1.0 across all conditions**: The model always emits `<|lvr_start|>` tokens, confirming it has learned the LVR format.

### Expected training trajectory

- **SFT**: GT accuracy rises → model learns to use latent tokens
- **GRPO**: Own accuracy rises → model learns to generate good latent tokens
- Success criterion: own accuracy approaches GT accuracy

---

## Open Questions

- At what step does "own" accuracy start rising above chance?
- Does GRPO successfully close the GT/own gap?
- Does performance on the Tetris analogy task correlate with gains on VisCoT/BLINK/VStar?
