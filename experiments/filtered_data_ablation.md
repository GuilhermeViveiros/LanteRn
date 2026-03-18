# Ablation: Filtered Training Data (Bbox-Decisive Samples Only)

## Motivation

All prior ablations show own-latents ≈ no-LVR — the latent prediction head doesn't generate
useful content. The corrupted image ablation confirms this is a **training signal problem**:
the MSE loss trains the model to minimize reconstruction error in feature space, with no
gradient signal tying latent content to answer correctness.

**The fix:** train on a subset where the bbox crop is strictly necessary to answer correctly.
When the model *must* route through the latent token to get the right answer, the CE loss
itself becomes the training signal for latent quality — no reconstruction proxy needed.

## Filter Criterion

A sample is kept only if Qwen2.5-VL-32B (baseline, no LVR):
1. Answers **wrong** with image + question alone
2. Answers **correct** with image + bbox_crop + question

Script: `synthetic/viscot/filter_easy_samples.py`

## ⚠️ Implementation Note

The filter script currently runs on `viscot_mc_test.jsonl` (test set format: JSONL, `bbox`,
`options` field). To filter the **training data** (`LantErn_VisCot_data.json`), it needs to be
adapted for the training format:
- Load as `json.load()` (JSON array, not JSONL)
- Use `bboxs[0]` (plural) not `bbox`
- Use `reasoning_traces["answer"]` not `answer`
- MC options are embedded in the question text, not a separate field

**Options:**
1. Adapt the filter script to run on training data directly
2. Run as-is on a MC-formatted version of the training data (if that exists)

Once `keep_ids.json` is generated from the training data, pass it via `--filter_ids_path`
to `train_sft.py` → `SFTDataset` already supports this parameter.

## Setup

- **Model:** new SFT checkpoint trained on filtered subset
- **Base config:** same as `sft_mse_lt_8_lambda_0.1` (γ=0.1)
- **Only change:** training data filtered to bbox-decisive samples
- **Checkpoint:** TBD

## Planned Runs

- [ ] **Adapt filter script** for training data format (or confirm MC training split exists)
- [ ] **Run filter** — identify keep_ids on training data
- [ ] **Train** — SFT on filtered subset
- [ ] **Eval (VisCoT)** — `--lvr --use_gt` vs `--lvr --no-use_gt` vs `--no-lvr`
- [ ] **Eval (Blink/VStar)** — `--lvr` vs `--no-lvr`

## Key Metric to Watch

**Own latents vs No LVR on VisCoT.** In the baseline model this is ~tied (0.784 vs 0.785).
A meaningful gap here (own latents > no-LVR) would confirm that:
1. The model depends on the latent channel (point 1 — confirmed by corrupted ablation)
2. The model generates *useful* latent content (point 2 — unconfirmed)

## Results

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| Own latents (filtered train) | | | |
| No LVR (filtered train) | | | |
| GT latents (filtered train) | | | |
| **Baseline** Own latents (full train) | 0.784 | 0.622 | 0.637 |
| **Baseline** No LVR (full train) | 0.785 | 0.582 | 0.629 |

## Conclusions

*(pending)*
