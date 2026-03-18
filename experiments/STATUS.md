# LantErn Project Status

_Last updated: 2026-03-18_

## Research Goal

Train a model (Qwen2.5-VL-3B) that learns to generate **meaningful latent visual tokens** during
reasoning — tokens that actually encode task-relevant visual information, not just structural
placeholders. The model should depend on these tokens to answer questions that require fine-grained
visual reasoning.

Two things must be true before the mechanism is considered to work end-to-end:

1. **Point 1 — Dependency:** The model routes signal through the latent channel when trained on
   the right data (i.e., the latent tokens carry information the model actually uses).
2. **Point 2 — Quality:** The model's self-generated latents encode spatially relevant visual
   information (i.e., own-latents > no-LVR on tasks requiring fine-grained reasoning).

**Current status:**
- Point 1: nearly confirmed (corrupted image ablation is strong evidence)
- Point 2: not yet confirmed (own-latents ≈ no-LVR on all benchmarks)

**Next milestone:** confirm both points before moving to GRPO/RL.

---

## Architecture Summary

LantErn extends Qwen2.5-VL with interleaved latent visual tokens (`<|lvr_start|>...<|lvr_end|>`).
During training, GT bbox crops are encoded and compressed via `apply_latent_compression()` into
`latent_size` embeddings that replace `<|lvr_sep|>` positions. The training loss is:

```
loss = ce_loss + γ * mse_loss
```

where MSE is computed between the model's predicted embeddings and the compressed GT visual features.
During generation, the model predicts its own latent embeddings token-by-token.

**The core tension:** MSE trains reconstruction fidelity in feature space, but there is no
direct gradient signal tying latent content to answer correctness. This is the root cause of
Point 2 being unconfirmed.

---

## Completed Ablations

### 1. GT Latent Signal — [`gt_latent_signal_ablation.md`](gt_latent_signal_ablation.md)
**Question:** Do GT latent tokens add signal over the base image context?

**Conclusion:** Yes (+0.8pp over no-LVR on VisCoT). But own-latents ≈ no-LVR,
meaning self-generated latents are uninformative. The mechanism works when given
the right content; the model just can't generate it.

---

### 2. Bbox Location — [`bbox_location_ablation.md`](bbox_location_ablation.md)
**Question:** Does the spatial location of the crop matter, or just its presence?

**Conclusion:** Location matters on VisCoT (+1.4pp GT vs random, strongest directional
flip margin in any pair: +17 net). Wrong-location crops are **worse than no latents**
on VisCoT. On Blink/VStar, location is irrelevant — structural presence alone drives gains.

---

### 3. Blink/VStar Baseline — [`blink_vstar_ablation.md`](blink_vstar_ablation.md)
**Question:** Does LVR help on held-out benchmarks?

**Conclusion:** LVR +4pp on Blink, +0.8pp on VStar vs no-LVR. However, random bbox ≈
own latents on both benchmarks — the gain is structural (model was trained with latent
slots, expects them at inference). **The LVR gain on Blink/VStar may be partly illusory.**

---

### 4. Systematic Ablation — [`ablation_eval.md`](ablation_eval.md)
**Question:** Full comparison of all conditions (GT, zeros, no-LVR, own, random) in one run.

**Conclusion:** GT > zeros > no-LVR ≈ own > random on VisCoT. The ranking confirms
spatial content matters, but the tiny no-LVR ≈ own gap is the key failure signal.
The corrupted image ablation was designed to explain this.

---

### 5. Corrupted Image — [`corrupted_image_ablation.md`](corrupted_image_ablation.md)
**Question:** Is the failure architectural (can't route through latents) or a training signal problem?

**Conclusion (decisive):** Architectural mechanism works — Corrupted+GT = **+14pp** over
Corrupted+no-LVR. The model can recover bbox-region information entirely through the latent
channel when forced to (main image blacked out). Own latents still ≈ no-LVR even with
corruption training, confirming **the MSE objective is the bottleneck**, not the architecture.
Corruption training also caused the model to structurally depend on latent slots
(no-LVR dropped ~17pp on Blink, ~15pp on VStar vs baseline).

---

## In Progress

### λ=0.8 Ablation — [`higher_lambda_ablation.md`](higher_lambda_ablation.md)
**Question:** Does increasing the MSE weight (γ=0.1 → 0.8) produce better latent quality,
or does it divert capacity from CE?

**Status:** training in progress

**Hypothesis:** More MSE pressure will improve reconstruction fidelity but won't fix
the core problem (MSE ≠ task relevance). Expected to see either (a) CE degrades as
capacity shifts to reconstruction, or (b) no change. Either outcome is informative.
A surprise positive (own-latents improves) would suggest the weight was the bottleneck.

---

## Planned Next

### Filtered Training Data — [`filtered_data_ablation.md`](filtered_data_ablation.md)
**Question:** If we train only on samples where the bbox crop is strictly necessary
to answer correctly, does the model learn to generate useful latents?

**Why this is the right fix:** When correct answers require the latent channel, the CE loss
itself becomes the training signal for latent quality. The MSE proxy is no longer needed
as the primary teacher — task performance and latent content become directly coupled.

**Filter criterion:** Keep samples where Qwen2.5-VL-32B (no LVR) is wrong, but correct
when given the bbox crop. Script: `synthetic/viscot/filter_easy_samples.py`.

**⚠️ Script needs adaptation:** currently filters the test set (JSONL format).
Must be adapted to filter training data (`LantErn_VisCot_data.json`):
- JSON array loading instead of JSONL
- `bboxs[0]` not `bbox`
- `reasoning_traces["answer"]` not `answer`
- Options are embedded in question text (no separate `options` field)

**Status:** filter script exists, adaptation needed before running.

**Key metric:** own-latents vs no-LVR gap on VisCoT. Currently ~0. Any positive gap
confirms Point 2 is achievable with the right training data.

---

## Decision Gate Before RL (GRPO)

Move to GRPO when at least one of:
- Filtered-data model shows **own-latents > no-LVR on VisCoT** (Point 2 confirmed)
- λ ablation provides a clear explanation and suggests a targeted RL reward strategy

The RL reward should be designed to signal whether self-generated latents contribute
to answer correctness — not just whether latent tokens are syntactically present.

---

## Baseline Numbers (Reference)

Model: `sft_mse_lt_8_lambda_0.1/checkpoint-1062`

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| GT latents | **0.793** | — | — |
| Zeros | 0.786 | — | — |
| No LVR | 0.785 | 0.582 | 0.629 |
| Own latents | 0.784 | 0.622 | 0.637 |
| Random bbox | 0.779 | 0.623 | 0.638 |
| Corrupted + GT latents | 0.747 | — | — |
| Corrupted + random bbox | 0.677 | 0.564 | 0.572 |
| Corrupted + own latents | 0.617 | 0.560 | 0.548 |
| Corrupted + no LVR | 0.607 | 0.385 | 0.397 |
