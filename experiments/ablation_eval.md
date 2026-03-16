# Systematic Latent Token Ablation (ablation_eval.py)

## Setup

- **Model:** `sft_mse_lt_8_lambda_0.1/checkpoint-1062`
- **Dataset:** viscot MC test, 300 steps × batch 4 ≈ 4800 samples
- **Script:** `evals/ablation_eval.py` — loads model once, runs all conditions sequentially

## Conditions

| Condition | Description |
|-----------|-------------|
| `gt_latents` | GT bbox crop → vision encoder → injected as latent |
| `zeros` | Zero embeddings injected (bypasses vision encoder) |
| `no_lvr` | Standard generation, no latent tokens |
| `own_latents` | LVR generation, model predicts its own latents freely |
| `random_bbox` | Random-location crop (same size) → vision encoder → injected |

## Results

### Accuracy

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| `gt_latents` | **0.793** | — (no GT bboxes) | — (no GT bboxes) |
| `zeros` | 0.786 | — | — |
| `no_lvr` | 0.785 | 0.582 | 0.629 |
| `own_latents` | 0.784 | 0.622 | 0.637 |
| `random_bbox` | 0.779 | 0.623 | 0.638 |

### Directional Flip Counts (A wins → A correct, B wrong)

| Pair | A wins | B wins | Net |
|------|-------:|-------:|----:|
| gt_latents vs no_lvr | 27 | 17 | **+10 GT** |
| gt_latents vs own_latents | 28 | 17 | **+11 GT** |
| gt_latents vs random_bbox | 33 | 16 | **+17 GT** |
| gt_latents vs zeros | 28 | 19 | **+9 GT** |
| zeros vs no_lvr | 15 | 14 | +1 zeros |
| zeros vs own_latents | 20 | 18 | +2 zeros |
| zeros vs random_bbox | 19 | 11 | **+8 zeros** |
| no_lvr vs own_latents | 20 | 19 | ≈ tied |
| no_lvr vs random_bbox | 28 | 21 | **+7 no_lvr** |
| own_latents vs random_bbox | 24 | 18 | +6 own_latents |

## Conclusions

**1. The LVR mechanism works architecturally.**
GT latents consistently outperform no-LVR on VisCoT (+0.8pp, +10 net directional wins). The model *can* exploit visual content from latent tokens when that content is correct.

**2. Self-generated latents are nearly uninformative on VisCoT.**
Own latents ≈ no-LVR on VisCoT (tied 20-19 directionally). The MSE training signal is insufficient to make the latent prediction head generate useful representations.

**3. LVR shows clear gains on Blink/VStar — but content doesn't matter.**
Own latents beats no-LVR by +4pp on Blink, +0.8pp on VStar. However, random bbox crops match own latents almost exactly (Blink: 0.623 vs 0.622; VStar: 0.638 vs 0.637). The gain comes from the *structural presence* of latent tokens, not their visual content.

**4. Wrong content is worse than no content (on VisCoT).**
Random bbox is the worst condition on VisCoT (-7 net wins vs no-LVR). When the task requires fine-grained visual reasoning, injecting features from the wrong location actively misleads the model.

**Core diagnosis:** The model has learned that latent token slots *exist* in the sequence and benefits from any visual occupant (Blink/VStar), but has not learned to generate the *right* visual content (VisCoT). The corrupted image ablation is the deciding experiment: if the model can route through GT latents to recover blacked-out bbox regions, the failure is a training signal problem (not architectural).

**Implication for training:** The GRPO/RL training should focus on reward-shaping the latent content — not just whether latent tokens appear, but whether they encode spatially relevant visual information.
