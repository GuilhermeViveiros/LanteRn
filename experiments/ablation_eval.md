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

| Condition | Accuracy |
|-----------|----------|
| `gt_latents` | **0.7933** |
| `zeros` | 0.7858 |
| `no_lvr` | 0.7850 |
| `own_latents` | 0.7842 |
| `random_bbox` | 0.7792 |

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

**GT latents carry genuine signal.** +10 net wins over no_lvr, +11 over own_latents.
The model CAN use visual information from GT bbox crops effectively.

**The model's own predicted latents are nearly uninformative.** Own_latents ≈ no_lvr
(tied 20-19). The latent prediction head is not learning to generate useful visual
representations — its outputs are functionally equivalent to not having latents at all.

**Random bbox actively misleads.** random_bbox < no_lvr (-7 net). Injecting visual
features from the wrong spatial location is worse than injecting nothing. The model
attends to latent positions and is sensitive to their content — but only benefits when
the content is correct.

**Zeros are safer than random crops.** zeros > random_bbox (+8 net). Zero embeddings
are marginally better than no_lvr (+1 net) — the structural presence of the latent slot
provides a tiny benefit, but without content it's nearly neutral.

**Core diagnosis:** The LVR mechanism works (model uses latents, GT latents help), but
the model has not learned to generate useful latent content on its own. The training
signal for the latent prediction head (MSE loss on compressed visual features) is not
sufficient to produce representations that improve downstream accuracy.

**Implication for training:** The GRPO/RL training should focus on reward-shaping the
latent content — not just whether latent tokens appear, but whether they encode
spatially relevant visual information.
