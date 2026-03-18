# Ablation: Does GT Bbox Location Matter for LVR?

## Hypothesis

When using `use_gt=True` during eval, the model receives ground-truth latent embeddings
derived from the image region defined by the GT bounding box. The question is: does the
**spatial location** of that crop carry meaningful signal, or does the model benefit just
from receiving *some* visual crop regardless of where it comes from?

If performance drops when the bbox is randomized (same size, wrong location), it implies
the model is learning to exploit spatial priors baked into the GT latents.

## Data Flow (reference)

```
GT bbox
  └─ center_and_crop_image(img, bbox)   ← ablation happens here (MCDataset.__getitem__)
       └─ cropped_image (PIL)
            └─ get_gt_latent_values()   → pixel_values, image_grid_thw
                 └─ apply_latent_compression()
                      └─ injected as LVR embeddings during generation
```

## Ablation Modes

| Mode | Description | Flag |
|------|-------------|------|
| None (baseline) | GT bbox, correct location | _(omit `--bbox_ablation`)_ |
| `random` | Same bbox size, random location within the image | `--bbox_ablation random` |

## Planned Runs

- [x] **Baseline** — GT bbox, `--lvr --use_gt`
- [x] **Random bbox** — `--bbox_ablation random`, `--lvr --use_gt`

Command template (requires GPU — submit via `srun`, prefer h100):
```bash
srun --partition=h100 --qos=gpu-h100 --job-name=ev --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
    bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
    python -m evals.eval \
        --model_ref /path/to/checkpoint \
        --lvr \
        --batch_size 16 \
        --output_dir results/bbox_ablation \
        [--bbox_ablation random]'
```

Note: eval runs 300 steps max (≈ 4800 samples at batch_size=16).

## Results

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| GT bbox (`--lvr --use_gt`) | **0.793** | — (no GT bboxes) | — (no GT bboxes) |
| Random bbox (`--bbox_ablation random`) | 0.779 | 0.623 | 0.638 |

## Conclusions

**Spatial location matters on VisCoT (+1.4pp).** GT bbox (0.793) beats random bbox (0.779). The directional flip count (33 GT wins vs 16 random wins, net +17) is the strongest flip margin across any pair in the full ablation — stronger even than GT vs no-LVR (+10). The model is genuinely exploiting the correct spatial crop, not just benefiting from receiving *a* visual token.

**Wrong-location crops are worse than no latents at all.** Random bbox is the lowest-performing condition on VisCoT, sitting below no-LVR by 7 net directional wins. When the task requires fine-grained visual reasoning (VisCoT), injecting features from the wrong image region actively misleads the model — the incorrect content is a confound, not a neutral placeholder.

**On Blink/VStar, location is irrelevant.** Random bbox (0.623/0.638) ≈ own latents (0.622/0.637). Those benchmarks don't have GT bboxes and use a random crop at a fixed size. The near-identical performance confirms that on these tasks the structural presence of a latent token drives the gain, not its spatial content.

**Implication.** The model has internalized spatial priors from training — it expects the latent token to encode the specific region the question refers to. This is a positive sign architecturally, but it raises the bar for the latent prediction head: the model needs to generate latents that are both present *and* spatially relevant, not just syntactically well-formed.
