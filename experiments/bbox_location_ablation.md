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

| Run | Accuracy | Latent Ratio | Notes |
|-----|----------|--------------|-------|
| Baseline (GT bbox) | 0.793 | 0.491 | batch_size=4, 300 steps |
| Random bbox | 0.779 | 0.491 | batch_size=4, 300 steps |

## Conclusions

_(fill in after runs complete)_
