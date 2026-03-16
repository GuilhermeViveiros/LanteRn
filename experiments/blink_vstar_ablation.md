# Ablation: Blink & VStar — LVR vs No-LVR vs Random Bbox

## Hypothesis

Same as viscot ablations but on held-out benchmarks (Blink, VStar).
Since these benchmarks have no GT bboxes, the random bbox ablation uses a
random crop (~40% of image dimensions) as the latent input.

Conditions:
- `--lvr --no-use_gt`: model generates its own latents freely
- `--no-lvr`: no latent generation (text-only baseline)
- `--lvr --bbox_ablation random`: random image crop as latent input

## Setup

- **Model:** `sft_mse_lt_8_lambda_0.1/checkpoint-1062`
- **Blink categories:** Object_Localization (n=122), Spatial_Relation (n=143)
- **VStar categories:** direct_attributes (n=115), relative_position (n=76)
- **Steps:** 300 max, batch_size=4

## Planned Runs

- [x] Blink — `--lvr --no-use_gt`
- [x] Blink — `--no-lvr`
- [x] Blink — `--lvr --bbox_ablation random`
- [x] VStar — `--no-lvr`
- [x] VStar — `--lvr --no-use_gt`
- [x] VStar — `--lvr --bbox_ablation random`

## Command Template

```bash
srun --partition=h100 --qos=gpu-h100 --job-name=ev --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
           export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
           python -u -m evals.[blink_eval|vstar_eval] \
             --model_ref /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/ \
             --[lvr|no-lvr] [--no-use_gt] [--bbox_ablation random] \
             --batch_size 4 --output_dir results/[gt_latent_signal|bbox_ablation] \
             > results/ev_[blink|vstar]_[lvr|nolvr|rbox].log 2>&1'
```

## Results

### Summary (all three benchmarks)

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| `--lvr --no-use_gt` | 0.784 | 0.622 | 0.637 |
| `--no-lvr` | 0.785 | 0.582 | 0.629 |
| `--lvr --bbox_ablation random` | 0.779 | 0.623 | 0.638 |

### Blink (per-category)

| Condition | Object_Localization | Spatial_Relation | Avg |
|-----------|--------------------:|----------------:|----:|
| `--lvr --no-use_gt` | 0.525 | 0.706 | 0.622 |
| `--no-lvr` | 0.516 | 0.636 | 0.582 |
| `--lvr --bbox_ablation random` | 0.533 | 0.699 | 0.623 |

### VStar (per-category)

| Condition | direct_attributes | relative_position | Avg |
|-----------|------------------:|------------------:|----:|
| `--lvr --no-use_gt` | 0.696 | 0.553 | 0.637 |
| `--no-lvr` | 0.678 | 0.553 | 0.629 |
| `--lvr --bbox_ablation random` | 0.704 | 0.539 | 0.638 |

## Conclusions

LVR consistently outperforms no-LVR on both benchmarks (~+4pp on Blink, ~+0.8pp on VStar).
However, random bbox crops perform nearly identically to the model's own predicted latents.

This is likely a **training distribution artifact**: the model was trained with latent tokens
present in the sequence, so at inference time it "expects" that structural slot to be filled.
Any visual crop — even a meaningless one — satisfies the learned distribution.

The visual *content* of the latent matters less than its *structural presence* in the sequence.
This is consistent with the viscot GT ablation where GT bbox > random (+1.4pp) — there,
the question explicitly requires fine-grained visual reasoning so content matters more.

Implication: the LVR gain on blink/vstar may be partly illusory — the model benefits
from seeing latent tokens rather than from seeing the right visual information.
