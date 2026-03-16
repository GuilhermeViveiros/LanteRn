# Ablation: Do GT Latent Tokens Contribute Signal?

## Hypothesis

The LantErn model is trained to use LVR latent embeddings derived from GT bounding-box crops.
At eval time with `use_gt=True`, those GT latents are injected into the generation.
The question is: does the model actually exploit that visual signal, or is the gain coming
purely from the text/image context?

If accuracy drops when latents are withheld (same model, same image+question, no latent injection),
it confirms GT latents carry meaningful signal beyond the base image context.

## Setup

- **Model:** `sft_mse_lt_8_lambda_0.1/checkpoint-1062`
- **Dataset:** viscot (300 steps × batch 16 ≈ 4800 samples)
- Both runs use the same LantErn checkpoint; only latent injection differs.

## Planned Runs

- [x] **With GT latents** — `--lvr --use_gt`
- [x] **Model's own latents** — `--lvr --no-use_gt`
- [x] **Without latents** — `--no-lvr` (same model, no latent injection)

## Command Template

```bash
srun --partition=h100 --qos=gpu-h100 --job-name=ev --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  python -m evals.eval \
    --model_ref /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/ \
    --[lvr|no-lvr] \
    --batch_size 16 \
    --output_dir results/gt_latent_signal'
```

## Results

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| GT latents (`--lvr --use_gt`) | **0.793** | — (no GT bboxes) | — (no GT bboxes) |
| Own latents (`--lvr --no-use_gt`) | 0.784 | 0.622 | 0.637 |
| No latents (`--no-lvr`) | 0.785 | 0.582 | 0.629 |

## Conclusions

GT latents (+0.9pp over model's own latents, +0.8pp over no-LVR) confirm that the bbox
content carries real signal when present. However, the gap between model's own latents
and no-LVR is small (+0.0pp on viscot), suggesting the model's self-generated latents
are close to uninformative on this benchmark.

Note: part of the LVR gain may stem from training distribution — the model was trained
with latent tokens present, so their structural presence alone may help regardless of content.
The GT bbox ablation (where content matters more) is the cleaner signal.
