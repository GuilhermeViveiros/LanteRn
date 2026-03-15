# Ablation: Do Latent Tokens Carry Visual Signal? (Corrupted Image Training)

## Motivation

Previous eval ablations showed near-identical performance across:
- GT latent tokens (`--lvr`, `use_gt=True`)
- Random latent tokens (`--bbox_ablation random`)
- Constant latent tokens
- No latent tokens (`--no-lvr`)

This suggests the model is **not using latent tokens** during inference, despite being trained on them.
The question is: **is the root cause in the data/training, or in the architecture?**

## Hypothesis

If the model were truly learning to rely on latent tokens, it should be able to use them to
compensate for missing visual information. We can test this by training a new model where:

- The **main input image** has its bbox regions blacked out → the raw visual context is degraded
- The **latent tokens** are computed from clean, uncorrupted bbox crops → the LVR signal is intact

If latent tokens are usable, a model trained this way should learn to route through them to recover
the missing information. If performance stays poor (or matches a no-latent baseline), it indicates
the training signal for latent tokens is broken regardless of data.

**This directly answers: is the failure a data problem or an architectural/loss problem?**

## Data Flow (with corruption)

```
main image (corrupted)
  └─ bbox regions blacked out in PIL before collation
       └─ fed to vision encoder as usual → visual tokens for attention
            └─ model sees degraded image context

bbox crops (clean, from original image)
  └─ center_and_crop_image(img, bbox)     ← uses original img, not corrupted
       └─ apply_latent_compression()
            └─ injected as LVR embeddings at <|lvr_sep|> positions
                 └─ only source of clean visual info for the bbox region
```

## Corruption Strategy

**Recommended: bbox blackout** — zero out the bbox rectangle in the main image before processing.

Rationale:
- Creates a precise information gap exactly where the latent tokens provide signal
- Clean test: the *only* way to recover bbox-region info is via the latent tokens
- Avoids confounds from full-image corruption (model may fail for unrelated reasons)

Alternative (if bbox blackout is too easy): Gaussian noise over the full image.

## Planned Runs

- [ ] **Train** — SFT with bbox blur corruption
- [ ] **Eval** — corrupted model, `--lvr`
- [ ] **Eval** — corrupted model, `--no-lvr`

## Command Template (Training — TBD after branch is created)

```bash
# Branch: agnv/corrupt-image-ablation (to be created from main)
# Cluster: hades (h200), 4 GPUs
srun --partition=hades --qos=gpu-h200 --job-name=train-corrupt --time=3-00:00:00 \
     --nodes=4 --gpus-per-node=4 --tasks-per-node=1 --mem=200GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  bash scripts/finetune_lantern_sft_3b.sh'
```

Set in `finetune_lantern_sft_3b.sh` before running:
```bash
RUN_NAME="sft_mse_lt_8_lambda_0.1_corrupt_blur"
# add --corrupt_image True --corruption_type bbox_blur to the deepspeed call
```

## Command Template (Eval)

```bash
srun --partition=hades --qos=gpu-h200 --job-name=ev --time=04:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  python -m evals.eval \
    --model_ref /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_corrupt_bbox/<checkpoint>/ \
    --[lvr|no-lvr] \
    --batch_size 16 \
    --output_dir results/corrupted_image_ablation'
```

## Files to Change (backlog — not yet implemented)

| File | Change |
|------|--------|
| `src/datasets/sft_viscot_data.py` | Black out bbox regions in main image in `__getitem__`; also remove `pdb.set_trace()` at line 103 |
| `src/params.py` | Add `corrupt_image: bool = False` and `corruption_type: str = "bbox_blackout"` to `SFTDataParams` |
| `src/train/train_sft.py` | Pass new params to dataset constructor |

## Results

| Run | Latent Mode | Accuracy | Notes |
|-----|-------------|----------|-------|
| Corrupted + GT latents | `--lvr` | — | key result |
| Corrupted + no latents | `--no-lvr` | — | lower bound |

## Expected Outcomes & Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Corrupted+GT >> Corrupted+no-latent | Latent tokens carry real signal; prior failure was a data/training issue |
| Corrupted+GT ≈ Corrupted+no-latent (both low) | Architecture cannot route through latent tokens; deeper issue |
| Corrupted+GT ≈ Corrupted+no-latent (both high) | Model somehow recovers without latents; corruption wasn't effective enough |

## Conclusions

_(fill in after runs complete)_
