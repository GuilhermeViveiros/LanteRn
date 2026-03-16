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

- [x] **Train** — SFT with bbox blackout corruption
- [ ] **Eval (VisCoT)** — corrupted model, `--lvr --use_gt`
- [ ] **Eval (VisCoT)** — corrupted model, `--lvr --no-use_gt`
- [ ] **Eval (VisCoT)** — corrupted model, `--no-lvr`
- [ ] **Eval (Blink)** — corrupted model, `--lvr`
- [ ] **Eval (Blink)** — corrupted model, `--no-lvr`
- [ ] **Eval (VStar)** — corrupted model, `--lvr`
- [ ] **Eval (VStar)** — corrupted model, `--no-lvr`

**Checkpoint:** `/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1_corrupt_bbox_blackout/checkpoint-1062`

## Command Templates (Eval)

```bash
CKPT=/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1_corrupt_bbox_blackout/checkpoint-1062

# VisCoT
srun --partition=h100 --qos=gpu-h100 --job-name=ev-corrupt-viscot --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  python -u -m evals.eval \
    --model_ref $CKPT --[lvr|no-lvr] --batch_size 4 \
    --output_dir results/corrupted_image_ablation > results/ev_corrupt_viscot_[lvr|nolvr].log 2>&1'

# Blink
srun --partition=h100 --qos=gpu-h100 --job-name=ev-corrupt-blink --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  python -u -m evals.blink_eval \
    --model_ref $CKPT --[lvr|no-lvr] --batch_size 4 \
    --output_dir results/corrupted_image_ablation > results/ev_corrupt_blink_[lvr|nolvr].log 2>&1'

# VStar
srun --partition=h100 --qos=gpu-h100 --job-name=ev-corrupt-vstar --time=06:00:00 \
     --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
  bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
  python -u -m evals.vstar_eval \
    --model_ref $CKPT --[lvr|no-lvr] --batch_size 4 \
    --output_dir results/corrupted_image_ablation > results/ev_corrupt_vstar_[lvr|nolvr].log 2>&1'
```

## Files to Change (backlog — not yet implemented)

| File | Change |
|------|--------|
| `src/datasets/sft_viscot_data.py` | Black out bbox regions in main image in `__getitem__`; also remove `pdb.set_trace()` at line 103 |
| `src/params.py` | Add `corrupt_image: bool = False` and `corruption_type: str = "bbox_blackout"` to `SFTDataParams` |
| `src/train/train_sft.py` | Pass new params to dataset constructor |

## Results

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| Corrupted + GT latents (`--lvr --use_gt`) | **0.747** | — (no GT bboxes) | — (no GT bboxes) |
| Corrupted + random bbox (`--bbox_ablation random`) | 0.677 | 0.564 | 0.572 |
| Corrupted + own latents (`--lvr --no-use_gt`) | 0.617 | 0.560 | 0.548 |
| Corrupted + no latents (`--no-lvr`) | 0.607 | 0.385 | 0.397 |

*Blink/VStar: corruption not applied at eval (no GT bboxes on those benchmarks). VisCoT: 300 steps × bs=1.*

## Expected Outcomes & Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Corrupted+GT >> Corrupted+no-latent | Latent tokens carry real signal; prior failure was a data/training issue |
| Corrupted+GT ≈ Corrupted+no-latent (both low) | Architecture cannot route through latent tokens; deeper issue |
| Corrupted+GT ≈ Corrupted+no-latent (both high) | Model somehow recovers without latents; corruption wasn't effective enough |

## Conclusions

**The architecture works — the training signal doesn't.**
GT latents +14pp over no-LVR on VisCoT (0.747 vs 0.607). The model CAN route through the latent channel to recover information lost in the corrupted main image. This is the clearest evidence yet that the mechanism is architecturally sound.

**Own latents remain uninformative.**
Own latents ≈ no-LVR on VisCoT (+1pp). Despite training with corruption that forces the model to depend on latents, the latent prediction head still fails to generate useful visual representations. The MSE loss on compressed features is not sufficient to teach the model what to encode.

**Corruption training increased latent slot dependency on all benchmarks.**
On Blink/VStar, no-LVR dropped massively (-17.5pp Blink, -15pp VStar) vs the original model (-4pp, -0.8pp). The model learned to structurally depend on the latent slot. But since it can't fill it with useful content, own latents ≈ random bbox ≈ any visual noise.

**Random bbox is a surprisingly strong baseline (+7pp over no-LVR on VisCoT).**
Any visual crop from the image is more useful than nothing when the main image is corrupted — the model extracts some partial signal even from the wrong region.

**Root cause:** The latent prediction head receives a MSE loss against compressed GT visual features, but there is no direct gradient signal encouraging the model to encode information that is *actually used* downstream for the answer. The model learns to produce embeddings that minimize reconstruction error in feature space, not embeddings that drive correct predictions.
