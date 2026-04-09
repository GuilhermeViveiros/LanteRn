# Ablation: Contrastive Loss for Latent Token Reconstruction

## Motivation

SFT trains latent positions with MSE between predicted hidden states and GT compressed visual
embeddings. MSE enforces point-wise Euclidean reconstruction — it doesn't shape the relative
geometry between different latents. At test time the model generates its own latent tokens and
the GT/own gap is large, suggesting self-generated latents are poorly calibrated.

Hypothesis: InfoNCE contrastive loss pulls each predicted latent toward its GT while pushing it
away from other samples' GTs, encouraging the model to generate latents that occupy the right
neighborhood in embedding space — more decodable at test time.

## Design

**Loss variants** (controlled via `--latent_loss_type`):
- `mse` — baseline, point-wise L2
- `cosine` — point-wise cosine, scale-invariant
- `infonce` — bidirectional NT-Xent, relative neighborhood structure

**Block-level aggregation**: rather than treating each of the `latent_size` tokens independently,
the full block is flattened into a single `[latent_size × D]` vector per sample before InfoNCE.
This treats the latent block as a single "visual thought" — consistent with how it's used at inference.

**Hard negative batching** (`--use_family_batching True --family_batch_key shape_C_name`):
batches are grouped by `shape_C_name` so all rotation strips in a batch show the same object in
different rotations. Negatives are maximally hard: same shape, different rotation state.

## Setup

- **Dataset:** Tetris analogy (`tetris_analogy`, `train.json`)
- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, latent_size=8, γ=0.1, τ=0.07
- **Batch:** 6 per GPU × 4 GPUs = 24 local; 192 global
- **Script:** `scripts/finetune_tetris_sft_3b.sh`

## Runs

| Run | `latent_loss_type` | family batching | wandb |
|-----|-------------------|-----------------|-------|
| Baseline | `mse` | no | — |
| Cosine | `cosine` | no | — |
| InfoNCE (random) | `infonce` | no | — |
| InfoNCE + hard negatives | `infonce` | `shape_C_name` | — |

## Evaluation

Compare GT/own gap from `LatentUtilityCallback` (`latent_utility/gt` vs `latent_utility/own`)
across checkpoints. A smaller gap = self-generated latents are more useful.

## Results

| Run | GT acc | Own acc | GT/Own gap |
|-----|-------:|--------:|-----------:|
| MSE (baseline) | — | — | — |
| Cosine | — | — | — |
| InfoNCE (random batching) | — | — | — |
| InfoNCE + hard negatives | — | — | — |
