# Ablation: Latent Retrieval — Can the Model Predict the Right Visual Content?

## Motivation

All prior ablations show own-latents ≈ no-LVR on VisCoT. The corrupted image ablation
confirms this is a training signal problem — the architecture can route through the latent
channel, but the latent prediction head generates uninformative embeddings.

**The question:** is this a representation failure (predicted latents are noise) or an
interface/routing failure (latents encode something but the LM doesn't use it)?

Reconstruction performance distinguishes these cases. If the model's predicted latents can
retrieve the correct GT latent from a gallery, the information is encoded — the problem
is downstream (the LM ignores it). If retrieval is at chance, the latent head itself is broken.

## Hypothesis

The model's predicted latent embeddings should be nearest-neighbor close to the corresponding
GT latent embeddings if the latent prediction head has learned to reconstruct useful visual
representations. We can test this directly: build a GT gallery from the test set, run inference
to get predicted latents, and measure top-k retrieval accuracy.

## Method

**Gallery:** For each test sample, compute GT latent embeddings:
```
bbox crop → vision encoder → apply_latent_compression() → latent_size vectors
```
Mean-pool across the `latent_size` dimension → single vector per sample.

**Queries:** Run LantErn inference with `--lvr --no-use_gt` on the same samples.
At each `<|lvr_start|>...<|lvr_end|>` block, collect the predicted hidden states
used as latent embeddings. Mean-pool → single query vector per sample.

**Retrieval:** For each query, rank all gallery vectors by cosine similarity.
Check if the correct GT entry ranks in top-k.

**Oracle check:** Use GT latents as both query and gallery. Should give retrieval@1 = 1.0.
Confirms the setup is correct before interpreting model predictions.

## Metrics

| Metric | Description |
|--------|-------------|
| Retrieval@1 | Fraction of queries where correct GT is rank-1 |
| Retrieval@5 | Fraction of queries where correct GT is in top-5 |
| Retrieval@10 | Fraction of queries where correct GT is in top-10 |
| Mean Rank | Average rank of correct GT across all queries |
| MRR | Mean reciprocal rank |
| Random baseline | 1/N (N = gallery size) |

Also report per-token retrieval (before mean-pooling) to check if signal concentrates
in specific latent positions.

## Setup

- **Model:** `sft_mse_lt_8_lambda_0.1/checkpoint-1062`
- **Dataset:** VisCoT MC test (same split as all prior ablations)
- **Gallery size:** full test set (~4800 samples)
- **Script:** TBD — `evals/latent_retrieval_eval.py`

## Planned Runs

- [ ] Build GT gallery (run vision encoder + apply_latent_compression on all test samples)
- [ ] Run inference, collect predicted latent hidden states
- [ ] Compute cosine similarity, report retrieval@k and mean rank
- [ ] Oracle check: GT-as-query vs GT gallery
- [ ] Compare: baseline model (γ=0.1) vs filtered-data model (once available)

## Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Retrieval@k >> random, own ≈ no-LVR on VisCoT | Interface failure — information encoded but not used by LM |
| Retrieval@k ≈ random | Representation failure — latent head generates noise |
| Retrieval@k >> random, own > no-LVR on VisCoT | Both working — latents are good and used |

This directly disambiguates cause (1) vs cause (2)/(3) from the three-hypothesis framing:
1. Representation failure: latent head can't encode meaningful information
2. Interface failure: information encoded but LM can't access/use it
3. Shortcut learning: model learns to ignore latent pathway during optimization

If retrieval is high → (1) ruled out, problem is (2) or (3).
If retrieval is low → (1) is the cause; the MSE loss is insufficient to train the latent head.

## Results

N = 1200, all samples emitted latents (100%).

| Condition | R@1 | R@5 | R@10 | Mean Rank | MRR |
|-----------|----:|----:|-----:|----------:|----:|
| Oracle (GT → GT gallery) | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0000 |
| Own latents (γ=0.1) | 0.0017 | 0.0075 | 0.0150 | 533.1 | 0.0096 |
| Own latents (filtered data) | — | — | — | — | — |
| Random baseline (1/N) | 0.0008 | — | — | ~600 | — |

Lift over random R@1: **2×**

## Conclusions

**Representation failure.** Predicted latents retrieve the correct GT embedding at essentially chance level (R@1 = 0.0017, only 2× above random 0.0008, mean rank 533/1200). The model emits latent tokens on every sample, so the routing/interface is working — but the MSE-trained latent head is generating near-random embeddings. The information is simply not encoded.

This rules out interface/routing failure as the primary cause. The MSE loss (γ=0.1) is insufficient to train the latent head to produce meaningful visual representations. Next steps: stronger latent supervision, higher γ, or filtering to samples where the latent matters.
