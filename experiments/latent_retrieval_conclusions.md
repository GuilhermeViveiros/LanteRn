# Latent Retrieval Evaluation — Conclusions

**Date:** 2026-03-25
**Eval:** `evals/latent_retrieval_eval.py` — mean-pooled predicted latents vs GT gallery (cosine similarity, gallery size = 300)

## Results

| Model | R@1 | R@5 | R@10 | Mean Rank | MRR | n_with_latents |
|-------|-----|-----|------|-----------|-----|----------------|
| Random baseline | 0.333% | 1.67% | 3.33% | 150.5 | — | — |
| SFT (checkpoint-1062) | 0.667% | 3.33% | 5.67% | 134.0 | 0.0288 | 300/300 |
| GRPO/RL (checkpoint-1500) | 0.000% | 2.67% | 5.33% | 140.3 | 0.0238 | 300/300 |

Oracle R@1 = 1.0 for both models (setup validated).

## Conclusions

1. **Both models produce latents near random** — R@1 is at or below random (0.33%), confirming the predicted latents carry very little content-specific visual information when mean-pooled.

2. **SFT slightly outperforms GRPO** on all metrics (R@1: 0.67% vs 0%, MRR: 0.029 vs 0.024, mean rank: 134 vs 140). GRPO training does not improve — and may slightly hurt — latent representational quality.

3. **This is a representation failure, not an interface failure** — both models reliably emit `<|lvr_start|>`…`<|lvr_end|>` (100% coverage), so the routing mechanism works. But the content of the predicted latents is near-random with respect to the GT visual embeddings.

4. **Implication:** The MSE loss during SFT is not sufficient to force the predicted latents to align with GT visual features in a retrieval-meaningful way. The latents may be encoding something (e.g. positional/structural patterns) that is useful for downstream answer generation but not for cross-sample retrieval.

## Next Steps

- Check whether task accuracy is correlated with latent quality per sample (do better-retrieved samples answer correctly more often?).
- Try retrieval with the uncompressed hidden states (pre-mean-pool) to see if the averaging is destroying structure.
- Consider adding a contrastive/retrieval auxiliary loss during SFT to encourage discriminative latents.
