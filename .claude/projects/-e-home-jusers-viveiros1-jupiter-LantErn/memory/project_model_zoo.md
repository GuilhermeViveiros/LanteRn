---
name: model_zoo — external model comparisons
description: Location and purpose of model_zoo/ folder; which models are cloned and how they're compared to LantErn
type: project
---

`model_zoo/` (at repo root) holds clones of external models for apples-to-apples comparison against LantErn.

Each model gets its own subfolder with a model-specific `evals/latent_retrieval_eval.py` that mirrors LantErn's retrieval probe.

**Why:** test whether LantErn's representation failure (R@1 ≈ random on N=1200) is unique or shared by similar latent visual reasoning models.

**How to apply:** when adding a new model, clone into `model_zoo/<name>/`, add `model_zoo/<name>/evals/latent_retrieval_eval.py` adapted to that model's latent extraction mechanism.

## Models

| Model | Path | Status |
|-------|------|--------|
| LVR (VincentLeebang/lvr) | `model_zoo/lvr/` | eval script written, no checkpoint run yet |
