# Ablations: Making the Model Actually Use Latent Tokens

**Problem statement**: MSE and InfoNCE losses go down (the model learns to *produce* hidden states that resemble GT visual embeddings), but `lvr_rate` stays low and `latent_utility/own ≈ latent_utility/random`. This means the model ignores the latent channel at generation time and routes answers through the direct visual tokens instead.

Root cause: no loss term currently penalises *ignoring* the latent tokens. The ablations below attack that from different angles, ordered by priority.

All ablations run on the mini dataset (7 shapes, ~1460 train / ~200 eval, `eval_frac=0.13`).

---

## Ablation 1 — Grayscale Intermediate Image

### Hypothesis
The model may be using colour as a shortcut to match latent embeddings to objects (each Tetris piece has a unique colour). Converting the intermediate image to greyscale forces the model to encode shape/rotation information rather than colour, making the latent representations more abstract and harder to shortcut. If training convergence is basically unchanged, colour was not load-bearing — which validates that the latent channel can work without it.

### Mechanism
Convert `intermediate_img` to greyscale (L→RGB) in `SFTTetrisDataset.__getitem__` before returning it. No other changes.

### Where to hook in
**`src/datasets/sft_tetris_data.py`** — `SFTTetrisDataset.__getitem__`

```python
intermediate_img = Image.open(inter_path).convert("L").convert("RGB")  # greyscale
```

### What to watch
- `mse_loss` / `infonce_loss` curves vs the colour baseline — should converge similarly.
- `latent_utility/gt` — if it drops significantly, the model was relying on colour to decode the latent. If it stays stable, greyscale is safe and we proceed with it for all subsequent ablations.

---

## Ablation 2 — Hard Negatives in InfoNCE

### Hypothesis
The current InfoNCE treats every other sample in the batch as equally valid negatives. Two samples with the *same* Tetris piece but a *different* rotation step have nearly identical GT latent embeddings — they are actually the hardest cases to discriminate — but currently get the same loss weight as completely different pieces. Up-weighting within-family negatives gives stronger gradient signal toward a more discriminative latent space.

### Mechanism
Pass `shape_C_name` and `rot_step` metadata through the collate into the batch. Inside `compute_loss`, build a soft negative-weight matrix: same shape or same rot_step → weight `w_hard` (e.g., 5×), otherwise 1. Apply to the InfoNCE logit matrix before softmax.

### Where to hook in

**`src/datasets/sft_tetris_data.py`** — `SFTTetrisDataset.__getitem__`

Add a metadata dict as the last element of the returned list:
```python
return [
    {"role": "user",      "content": user_content},
    {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
    {"role": "assistant", "content": [{"type": "image", "image": intermediate_img}]},
    {"role": "metadata",  "shape_C_name": data["shape_C_name"],
                          "rot_step": data["option_transforms"][0].get("rot_step", 0)},
]
```

**`src/datasets/sft_tetris_data.py`** — `collate_fn_latent_sft`

Pop the metadata before collation and add to inputs:
```python
metadata = [s.pop(-1) for s in samples]
inputs["shape_C_names"] = [m["shape_C_name"] for m in metadata]
inputs["rot_steps"]     = torch.tensor([m["rot_step"] for m in metadata], dtype=torch.long)
```

**`src/trainer/sft_trainer.py`** — replace `infonce_loss` with a weighted version:

```python
@staticmethod
def infonce_loss(pred, gt, temperature=0.07, shape_C_names=None, rot_steps=None, w_hard=5.0):
    pred_n = F.normalize(pred, dim=-1)
    gt_n   = F.normalize(gt,   dim=-1)
    logits = torch.matmul(pred_n, gt_n.t()) / temperature   # [N, N]

    if shape_C_names is not None and rot_steps is not None:
        same_shape    = torch.tensor([[a == b for b in shape_C_names] for a in shape_C_names],
                                     device=pred.device)
        same_rot_step = (rot_steps.unsqueeze(1) == rot_steps.unsqueeze(0))
        eye           = torch.eye(len(pred), dtype=torch.bool, device=pred.device)
        hard_mask     = (same_shape | same_rot_step) & ~eye
        weight        = torch.ones_like(logits)
        weight[hard_mask] = w_hard
        logits = logits * weight

    labels = torch.arange(pred.size(0), device=pred.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2.0
```

**`src/params.py`** — add `hard_neg_weight: float = field(default=5.0)` to `TrainingParams`.

### Variants to run
| run | `w_hard` | `latent_loss_type`  |
|-----|----------|---------------------|
| 2a  | 5.0      | infonce (hard)      |
| 2b  | 10.0     | infonce (hard)      |
| 2c  | 5.0      | mse + infonce (hard)|

---

## Ablation 3 — Scheduled Sampling for Latent Generation

### Hypothesis
The core train-test mismatch: during training, `<|lvr_sep|>` positions are always filled with GT visual embeddings (teacher forcing). At generation time, the model feeds back its *own* hidden states instead. Since the model never sees its own latent predictions during training, it has no signal to self-correct when they drift from the GT distribution — even though `latent_utility/gt` proves the consumption mechanism works fine when embeddings are good.

Scheduled sampling closes this gap: as training progresses, replace GT embeddings at some `lvr_sep` positions with the model's own predicted hidden states from the previous step. This forces the model to learn to generate latents that are robust enough to feed back into itself.

**Caveat**: this only makes sense once the model's self-generated latents are somewhat coherent (not pure noise). Run this after Ablation 2 has shown some improvement in latent discrimination.

### Mechanism
Two-pass approach in `compute_loss`:
1. **First forward** (standard): inject GT embeddings, collect hidden states at `[lvr_start, sep_0..sep_{k-2}]` — these are the model's predicted embeddings.
2. **Second forward** (scheduled): with probability `p_self`, replace GT embeddings at `sep` positions with the predicted hidden states from pass 1 before running the loss.

Anneal `p_self` from 0 → target (e.g. 0.5) over training so early instability is avoided.

### Where to hook in

**`src/trainer/sft_trainer.py`** — `LantErnSFTrainer.compute_loss`

After the first forward and `pred_list` is assembled, do a conditional second forward:
```python
if self.scheduled_sampling_prob > 0 and torch.rand(1).item() < self.scheduled_sampling_prob:
    # Build new inputs_embeds replacing GT sep embeddings with predicted hidden states
    new_embeds = input_embeddings.clone()
    for b in range(input_ids.shape[0]):
        sep_pos = lvr_sep_mask[b].nonzero(as_tuple=False).squeeze(-1)
        # replace sep_0..sep_{k-1} with predicted hidden states (pred_list[b])
        new_embeds[b, sep_pos] = pred_list[b].detach()  # detach to avoid double backprop
    outputs2 = model(inputs_embeds=new_embeds, ...)
    ce_loss = outputs2.loss   # recompute CE on self-generated path
```

**`src/params.py`** — add to `TrainingParams`:
```python
scheduled_sampling_prob: float = field(default=0.0,
    metadata={"help": "Probability of replacing GT latent embeddings with model's own predictions during training."})
scheduled_sampling_anneal: bool = field(default=True,
    metadata={"help": "Linearly anneal scheduled_sampling_prob from 0 to target over training."})
```

### Variants to run
| run | `scheduled_sampling_prob` | annealing          | prereq           |
|-----|---------------------------|--------------------|------------------|
| 3a  | 0.3                       | no                 | after ablation 2 |
| 3b  | 0.5                       | no                 | after ablation 2 |
| 3c  | 0.0 → 0.5                 | linear over epochs | after ablation 2 |

---

## Ablation 4 — Visual Masking Curriculum

### Hypothesis
If the composite image is zeroed out during some training steps, the model can no longer answer via the direct visual shortcut and is forced to route computation through the latent channel.

### Mechanism
During collation, replace the composite image with a black image with probability `p_mask`. The latent `intermediate_img` is always kept clean. Optionally anneal `p_mask` from 0 → target over training.

### Where to hook in
**`src/datasets/sft_tetris_data.py`** — `collate_fn_latent_sft`

```python
import random as _random
# after image_inputs is assembled, before processor(...)
if image_mask_prob > 0 and _random.random() < image_mask_prob:
    image_inputs = [Image.new("RGB", img.size, (0, 0, 0)) if i == 0 else img
                    for i, img in enumerate(image_inputs)]
```

Pass `image_mask_prob` down via `make_tetris_data_module` into the partial-applied collate.

**`src/params.py`** — `SFTDataParams`:
```python
image_mask_prob: float = field(default=0.0)
image_mask_anneal: bool = field(default=False)
```

### Variants to run
| run | `image_mask_prob` | annealing           |
|-----|-------------------|---------------------|
| 4a  | 0.3               | no                  |
| 4b  | 0.7               | no                  |
| 4c  | 0.0 → 0.7         | linear over epochs  |

### Expected signal
`latent_utility/own − latent_utility/zeros` should open up. Watch that `latent_utility/gt` does not degrade — that is the ceiling.

---

## Shared eval signal to watch

All runs should be compared on the same four `LatentUtilityCallback` conditions logged at each eval step:

| metric | what it measures |
|--------|-----------------|
| `latent_utility/gt`     | upper bound — model given correct latent |
| `latent_utility/own`    | **the target metric** — model uses its own latents |
| `latent_utility/random` | lower bound with noise |
| `latent_utility/zeros`  | lower bound with zeros |

A successful ablation shows `own − zeros` opening up. If `own ≈ zeros`, latents are still ignored regardless of the training loss.
