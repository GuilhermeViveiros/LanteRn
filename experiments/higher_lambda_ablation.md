# Ablation: Higher MSE Weight (λ = 0.8)

## Motivation

The standard model was trained with `loss = ce_loss + γ * mse_loss` at γ=0.1, meaning
the MSE reconstruction term is lightly weighted. All ablations on that model show
own-latents ≈ no-LVR on VisCoT — the latent prediction head produces uninformative embeddings.

**Question:** Does increasing γ (more MSE pressure) force the model to generate better
visual reconstructions, or does it simply divert capacity from CE at the cost of answer quality?

This is explicitly a trade-off probe: we expect either (a) higher reconstruction fidelity
at the cost of task alignment, or (b) both degrade because MSE is the wrong objective regardless.
Either outcome is informative — if (a), we know the issue is the objective, not the weight.

## Setup

- **Model:** `sft_mse_lt_8_lambda_0.8/checkpoint-????` ← currently training
- **Base:** same architecture and data as `sft_mse_lt_8_lambda_0.1/checkpoint-1062`
- **Only change:** γ = 0.8 (vs 0.1 baseline)
- **Dataset for eval:** same as all prior ablations (VisCoT MC test, Blink, VStar)

## Planned Runs

- [ ] **Train** — SFT with γ=0.8 (in progress)
- [ ] **Eval (VisCoT)** — `--lvr --use_gt`
- [ ] **Eval (VisCoT)** — `--lvr --no-use_gt` (own latents)
- [ ] **Eval (VisCoT)** — `--no-lvr`
- [ ] **Eval (Blink)** — `--lvr` / `--no-lvr`
- [ ] **Eval (VStar)** — `--lvr` / `--no-lvr`

## Results

| Condition | VisCoT | Blink Avg | VStar Avg |
|-----------|-------:|----------:|----------:|
| GT latents (γ=0.8) | | | |
| Own latents (γ=0.8) | | | |
| No LVR (γ=0.8) | | | |
| **Baseline** GT latents (γ=0.1) | **0.793** | — | — |
| **Baseline** Own latents (γ=0.1) | 0.784 | 0.622 | 0.637 |
| **Baseline** No LVR (γ=0.1) | 0.785 | 0.582 | 0.629 |

## Expected Outcomes & Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Own latents (γ=0.8) > Own latents (γ=0.1), CE drops | More MSE pressure improves reconstruction but hurts task — MSE is the wrong objective |
| Own latents (γ=0.8) ≈ Own latents (γ=0.1), CE ≈ same | Weight doesn't matter; MSE is simply insufficient |
| Own latents (γ=0.8) > Own latents (γ=0.1), CE same | Surprising positive — MSE scale was the bottleneck |

## Conclusions

*(pending)*
