# LantErn: Latent Visual Reasoning

> **⚠️ Beta — work in progress.** This repository is the shared codebase for two ongoing research papers. APIs and training pipelines may change without notice.

<p align="center">
  <img src="imgs/lantern_fig.png" alt="LantErn Architecture" width="500"/>
</p>

---

## Updates

**April 2026** — LantErn accepted at the [ICLR 2026 Workshop: Multimodal Intelligence](https://multimodal-intelligence.github.io/) 🎉 See you in Brazil 🇧🇷

---

## Papers

This codebase covers two works:

**1. LantErn: Latent Visual Reasoning in Vision-Language Models**
[![arXiv](https://img.shields.io/badge/arXiv-2603.25629-b31b1b.svg)](https://arxiv.org/abs/2603.25629)
> Introduces Latent Visual Reasoning (LVR) tokens — compressed visual embeddings interleaved with text during reasoning. Extends Qwen2.5-VL with SFT and GRPO training to emit non-verbalized visual representations.

**2. What's Holding Back Latent Visual Reasoning?** *(in proceedings — link coming soon)*
> Follow-up work using the Tetris synthetic analogy benchmark to diagnose what prevents LVR from working. Isolates representation failure from interface/routing failure via controlled experiments.

**Models on HuggingFace:** [AGViveiros/lantern-models](https://huggingface.co/collections/AGViveiros/lantern-models)

---

## Overview

LantErn extends Qwen2.5-VL to produce interleaved text and **Latent Visual Reasoning (LVR)** tokens. Instead of always describing what it sees in words, the model can emit compressed visual embeddings (`<|lvr_start|>...<|lvr_end|>`) during its reasoning chain.

**Special tokens:**
- `<|lvr_start|>` — begins a latent reasoning block
- `<|lvr_sep|>` — placeholder replaced by compressed visual embeddings during training
- `<|lvr_end|>` — ends a latent reasoning block

**Training loss:** `γ · CE_loss(text tokens) + MSE_loss(latent positions)`

---

## Installation

```bash
git clone https://github.com/GuilhermeViveiros/LantErn.git
cd LantErn
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=/path/to/LantErn:$PYTHONPATH
```

---

## Training

### SFT — VisCoT (3B)
```bash
sbatch scripts/finetune_lantern_sft_3b.sh
```

### SFT — VisCoT (7B)
```bash
sbatch scripts/finetune_lantern_sft_7b.sh
```

### SFT — Tetris (LVR model)
```bash
sbatch scripts/finetune_tetris_sft_3b.sh
```

### SFT — Tetris (NTP baseline, no LVR)
```bash
sbatch scripts/finetune_tetris_ntp_3b.sh
```

### GRPO (RL fine-tuning from SFT checkpoint)
```bash
sbatch scripts/finetune_lantern_rl_3b.sh
```

Key hyperparameters (see `src/params.py` for all defaults):

| Param | Description | Default |
|-------|-------------|---------|
| `--latent_size` | Number of LVR tokens per block | `-1` (dynamic) |
| `--gamma` | Weight on CE loss (vs MSE) | `0.1` |
| `--latent_loss_type` | `mse` \| `infonce` \| `cosine` | `mse` |
| `--freeze_vision_tower` | Freeze vision encoder | `True` |
| `--use_lvr` | Enable LVR tokens (False = NTP baseline) | `True` |

Add `--dummy True` to any training command to run on the first 1000 samples for quick testing.

---

## Evaluation

### VisCoT / BLINK / V* (combined)
```bash
python -m evals.viscot_blink_vstar_eval \
    --model_ref /path/to/checkpoint \
    --benchmarks viscot blink vstar \
    --output_dir results/
```

### Latent Retrieval (R@k, MRR)
```bash
python -m evals.latent_retrieval_eval \
    --model_ref /path/to/checkpoint \
    --output_dir results/latent_retrieval
```

### OOD Generalization (Tetris held-out shapes)
```bash
python -m evals.generalization \
    --checkpoints_dir /path/to/checkpoints \
    --held_out_path /path/to/analogy_data/held_out/held_out.json \
    --eval_path /path/to/analogy_data/eval.json \
    --n_samples 200 --batch_size 8
```

---

## Tetris Synthetic Benchmark

The Tetris benchmark (`synthetic/Tetris/`) is a controlled analogy task used in Paper 2. The model is given a visual analogy over Tetris pieces and must identify the correct transformation applied to a query shape.

Generate the dataset:
```bash
# Training + eval split
python -m synthetic.Tetris.create_dataset \
    --output_dir /path/to/analogy_data

# Held-out shapes (for generalization eval)
python -m synthetic.Tetris.create_dataset \
    --output_dir /path/to/analogy_data \
    --held_out
```

---

## Data Format (SFT)

```json
{
  "question": "...",
  "img_path": "/path/to/image.jpg",
  "bboxs": [[x1, y1, x2, y2], ...],
  "reasoning_traces": {
    "pre_visual_text_think": "...",
    "post_visual_latent_reasoning": ["...", "..."],
    "text_think": "...",
    "answer": "..."
  }
}
```

`len(bboxs)` must equal `len(post_visual_latent_reasoning)`.

---

## Project Structure

```
LantErn/
├── src/
│   ├── models/             # load_model(), patched Qwen2.5-VL forward
│   ├── train/              # train_sft.py, train_grpo.py
│   ├── trainer/            # LantErnSFTrainer, LantErnGRPOTrainer
│   ├── lantern_generate/   # Custom token-by-token generation loop
│   ├── datasets/           # SFT dataset loaders (viscot, tetris, monet)
│   ├── rl/                 # GRPO reward functions
│   └── params.py           # All config dataclasses
├── evals/
│   ├── viscot_blink_vstar_eval.py   # Main eval (VisCoT, BLINK, V*)
│   ├── latent_retrieval_eval.py     # Retrieval metrics (R@k, MRR)
│   └── generalization.py            # OOD generalization on Tetris
├── synthetic/Tetris/       # Tetris analogy dataset generator
├── scripts/                # SLURM training scripts + DeepSpeed configs
└── experiments/            # Analysis notebooks and notes
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{lantern2025,
  title={LantErn: Latent Visual Reasoning in Vision-Language Models},
  author={Viveiros, Guilherme and others},
  journal={arXiv preprint arXiv:2603.25629},
  year={2025}
}
```

Paper 2 citation coming soon.
