# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Set `PYTHONPATH` if running scripts directly:
```bash
export PYTHONPATH=/path/to/LantErn:$PYTHONPATH
```

## Common Commands

### SFT Training (single GPU)
```bash
python -m src.train.train_sft \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_path /path/to/LantErn_VisCot_data.json \
    --output_dir /path/to/checkpoints \
    --latent_size 8 \
    --gamma 0.1
```

### SFT Training (multi-GPU with DeepSpeed)
```bash
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3.json \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --latent_size 8 --gamma 0.1 ...
# See scripts/finetune_lantern_sft_3b.sh for a full example
```

### GRPO Training
```bash
python -m src.train.train_grpo  # configured via GRPOArguments in src/params.py
```

### Evaluation (all benchmarks)
```bash
python -m evals.eval --model_ref /path/to/checkpoint --lvr --batch_size 16 --output_dir results/
python -m evals.vstar_eval --model_ref /path/to/checkpoint --lvr --batch_size 16
python -m evals.blink_eval --model_ref /path/to/checkpoint --lvr --batch_size 16
# Or use the convenience script:
bash scripts/eval.sh
```

Use `--no-lvr` to disable latent visual reasoning (e.g., for baseline Qwen models).

### Quick debug run (subset of data)
Add `--dummy True` to any training command to use only the first 1000 samples.

## Architecture Overview

LantErn extends Qwen2.5-VL to produce interleaved text and **Latent Visual Reasoning (LVR)** tokens. The key idea: instead of always describing what it sees in text, the model can emit compressed visual embeddings (`<|lvr_start|>...<|lvr_end|>`) during reasoning.

### How Latent Tokens Work

**Training:**
1. Bounding-box image crops are passed through the vision encoder.
2. `apply_latent_compression()` (`src/models/utils.py`) averages visual features into exactly `latent_size` embeddings.
3. These compressed embeddings replace `<|lvr_sep|>` token positions in the input sequence.
4. Loss = `γ * CE_loss(text tokens) + MSE_loss(latent positions)`.

**Generation:**
- The model's `generate()` is monkey-patched (via `custom_generate` kwarg, see `src/models/qwen2_5VL/forward.py`) to call the custom loop in `src/lantern_generate/generate.py`.
- When `<|lvr_start|>` is predicted, the loop switches to latent mode: it feeds back hidden states as embeddings for the next `latent_size` steps, then resumes text generation after `<|lvr_end|>`.

### Key Files

| File | Role |
|------|------|
| `src/models/__init__.py` | `load_model()` — loads Qwen2.5-VL and monkey-patches its forward method |
| `src/models/qwen2_5VL/forward.py` | Patched forward that supports mixed text/latent modality |
| `src/lantern_generate/generate.py` | Custom token-by-token generation loop for LVR |
| `src/models/utils.py` | `apply_latent_compression()` — vision features → `latent_size` averaged tokens |
| `src/train/__init__.py` | `set_latent_tokens()`, `configure_vision_tower()`, `configure_llm()` |
| `src/params.py` | All config dataclasses: `ModelParams`, `TrainingParams`, `SFTDataParams`, `GRPOArguments`, `RLDataParams` |
| `src/trainer/sft_trainer.py` | `LantErnSFTrainer` (HF `Trainer` subclass with dual CE+MSE loss) |
| `src/trainer/grpo_trainer.py` | `LantErnGRPOTrainer` (TRL `GRPOTrainer` subclass) |
| `src/rl/rewards.py` | GRPO reward functions via `REWARD_REGISTRY`; add rewards with `@register_reward("name")` |
| `evals/__init__.py` | `run_batch_inference()` — shared inference helper used by all eval scripts |

### Model Loading

`load_model()` in `src/models/__init__.py`:
- Patches `Qwen2_5_VLForConditionalGeneration.forward` at import time with the LantErn-aware forward.
- Accepts either a HuggingFace Hub ID or a local checkpoint path.
- Automatically detects local vs. remote via `os.path.isdir()`.

### Special Tokens

- `<|lvr_start|>`: begins a latent reasoning block
- `<|lvr_sep|>`: placeholder; replaced by compressed visual embeddings during training; used as padding during generation latent steps
- `<|lvr_end|>`: ends a latent reasoning block

These are registered by `set_latent_tokens()` in `src/train/__init__.py` and stored in `model.config` as `lvr_start_id`, `lvr_sep_id`, `lvr_end_id`.

### Two-Stage Training

1. **SFT** (`src/train/train_sft.py`): Supervised fine-tuning on VisCoT-style data with visual reasoning traces. Uses `LantErnSFTrainer` which computes CE + MSE loss.
2. **GRPO** (`src/train/train_grpo.py`): RL fine-tuning from an SFT checkpoint. Uses `LantErnGRPOTrainer`. Reward functions are registered in `src/rl/rewards.py`; configured via `reward_names` and `reward_weights` in `GRPOArguments`.

### Data Format (SFT)

```json
{
  "question": "...",
  "img_path": "/path/to/image.jpg",
  "bboxs": [[x1, y1, x2, y2], ...],
  "reasoning_traces": {
    "pre_visual_text_think": "...",
    "post_visual_latent_reasoning": ["...", ...],
    "text_think": "...",
    "answer": "..."
  }
}
```

`len(bboxs)` must equal `len(post_visual_latent_reasoning)`.

## Notes

- `attn_implementation` is forced to `"eager"` in `load_model()` (flash_attention_2 is commented out).
- `latent_size=-1` means dynamic size (same number of tokens as visual features); any positive int fixes the compression target.
