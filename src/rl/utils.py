import re
from typing import Any, Dict, Optional

from src.rl.prompt import make_prompt_messages


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL | re.IGNORECASE)
ANSWER_SPAN_RE = re.compile(r"<answer>.*?</answer>", flags=re.DOTALL | re.IGNORECASE)


# ----------------------------
# 1) Token setup (LVR tokens)
# ----------------------------
def ensure_lvr_tokens(model, processor) -> None:
    tok = processor.tokenizer
    to_add = []
    for t in ["<|lvr_start|>", "<|lvr_sep|>", "<|lvr_end|>"]:
        if tok.convert_tokens_to_ids(t) == tok.unk_token_id:
            to_add.append(t)

    if to_add:
        tok.add_tokens(to_add, special_tokens=True)
        model.resize_token_embeddings(len(tok))

    model.config.lvr_start_id = tok.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_sep_id = tok.convert_tokens_to_ids("<|lvr_sep|>")
    model.config.lvr_end_id = tok.convert_tokens_to_ids("<|lvr_end|>")


# ----------------------------
# 2) Text manipulation functions
# ----------------------------


def extract_last_answer_from_text(text: str) -> str:
    """
    Reason:
      - We only use the last <answer>...</answer>. TODO: maybe change this?

    Assumptions:
      - If the model didn't produce tags => incorrect => return "".
    """
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def convert_example(example: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Reason:
      - Convert your JSON example into GRPO-friendly columns:
        prompt (messages) + images (list) + ground_truth.

    Assumptions:
      - JSON has "image" path and "conversations".
      - One human and one gpt/assistant message exist.
      - The gpt message has <answer> tags.
    """
    conv = example["conversations"]
    user_msg = next(m["value"] for m in conv if m.get("from") == "human")
    asst_msg = next(m["value"] for m in conv if m.get("from") in {"gpt", "assistant"})

    gt = extract_last_answer_from_text(asst_msg)

    return {
        "prompt": make_prompt_messages(user_msg, system_prompt),
        "images": example["images"],  # list of PIL after cast
        "ground_truth": gt,
        "id": str(example.get("id", "")),
        "dataset": str(example.get("dataset", "")),
    }


# ----------------------------
# 3) Model configuration (freeze vision tower)
# ----------------------------


def configure_generation_cache(model):
    """
    Reason:
      - GRPO spends a lot of time in model.generate().
      - KV-cache reduces compute for decoding.

    Assumptions:
      - NOT using gradient checkpointing for the generation path.
      - If gradient checkpointing is enabled, Transformers may force use_cache=False.
    """
    # If previously enabled gradient checkpointing anywhere, disable it for speed.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        print("[cache] disabled gradient checkpointing for generation speed")

    # Core switches
    model.config.use_cache = True

    # Also set generation_config
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = True

    # For some models, language_model holds the actual decoder config
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = True

    print("[cache] model.config.use_cache =", model.config.use_cache)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        print(
            "[cache] model.generation_config.use_cache =",
            model.generation_config.use_cache,
        )


def freeze_vision(model, freeze_projector: bool = False):
    """
    Reason:
      - In VLM RL, vision backward is expensive and often unnecessary at first.
      - Freezing vision reduces memory and compute.

    What it does:
      - Finds common vision module attributes and freezes their params.
      - Optionally freezes multimodal projector too.
    """

    frozen_any = False

    def _freeze_module(mod, tag):
        nonlocal frozen_any
        if mod is None:
            return
        for p in mod.parameters():
            p.requires_grad = False
        frozen_any = True
        print(f"[freeze] froze {tag}")

    # Try direct attributes
    for name in ["visual", "vision_tower", "vision_model", "image_tower"]:
        if hasattr(model, name):
            _freeze_module(getattr(model, name), f"model.{name}")

    # Try nested under model.model (HF often nests)
    if hasattr(model, "model"):
        for name in ["visual", "vision_tower", "vision_model", "image_tower"]:
            if hasattr(model.model, name):
                _freeze_module(getattr(model.model, name), f"model.model.{name}")

    # Qwen2.5-VL often has image feature encoder reachable via model.get_image_features,
    # but the actual weights live in vision modules above.

    # Optionally freeze projector (varies by implementation)
    if freeze_projector:
        for proj_name in [
            "mm_projector",
            "multi_modal_projector",
            "visual_projection",
            "projector",
        ]:
            if hasattr(model, proj_name):
                _freeze_module(getattr(model, proj_name), f"model.{proj_name}")
            if hasattr(model, "model") and hasattr(model.model, proj_name):
                _freeze_module(
                    getattr(model.model, proj_name), f"model.model.{proj_name}"
                )

    # Print trainable parameter count (sanity check)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[freeze] trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    if not frozen_any:
        print(
            "[freeze] WARNING: did not find a recognized vision module to freeze. "
            "Print model modules to locate vision tower naming."
        )
