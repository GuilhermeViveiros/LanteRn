"""
LantErn inference template.

Supports all three released checkpoints:
  - AGViveiros/LanteRn-3B-SFT        (VisCoT supervised fine-tuning)
  - AGViveiros/LanteRn-3B-RL         (GRPO reinforcement learning)
  - AGViveiros/LanteRn-3B-Tetris     (Tetris analogy SFT)

Usage:
    python examples/inference.py --model AGViveiros/LanteRn-3B-Tetris --image path/to/img.jpg
"""

import argparse

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.lantern_generate.generate import generate as lantern_generate
from src.models import load_model


def build_inputs(processor, image: Image.Image, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")


@torch.no_grad()
def run(
    model_ref: str,
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
):
    # ── Load ──────────────────────────────────────────────────────────────────
    model, processor = load_model(model_ref, compute_dtype=torch.bfloat16, use_cache=True)
    model.eval().to(device)
    processor.tokenizer.padding_side = "left"

    # ── Prepare inputs ────────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(processor, image, question).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # ── Generate ──────────────────────────────────────────────────────────────
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        custom_generate=lantern_generate,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=False,
    )

    # ── Decode ────────────────────────────────────────────────────────────────
    generated_ids = output.sequences[0][prompt_len:]
    text = processor.decode(generated_ids, skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(tok, "")
    text = text.strip()

    used_lvr = hasattr(output, "latent_embeds") and output.latent_embeds is not None
    n_latent_blocks = int((generated_ids == model.config.lvr_start_id).sum())

    print(f"\n{'─'*60}")
    print(f"Model      : {model_ref}")
    print(f"LVR active : {used_lvr}  (latent blocks: {n_latent_blocks})")
    print(f"{'─'*60}")
    print(text)
    print(f"{'─'*60}\n")

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _default_img = "/e/project1/jureap131/gviveiros/lantern/analogy_data/images/000000.png"
    _default_q = (
        "Image (A) is to image (B) as image (C) is to which of the following options?\n"
        "Options:\n(a) Option a\n(b) Option b\n(c) Option c\n(d) Option d"
    )
    parser.add_argument("--model", default="AGViveiros/LanteRn-3B-Tetris", help="HF repo ID or local checkpoint path")
    parser.add_argument("--image", default=_default_img, help="Path to input image")
    parser.add_argument("--question", default=_default_q)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run(
        model_ref=args.model,
        image_path=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
