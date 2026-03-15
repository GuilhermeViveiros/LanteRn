"""
Systematic latent token ablation on the VisCoT MC benchmark.

Loads the model ONCE and runs all conditions sequentially, saving per-sample
predictions for flip-set analysis.

Conditions:
  no_lvr       — standard generation, no latent tokens
  own_latents  — LVR generation, model predicts its own latents
  gt_latents   — LVR generation, GT bbox crop injected as latent
  random_bbox  — LVR generation, random-location crop injected as latent
  zeros        — LVR generation, zero embeddings injected (bypasses vision encoder)

Output: results/ablation/<run_name>/
  predictions.json  — per-sample predictions for all conditions
  flip_analysis.json — flip sets between condition pairs

Submit:
    srun --partition=h100 --qos=gpu-h100 --job-name=ablation --time=06:00:00 \
         --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \
      bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               python -u -m evals.ablation_eval \
               > results/ev_ablation.log 2>&1'
"""

import json
import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader

from src.models import load_model
from src.models.utils import apply_latent_compression
from src.lantern_generate.generate import generate as lantern_generate
from src.utils import extract_mc_answer, center_and_crop_image
from evals import get_gt_latent_values, random_crop_bbox
from evals.eval import MCDataset, collate_fn_mc

CHECKPOINT = "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/"
MAX_STEPS  = 300

ALL_CONDITIONS = ["no_lvr", "own_latents", "gt_latents", "random_bbox", "zeros"]


# ---------------------------------------------------------------------------
# Per-condition latent builder
# ---------------------------------------------------------------------------

def build_latent_embeds(condition, model, processor, cropped_images, batch_size):
    """Return gt_latent_embeds tensor or None for the given condition."""
    if condition in ("no_lvr", "own_latents"):
        return None
    if condition == "zeros":
        return torch.zeros(
            batch_size,
            model.config.latent_size,
            model.config.hidden_size,
            dtype=next(model.parameters()).dtype,
            device=model.device,
        )
    if condition in ("gt_latents", "random_bbox"):
        gt_latent_values, gt_latent_grid_thw = get_gt_latent_values(cropped_images, processor)
        return apply_latent_compression(
            model,
            latent_values=gt_latent_values.to(model.device),
            latent_grid_thw=gt_latent_grid_thw.to(model.device),
            latent_size=model.config.latent_size,
        )
    raise ValueError(f"Unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Single-condition inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_condition(condition, model, processor, dataloader):
    """Run one full pass over the dataloader and return per-sample results."""
    results = []  # list of dicts: idx, prediction, label, correct

    pbar = tqdm(enumerate(dataloader), total=min(MAX_STEPS, len(dataloader)),
                desc=f"[{condition}]")

    for step, (batch, labels, options, bboxs, cropped_images) in pbar:
        if step >= MAX_STEPS:
            break

        # Build latents — cropped_images already reflect the condition's crop strategy
        # (GT crops from loader, random crops from loader_rbox)
        gt_latent_embeds = build_latent_embeds(
            condition, model, processor, cropped_images, len(labels)
        )

        batch = batch.to(model.device)

        use_lvr = condition != "no_lvr"
        generated_ids = model.generate(
            **batch,
            max_new_tokens=526,
            do_sample=False,
            custom_generate=partial(
                lantern_generate,
                gt_latent_embeds=gt_latent_embeds,
            ) if use_lvr else None,
            use_cache=True,
            return_dict_in_generate=False,
            output_attentions=False,
        )

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(batch.input_ids, generated_ids)
        ]
        decoded = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        answers = [extract_mc_answer(x, opts) for x, opts in zip(decoded, options)]

        for i, (ans, label) in enumerate(zip(answers, labels)):
            results.append({
                "step": step,
                "prediction": ans,
                "label": label,
                "correct": ans == label,
            })

        acc = np.mean([r["correct"] for r in results])
        pbar.set_description(f"[{condition}] acc={acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Flip analysis
# ---------------------------------------------------------------------------

def flip_analysis(all_results):
    """For each pair of conditions, compute flip sets and stats."""
    conditions = list(all_results.keys())
    flips = {}

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            r1 = all_results[c1]
            r2 = all_results[c2]
            n = min(len(r1), len(r2))

            # c1 correct, c2 wrong
            c1_wins = [j for j in range(n) if r1[j]["correct"] and not r2[j]["correct"]]
            # c2 correct, c1 wrong
            c2_wins = [j for j in range(n) if r2[j]["correct"] and not r1[j]["correct"]]

            flips[f"{c1}_vs_{c2}"] = {
                f"{c1}_correct_{c2}_wrong": len(c1_wins),
                f"{c2}_correct_{c1}_wrong": len(c2_wins),
                "total_disagreements": len(c1_wins) + len(c2_wins),
                "sample_indices": {
                    f"{c1}_wins": c1_wins[:50],  # cap for readability
                    f"{c2}_wins": c2_wins[:50],
                }
            }

    return flips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ref",   type=str, default=CHECKPOINT)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--output_dir",  type=str, default="results/ablation")
    parser.add_argument("--conditions",  type=str, nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Which conditions to run (default: all)")
    args = parser.parse_args()

    run_name = "_".join(args.model_ref.rstrip("/").split("/")[-2:])
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model:      {args.model_ref}")
    print(f"Conditions: {args.conditions}")
    print(f"Output:     {out_dir}")

    # Load model once
    model, processor = load_model(
        model_ref=args.model_ref,
        device_map="cuda",
        compute_dtype=torch.bfloat16,
        use_cache=True,
    )
    model.eval()

    # Dataset — use random_bbox mode from MCDataset for the random_bbox condition;
    # for all others we use none (GT crops available in cropped_images either way)
    dataset = MCDataset(datasets=["viscot"], use_lvr=True, bbox_ablation=None)
    dataset_rbox = MCDataset(datasets=["viscot"], use_lvr=True, bbox_ablation="random")

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn_mc, processor=processor),
        )

    loader     = make_loader(dataset)
    loader_rbox = make_loader(dataset_rbox)

    # Run all conditions
    all_results = {}
    for condition in args.conditions:
        dl = loader_rbox if condition == "random_bbox" else loader
        all_results[condition] = run_condition(condition, model, processor, dl)

        acc = np.mean([r["correct"] for r in all_results[condition]])
        print(f"\n{condition}: accuracy = {acc:.4f}")

    # Accuracy summary
    print("\n=== Accuracy Summary ===")
    for cond, results in all_results.items():
        acc = np.mean([r["correct"] for r in results])
        print(f"  {cond:20s}: {acc:.4f}")

    # Flip analysis
    flips = flip_analysis(all_results)
    print("\n=== Flip Analysis ===")
    for pair, stats in flips.items():
        print(f"  {pair}: {stats['total_disagreements']} disagreements")

    # Save
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(out_dir, "flip_analysis.json"), "w") as f:
        json.dump(flips, f, indent=2)

    print(f"\nSaved to {out_dir}/")
