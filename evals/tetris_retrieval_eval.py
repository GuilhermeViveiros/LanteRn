"""
Latent Retrieval Evaluation for the Tetris analogy task.

For each sample the model generates its own latent embedding (use_gt=False).
We then measure whether that predicted embedding can retrieve the correct GT
latent from a gallery built over the full split.

Runs on both eval (in-distribution) and held_out (OOD) splits in one pass.

Metrics: R@1, R@5, R@10, Mean Rank, MRR, paired cosine similarity.
Oracle check: GT → GT gallery must give R@1 = 1.0.

Usage:
    python -m evals.tetris_retrieval_eval \\
        --model_ref /path/to/checkpoint \\
        --eval_path /path/to/analogy_data/eval.json \\
        --held_out_path /path/to/analogy_data/held_out/held_out.json \\
        --output_dir results/tetris_retrieval \\
        --batch_size 8

    Add --grayscale for gray-trained checkpoints.
"""

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from evals import run_batch_inference
from src.models import load_model
from src.models.utils import apply_latent_compression
from src.train import set_latent_tokens
from src.datasets.sft_tetris_data import SFTTetrisDataset, collate_fn_generate


# ── Helpers ───────────────────────────────────────────────────────────────────

def pool_predicted_latents(latent_embeds_list):
    """
    latent_embeds_list: List[List[Tensor(D,)]] from LantErnGenerateOutput.latent_embeds
    Returns: list of (D,) tensors or None per sample, and bool valid mask.
    """
    vecs, valid = [], []
    for embeds in latent_embeds_list:
        if embeds and len(embeds) > 0:
            tensor = torch.stack(embeds).mean(dim=0).float()
            vecs.append(tensor if tensor.ndim == 1 else None)
            valid.append(tensor.ndim == 1)
        else:
            vecs.append(None)
            valid.append(False)
    return vecs, valid


def cosine_sim_matrix(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    return F.normalize(queries, dim=-1) @ F.normalize(gallery, dim=-1).T


def retrieval_metrics(sim_matrix: torch.Tensor, correct_indices: list) -> dict:
    ranks = []
    for i, gi in enumerate(correct_indices):
        row  = sim_matrix[i]
        rank = int((row > row[gi]).sum().item()) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "R@1":        float((ranks <= 1).mean()),
        "R@5":        float((ranks <= 5).mean()),
        "R@10":       float((ranks <= 10).mean()),
        "mean_rank":  float(ranks.mean()),
        "mrr":        float((1.0 / ranks).mean()),
        "n_queries":  int(len(ranks)),
        "gallery_size": int(sim_matrix.shape[1]),
    }


# ── Per-split eval ────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_split(model, processor, dataset, batch_size: int, split_name: str) -> dict:
    collate = partial(collate_fn_generate, processor=processor)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    processor.tokenizer.padding_side = "left"

    gt_vecs   = []
    pred_vecs = []
    valid     = []

    for inputs, _ in tqdm(loader, desc=f"{split_name} collecting"):
        # ── GT embeddings ──────────────────────────────────────────────────
        latent_values  = inputs["latent_values"].to(model.device)
        latent_grid_thw = inputs["latent_grid_thw"].to(model.device)
        gt_embeds = apply_latent_compression(
            model,
            latent_values=latent_values,
            latent_grid_thw=latent_grid_thw,
            latent_size=model.config.latent_size,
        )  # (B, latent_size, D)
        gt_vecs.extend(gt_embeds.mean(dim=1).float().cpu().unbind(0))

        # ── Predicted latents (own generation) ────────────────────────────
        output = run_batch_inference(model, inputs, use_lvr=True, use_gt=False,
                                     return_dict=True, output_attentions=False)
        if output.latent_embeds is not None:
            pvecs, pvalid = pool_predicted_latents(output.latent_embeds)
        else:
            pvecs  = [None] * batch_size
            pvalid = [False] * batch_size
        pred_vecs.extend(pvecs)
        valid.extend(pvalid)

    processor.tokenizer.padding_side = "right"

    N       = len(gt_vecs)
    n_valid = sum(valid)
    print(f"\n[{split_name}] {N} samples, {n_valid} with own latents ({n_valid/N:.1%})")

    gallery = torch.stack(gt_vecs).cpu()

    # Oracle check
    oracle_sim     = cosine_sim_matrix(gallery, gallery)
    oracle_metrics = retrieval_metrics(oracle_sim, list(range(N)))
    print(f"  Oracle R@1: {oracle_metrics['R@1']:.3f}  (must be 1.0)")

    if n_valid == 0:
        print("  No predicted latents — model never emitted <|lvr_start|>")
        return {"oracle": oracle_metrics, "predicted": {}, "n_total": N, "n_valid": n_valid}

    valid_idx  = [i for i, v in enumerate(valid) if v]
    queries    = torch.stack([pred_vecs[i] for i in valid_idx]).cpu()
    pred_sim   = cosine_sim_matrix(queries, gallery)
    pred_metrics = retrieval_metrics(pred_sim, valid_idx)

    # Paired cosine between predicted and own GT
    gt_for_valid   = gallery[valid_idx]
    cos_per_sample = F.cosine_similarity(queries, gt_for_valid, dim=-1)
    mse_per_sample = ((queries - gt_for_valid) ** 2).mean(dim=-1)
    pred_metrics["paired_cosine_mean"] = float(cos_per_sample.mean())
    pred_metrics["paired_cosine_std"]  = float(cos_per_sample.std())
    pred_metrics["paired_mse_mean"]    = float(mse_per_sample.mean())

    random_r1 = 1.0 / N
    print(f"  Pred  R@1: {pred_metrics['R@1']:.3f}  "
          f"R@5: {pred_metrics['R@5']:.3f}  "
          f"R@10: {pred_metrics['R@10']:.3f}  "
          f"(random R@1={random_r1:.4f}, lift={pred_metrics['R@1']/random_r1:.1f}x)")
    print(f"  Paired cosine: {pred_metrics['paired_cosine_mean']:.4f} ± {pred_metrics['paired_cosine_std']:.4f}")

    return {
        "oracle":    oracle_metrics,
        "predicted": pred_metrics,
        "n_total":   N,
        "n_valid":   n_valid,
        "random_r1": random_r1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ref",      required=True)
    parser.add_argument("--eval_path",      required=True)
    parser.add_argument("--held_out_path",  required=True)
    parser.add_argument("--output_dir",     default="results/tetris_retrieval")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--latent_size",    type=int, default=8)
    parser.add_argument("--grayscale",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model(args.model_ref, compute_dtype=torch.bfloat16,
                                  use_cache=True, attn_implementation="flash_attention_2")
    set_latent_tokens(processor, model, args.latent_size)
    model.eval().cuda()

    eval_ds     = SFTTetrisDataset(args.eval_path,    processor, use_lvr=True,
                                   grayscale_intermediate=args.grayscale)
    held_out_ds = SFTTetrisDataset(args.held_out_path, processor, use_lvr=True,
                                   grayscale_intermediate=args.grayscale)
    print(f"Eval samples    : {len(eval_ds)}")
    print(f"Held-out samples: {len(held_out_ds)}  "
          f"(shapes: {sorted({s['shape_C_name'] for s in held_out_ds.dataset})})")

    eval_results    = eval_split(model, processor, eval_ds,     args.batch_size, "eval")
    held_results    = eval_split(model, processor, held_out_ds, args.batch_size, "held_out")

    ckpt_name = "_".join(args.model_ref.rstrip("/").split("/")[-2:])
    out = {
        "model":    args.model_ref,
        "eval":     eval_results,
        "held_out": held_results,
    }
    out_path = os.path.join(args.output_dir, f"{ckpt_name}_retrieval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'split':<10} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'paired_cos':>12}")
    print("-" * 60)
    for split_name, res in [("eval", eval_results), ("held_out", held_results)]:
        p = res.get("predicted", {})
        if p:
            print(f"{split_name:<10} {p['R@1']:>6.3f} {p['R@5']:>6.3f} {p['R@10']:>6.3f} "
                  f"{p.get('paired_cosine_mean', float('nan')):>12.4f}")
        else:
            print(f"{split_name:<10} {'N/A':>6}")
    print("=" * 60)


if __name__ == "__main__":
    main()
