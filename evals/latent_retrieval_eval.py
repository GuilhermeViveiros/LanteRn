"""
Latent Retrieval Evaluation for LantErn.

Measures whether the model's self-predicted latent embeddings can retrieve the
correct GT latent from a gallery built from the full test set.

This distinguishes representation failure (predicted latents are noise, R@k ≈ random)
from interface/routing failure (latents encode signal but the LM ignores them, R@k >> random
yet own-latents ≈ no-LVR on VisCoT).

Metrics: R@1, R@5, R@10, Mean Rank, MRR
Oracle check: GT latents as queries → must give R@1 = 1.0 to validate the setup.

Usage:
    python -m evals.latent_retrieval_eval \\
        --model_ref /path/to/checkpoint \\
        --output_dir results/latent_retrieval \\
        --batch_size 4

Submit via srun:
    srun -A jureap126 -p booster --nodes=1 --time=06:00:00 \\
         --cpus-per-task=70 --gres=gpu:4 \\
      bash -c 'HF_HUB_OFFLINE=1 python -u -m evals.latent_retrieval_eval \\
               --model_ref /path/to/checkpoint \\
               --output_dir results/latent_retrieval \\
               > results/latent_retrieval/run.log 2>&1'
"""

import json
import os
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from evals import get_gt_latent_values, run_batch_inference
from evals.eval import MCDataset, collate_fn_mc
from src.models import load_model
from src.models.utils import apply_latent_compression


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_gt_vec(model, cropped_images, processor):
    """GT bbox crop → vision encoder → apply_latent_compression → flatten → (batch, latent_size*D)."""
    pv, grid_thw = get_gt_latent_values(cropped_images, processor)
    pv       = pv.to(model.device)
    grid_thw = grid_thw.to(model.device)
    embeds = apply_latent_compression(
        model,
        latent_values=pv,
        latent_grid_thw=grid_thw,
        latent_size=model.config.latent_size,
    )  # (batch, latent_size, D)
    return embeds.mean(dim=1)  # (batch, D)


def pool_predicted_latents(latent_embeds_list):
    """
    latent_embeds_list: List[List[Tensor(D,)]] from LantErnGenerateOutput.latent_embeds
    Returns: list of (latent_size*D,) tensors or None per sample, and bool valid mask.
    """
    vecs, valid = [], []
    for embeds in latent_embeds_list:
        if embeds and len(embeds) > 0:
            tensor = torch.stack(embeds).mean(dim=0).float()  # (D,)
            if tensor.ndim != 1:
                print(f"Latent vector shape mismatch: got shape {tuple(tensor.shape)}, expected 1D")
                vecs.append(None)
                valid.append(False)
            else:        
                vecs.append(tensor)
                valid.append(True)
        else:
            vecs.append(None)
            valid.append(False)
    return vecs, valid


def cosine_sim_matrix(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """(N, D) × (M, D) → (N, M) cosine similarity matrix."""
    return F.normalize(queries, dim=-1) @ F.normalize(gallery, dim=-1).T


def retrieval_metrics(sim_matrix: torch.Tensor, correct_indices: list) -> dict:
    """
    Compute R@k, mean rank, MRR.
    correct_indices[i] = index in gallery that is the correct match for query i.
    """
    ranks = []
    for i, gi in enumerate(correct_indices):
        row  = sim_matrix[i]
        rank = int((row > row[gi]).sum().item()) + 1  # 1-indexed
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "R@1":       float((ranks <= 1).mean()),
        "R@5":       float((ranks <= 5).mean()),
        "R@10":      float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "mrr":       float((1.0 / ranks).mean()),
        "n_queries":     int(len(ranks)),
        "gallery_size":  int(sim_matrix.shape[1]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_ref",  type=str,
    #                     default="/e/project1/jureap126/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062")
    #                     default="/e/project1/jureap126/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1500")
    #                     default="/e/project1/jureap126/gviveiros/lantern/checkpoints/qwen_7b_sft_mse_lt_8_lambda_0.1/checkpoint-708"
    
    parser.add_argument("--model_ref",  type=str,
                        default="/e/project1/jureap126/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062")
    # grpo_lt_8_lambda_0.1/checkpoint-1500
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps",  type=int, default=300,
                        help="Max dataloader steps (300 × batch_size ≈ full VisCoT test set)")
    parser.add_argument("--dummy",      action="store_true",
                        help="Run on 5 samples only for quick sanity check")
    parser.add_argument("--output_dir", type=str, default="results/latent_retrieval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(
        model_ref=args.model_ref,
        device_map="cuda",
        compute_dtype=torch.bfloat16,
        use_cache=True,
    )
    model.eval()
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=False)
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id   = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    model.config.lvr_sep_id   = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    processor.tokenizer.padding_side = "left"
    print(f"Model loaded: {args.model_ref}")
    print(f"Latent size:  {model.config.latent_size}")
    print(f"LVR token IDs: start={model.config.lvr_start_id}, sep={model.config.lvr_sep_id}, end={model.config.lvr_end_id}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset    = MCDataset(datasets=["viscot"], use_lvr=True)
    if args.dummy:
        dataset.data = dataset.data[:5]
        print("*** DUMMY MODE: 5 samples only ***")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_fn_mc, processor=processor),
    )
    print(f"Dataset size: {len(dataset)}")

    # ── Collect GT gallery and predicted latents ───────────────────────────────
    gt_vecs   = []   # (D,) per sample — full gallery
    pred_vecs = []   # (D,) or None per sample
    valid     = []   # True if model predicted latents for this sample

    for step, (batch, labels, options, bboxs, cropped_images) in tqdm(
        enumerate(dataloader), total=min(args.max_steps, len(dataloader)), desc="Collecting"
    ):
        if step >= args.max_steps:
            break

        # GT gallery vectors
        gt = compute_gt_vec(model, cropped_images, processor)  # (B, D)
        gt_vecs.extend(gt.unbind(0))

        # Predicted latents — own generation, no GT injection
        output = run_batch_inference(
            model, batch,
            use_lvr=True, use_gt=False, return_dict=True, output_attentions=False,
        )

        #if args.dummy:
        decoded = processor.tokenizer.batch_decode(
            [seq[len(batch["input_ids"][0]):] for seq in output.sequences],
            skip_special_tokens=True
        )
        print("Decoded output (skipping input):")
        for i, d in enumerate(decoded):
            print(f"  Sample {i}: {d}")

        if output.latent_embeds is not None:
            pvecs, pvalid = pool_predicted_latents(output.latent_embeds)
        else:
            pvecs  = [None] * len(bboxs)
            pvalid = [False] * len(bboxs)
        #import pdb; pdb.set_trace()
        pred_vecs.extend(pvecs)
        valid.extend(pvalid)

    N = len(gt_vecs)
    n_valid = sum(valid)
    print(f"\nTotal samples:            {N}")
    print(f"Samples with own latents: {n_valid} ({n_valid/N:.1%})")

    # ── Build gallery tensor ──────────────────────────────────────────────────
    gallery = torch.stack(gt_vecs).float().cpu()  # (N, D)

    # ── Oracle: GT as queries → R@1 must be 1.0 ──────────────────────────────
    oracle_sim     = cosine_sim_matrix(gallery, gallery)
    oracle_metrics = retrieval_metrics(oracle_sim, list(range(N)))
    print(f"\nOracle (GT → GT gallery):")
    for k, v in oracle_metrics.items():
        print(f"  {k}: {v}")

    # ── Predicted latents retrieval ───────────────────────────────────────────
    valid_idx = [i for i, v in enumerate(valid) if v]
    valid_pct = 100.0 * len(valid_idx) / len(valid) if valid else 0.0
    print(f"Valid percentage: {valid_pct:.2f}% ({len(valid_idx)}/{len(valid)})")
    if valid_idx:
        queries    = torch.stack([pred_vecs[i] for i in valid_idx]).cpu()  # (n_valid, D)
        pred_sim   = cosine_sim_matrix(queries, gallery)                   # (n_valid, N)
        pred_metrics = retrieval_metrics(pred_sim, valid_idx)
        random_r1    = 1.0 / N
        print(f"\nPredicted latents retrieval:")
        for k, v in pred_metrics.items():
            print(f"  {k}: {v}")
        print(f"\nRandom baseline R@1: {random_r1:.6f} (1/{N})")
        print(f"Lift over random:    {pred_metrics['R@1'] / random_r1:.1f}x")

        # ── Paired MSE and cosine similarity (pred vs own GT) ─────────────────
        gt_for_valid = gallery[valid_idx]  # (n_valid, D)
        mse_per_sample = ((queries - gt_for_valid) ** 2).mean(dim=-1)  # (n_valid,)
        cos_per_sample = F.cosine_similarity(queries, gt_for_valid, dim=-1)  # (n_valid,)
        paired_mse = mse_per_sample.mean().item()
        paired_cos = cos_per_sample.mean().item()
        print(f"\nPaired pred↔GT (own sample only):")
        print(f"  MSE (mean):    {paired_mse:.6f}")
        print(f"  MSE (std):     {mse_per_sample.std().item():.6f}")
        print(f"  Cosine (mean): {paired_cos:.6f}")
        print(f"  Cosine (std):  {cos_per_sample.std().item():.6f}")
        pred_metrics["paired_mse"] = paired_mse
        pred_metrics["paired_cosine"] = paired_cos
    else:
        pred_metrics = {}
        paired_mse = None
        print("\nNo samples predicted latents — model never emitted <|lvr_start|>")

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt_name = "_".join(args.model_ref.rstrip("/").split("/")[-2:])
    results = {
        "model":          args.model_ref,
        "oracle":         oracle_metrics,
        "predicted":      pred_metrics,
        "random_r1":      1.0 / N,
        "n_total":        N,
        "n_with_latents": n_valid,
        "paired_mse":     paired_mse if valid_idx else None,
    }
    out_path = os.path.join(args.output_dir, f"{ckpt_name}_retrieval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
