"""
Latent Retrieval Evaluation for LantErn.

Measures whether the model's self-predicted latent embeddings can retrieve the
correct GT latent from a gallery built from the full test set.

This distinguishes representation failure (predicted latents are noise, R@k ≈ random)
from interface/routing failure (latents encode signal but the LM ignores them, R@k >> random
yet own-latents ≈ no-LVR on VisCoT).

Metrics: R@1, R@5, R@10, Mean Rank, MRR
Oracle check: GT latents as queries → must give R@1 = 1.0 to validate the setup.

Output layout:
    output_dir/
      crops/                        # bbox crop PNG per sample
      latents/                      # .pt tensors (latent_size, D) before mean-pooling
      results.jsonl                 # one record per sample
      {checkpoint}_retrieval.json   # aggregate metrics

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

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from evals import get_gt_latent_values, run_batch_inference
from evals.eval import MCDataset, collate_fn_mc
from src.models import load_model
from src.models.utils import apply_latent_compression

# ── Helpers ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_gt_tensors(model, cropped_images, processor):
    """
    GT bbox crop → vision encoder → apply_latent_compression.
    Returns:
        mean_vecs:   (batch, D)           — mean-pooled, used for retrieval
        raw_tensors: (batch, latent_size, D) — full, saved to disk
    """
    pv, grid_thw = get_gt_latent_values(cropped_images, processor)
    pv = pv.to(model.device)
    grid_thw = grid_thw.to(model.device)
    embeds = apply_latent_compression(
        model,
        latent_values=pv,
        latent_grid_thw=grid_thw,
        latent_size=model.config.latent_size,
    )  # (batch, latent_size, D)
    return embeds.mean(dim=1), embeds  # (batch, D), (batch, latent_size, D)


def pool_predicted_latents(latent_embeds_list):
    """
    latent_embeds_list: List[List[Tensor(D,)]] from LantErnGenerateOutput.latent_embeds
    Returns:
        mean_vecs:   list of (D,) tensors or None per sample
        raw_tensors: list of (latent_size, D) tensors or None per sample
        valid:       list of bool
    """
    mean_vecs, raw_tensors, valid = [], [], []
    for embeds in latent_embeds_list:
        if embeds and len(embeds) > 0:
            stacked = torch.stack(embeds).float()  # (latent_size, D)
            if stacked.ndim != 2:
                print(f"Latent tensor shape mismatch: got {tuple(stacked.shape)}, expected 2D")
                mean_vecs.append(None)
                raw_tensors.append(None)
                valid.append(False)
            else:
                mean_vecs.append(stacked.mean(dim=0))  # (D,)
                raw_tensors.append(stacked)  # (latent_size, D)
                valid.append(True)
        else:
            mean_vecs.append(None)
            raw_tensors.append(None)
            valid.append(False)
    return mean_vecs, raw_tensors, valid


def count_tokens(seq, input_len, lvr_start_id, lvr_end_id):
    """Count text and latent tokens in the new portion of an output sequence."""
    new_tokens = seq[input_len:].tolist()
    n_text, n_latent = 0, 0
    in_latent = False
    for tok in new_tokens:
        if tok == lvr_start_id:
            in_latent = True
        elif tok == lvr_end_id:
            in_latent = False
        elif in_latent:
            n_latent += 1
        else:
            n_text += 1
    return n_text, n_latent


def cosine_sim_matrix(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """(N, D) × (M, D) → (N, M) cosine similarity matrix."""
    return F.normalize(queries, dim=-1) @ F.normalize(gallery, dim=-1).T


def retrieval_metrics(sim_matrix: torch.Tensor, correct_indices: list) -> dict:
    """
    Compute R@k, mean rank, MRR, and per-query ranks.
    correct_indices[i] = index in gallery that is the correct match for query i.
    Returns metrics dict plus a 'ranks' list (1-indexed).
    """
    ranks = []
    for i, gi in enumerate(correct_indices):
        row = sim_matrix[i]
        rank = int((row > row[gi]).sum().item()) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
        "n_queries": int(len(ranks)),
        "gallery_size": int(sim_matrix.shape[1]),
        "_ranks": ranks.tolist(),  # per-query, not saved to final JSON
    }


def collate_fn_retrieval(samples, processor: AutoProcessor):
    """Wraps collate_fn_mc and also returns per-sample metadata."""
    metadata = []
    for s in samples:
        metadata.append(
            {
                "question": s.get("question", ""),
                "image_path": s.get("image", [None])[0],
                "gt_answer": s.get("label", None),
                "gt_rationale": s.get("reasoning_traces", None),
            }
        )
    inputs, labels, options, bboxs, cropped_images = collate_fn_mc(samples, processor)
    return inputs, labels, options, bboxs, cropped_images, metadata


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ref",
        type=str,
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_samples", type=int, default=300, help="Max number of samples to evaluate (independent of batch size)"
    )
    parser.add_argument("--dummy", action="store_true", help="Run on 5 samples only for quick sanity check")
    parser.add_argument("--output_dir", type=str, default="results/latent_retrieval")
    args = parser.parse_args()

    crops_dir = os.path.abspath(os.path.join(args.output_dir, "crops"))
    latents_dir = os.path.abspath(os.path.join(args.output_dir, "latents"))
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

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
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    model.config.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    processor.tokenizer.padding_side = "left"
    print(f"Model loaded: {args.model_ref}")
    print(f"Latent size:  {model.config.latent_size}")
    print(
        f"LVR token IDs: start={model.config.lvr_start_id}, sep={model.config.lvr_sep_id}, end={model.config.lvr_end_id}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MCDataset(datasets=["viscot"], use_lvr=True)
    max_samples = 5 if args.dummy else args.max_samples
    if max_samples < len(dataset.data):
        dataset.data = dataset.data[:max_samples]
    if args.dummy:
        print("*** DUMMY MODE: 5 samples only ***")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_retrieval, processor=processor),
    )
    print(f"Dataset size (capped): {len(dataset)}")

    # ── Collect GT gallery and predicted latents ───────────────────────────────
    gt_mean_vecs = []  # (D,) per sample — for retrieval
    gt_raw = []  # (latent_size, D) per sample — saved to disk
    pred_mean_vecs = []  # (D,) or None per sample
    pred_raw = []  # (latent_size, D) or None per sample
    valid = []  # True if model predicted latents
    records = []  # per-sample metadata accumulated during loop

    sample_idx = 0
    input_len = None  # set from first batch (same for all in the run)

    for step, (batch, labels, options, bboxs, cropped_images, metadata) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Collecting"
    ):
        # GT gallery tensors
        gt_mean, gt_raw_batch = compute_gt_tensors(model, cropped_images, processor)
        gt_mean_vecs.extend(gt_mean.cpu().unbind(0))
        gt_raw.extend(gt_raw_batch.cpu().unbind(0))

        # Predicted latents — own generation, no GT injection
        output = run_batch_inference(
            model,
            batch,
            use_lvr=True,
            use_gt=False,
            return_dict=True,
            output_attentions=False,
        )

        if input_len is None:
            input_len = batch["input_ids"].shape[1]

        decoded = processor.tokenizer.batch_decode(
            [seq[input_len:] for seq in output.sequences],
            skip_special_tokens=True,
        )
        print("Decoded output (skipping input):")
        for i, d in enumerate(decoded):
            print(f"  Sample {i}: {d}")

        if output.latent_embeds is not None:
            p_mean, p_raw, p_valid = pool_predicted_latents(output.latent_embeds)
        else:
            B = len(bboxs)
            p_mean, p_raw, p_valid = [None] * B, [None] * B, [False] * B

        pred_mean_vecs.extend(p_mean)
        pred_raw.extend(p_raw)
        valid.extend(p_valid)

        # Per-sample artifacts and record stubs
        for i in range(len(metadata)):
            sid = sample_idx + i

            # Save bbox crop image
            crop_img = cropped_images[i][0]
            crop_path = os.path.join(crops_dir, f"sample_{sid}_crop.png")
            if isinstance(crop_img, Image.Image):
                crop_img.save(crop_path)
            else:
                crop_path = str(crop_img)

            # Save GT latent tensor
            oracle_latent_path = os.path.abspath(os.path.join(latents_dir, f"sample_{sid}_oracle.pt"))
            torch.save(gt_raw_batch[i].cpu(), oracle_latent_path)

            # Save predicted latent tensor (if valid)
            pred_latent_path = None
            if p_valid[i] and p_raw[i] is not None:
                pred_latent_path = os.path.abspath(os.path.join(latents_dir, f"sample_{sid}_pred.pt"))
                torch.save(p_raw[i].cpu(), pred_latent_path)

            # Count text / latent tokens
            seq = output.sequences[i].cpu()
            n_text, n_latent = count_tokens(
                seq,
                input_len,
                model.config.lvr_start_id,
                model.config.lvr_end_id,
            )

            records.append(
                {
                    "sample_id": sid,
                    "question": metadata[i]["question"],
                    "image_path": metadata[i]["image_path"],
                    "cropped_image_path": crop_path,
                    "prediction": decoded[i],
                    "n_text_tokens": n_text,
                    "n_latent_tokens": n_latent,
                    "gt_answer": metadata[i]["gt_answer"],
                    "gt_rationale": metadata[i]["gt_rationale"],
                    "pred_latent_path": pred_latent_path,
                    "oracle_latent_path": oracle_latent_path,
                }
            )

        sample_idx += len(metadata)

    N = len(gt_mean_vecs)
    n_valid = sum(valid)
    print(f"\nTotal samples:            {N}")
    print(f"Samples with own latents: {n_valid} ({n_valid/N:.1%})")

    # ── Build gallery tensor ──────────────────────────────────────────────────
    gallery = torch.stack(gt_mean_vecs).float().cpu()  # (N, D)

    # ── Oracle: GT as queries → R@1 must be 1.0 ──────────────────────────────
    oracle_sim = cosine_sim_matrix(gallery, gallery)
    oracle_metrics = retrieval_metrics(oracle_sim, list(range(N)))
    oracle_metrics.pop("_ranks")
    print("\nOracle (GT → GT gallery):")
    for k, v in oracle_metrics.items():
        print(f"  {k}: {v}")

    # ── Predicted latents retrieval ───────────────────────────────────────────
    valid_idx = [i for i, v in enumerate(valid) if v]
    valid_pct = 100.0 * len(valid_idx) / len(valid) if valid else 0.0
    print(f"Valid percentage: {valid_pct:.2f}% ({len(valid_idx)}/{len(valid)})")

    pred_metrics = {}
    paired_mse = None

    if valid_idx:
        queries = torch.stack([pred_mean_vecs[i] for i in valid_idx]).cpu()
        pred_sim = cosine_sim_matrix(queries, gallery)
        pred_metrics = retrieval_metrics(pred_sim, valid_idx)
        pred_metrics.pop("_ranks")

        random_r1 = 1.0 / N
        print("\nPredicted latents retrieval:")
        for k, v in pred_metrics.items():
            print(f"  {k}: {v}")
        print(f"\nRandom baseline R@1: {random_r1:.6f} (1/{N})")
        print(f"Lift over random:    {pred_metrics['R@1'] / random_r1:.1f}x")

        # Paired MSE / cosine
        gt_for_valid = gallery[valid_idx]
        mse_per_sample = ((queries - gt_for_valid) ** 2).mean(dim=-1)
        cos_per_sample = F.cosine_similarity(queries, gt_for_valid, dim=-1)
        paired_mse = mse_per_sample.mean().item()
        paired_cos = cos_per_sample.mean().item()
        print("\nPaired pred↔GT (own sample only):")
        print(f"  MSE (mean):    {paired_mse:.6f}")
        print(f"  MSE (std):     {mse_per_sample.std().item():.6f}")
        print(f"  Cosine (mean): {paired_cos:.6f}")
        print(f"  Cosine (std):  {cos_per_sample.std().item():.6f}")
        pred_metrics["paired_mse"] = paired_mse
        pred_metrics["paired_cosine"] = paired_cos
    else:
        print("\nNo samples predicted latents — model never emitted <|lvr_start|>")

    # ── Write per-sample JSONL ────────────────────────────────────────────────
    jsonl_path = os.path.join(args.output_dir, "results.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved per-sample records → {jsonl_path}")

    # ── Save aggregate JSON ───────────────────────────────────────────────────
    ckpt_name = "_".join(args.model_ref.rstrip("/").split("/")[-2:])
    results = {
        "model": args.model_ref,
        "oracle": oracle_metrics,
        "predicted": pred_metrics,
        "random_r1": 1.0 / N,
        "n_total": N,
        "n_with_latents": n_valid,
        "paired_mse": paired_mse if valid_idx else None,
    }
    out_path = os.path.join(args.output_dir, f"{ckpt_name}_retrieval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved aggregate metrics → {out_path}")
