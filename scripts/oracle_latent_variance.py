"""
Diagnostic: inter-sample variance of oracle (GT) latent embeddings.

Loads the tetris dataset, samples N intermediate images, runs them through the
vision encoder + latent compression pipeline (identical to training), then reports:
  - mean / std of pairwise cosine similarity between samples
  - inter-sample variance of the mean-pooled embeddings
  - per-token-position inter-sample variance
  - a quick t-SNE plot saved to --output_dir

Usage:
    python scripts/oracle_latent_variance.py \
        --data_path /mnt/scratch-nyx/gviveiros/analogy_data/analogy_data/train.json \
        --model_id Qwen/Qwen2.5-VL-3B-Instruct \
        --latent_size 8 \
        --n_samples 1000 \
        --batch_size 16 \
        --output_dir results/oracle_variance
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--latent_size", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/oracle_variance")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    with open(args.data_path) as f:
        dataset = json.load(f)

    def _normalize(path: str) -> str:
        marker = "analogy_data" + os.sep
        idx = path.find(marker)
        if idx != -1:
            path = path[idx + len(marker):]
        return os.path.join(data_dir, path)

    for s in dataset:
        s["reasoning_traces"]["intermediate_img_path"] = _normalize(
            s["reasoning_traces"]["intermediate_img_path"]
        )

    indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    shape_names = [s.get("shape_C_name", "unknown") for s in samples]
    inter_paths = [s["reasoning_traces"]["intermediate_img_path"] for s in samples]

    print(f"Sampled {len(samples)} records from {args.data_path}")

    # ── Load model (vision encoder only needed) ───────────────────────────────
    print("Loading model...")
    from src.models import load_model
    from src.models.utils import apply_latent_compression
    from src.train import set_latent_tokens
    from qwen_vl_utils import process_vision_info

    model, processor = load_model(args.model_id, compute_dtype=torch.bfloat16, use_cache=False)
    set_latent_tokens(processor, model, args.latent_size)
    model.eval()
    device = next(model.parameters()).device

    # ── Extract oracle embeddings in batches ──────────────────────────────────
    all_embeds = []   # list of [latent_size, D] tensors

    for start in tqdm(range(0, len(samples), args.batch_size), desc="Encoding"):
        batch_paths = inter_paths[start: start + args.batch_size]
        batch_imgs  = [Image.open(p).convert("RGB") for p in batch_paths]

        # Build messages the same way the dataset collate does
        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in batch_imgs
        ]

        # Process each image individually to get pixel_values + grid_thw
        pv_list, gt_list = [], []
        for msgs in messages_batch:
            image_inputs, _ = process_vision_info(msgs)
            proc = processor(
                text=processor.apply_chat_template(msgs, tokenize=False),
                images=image_inputs,
                return_tensors="pt",
            )
            pv_list.append(proc["pixel_values"])
            gt_list.append(proc["image_grid_thw"])

        # Concatenate along batch dim (each has 1 image)
        pixel_values  = torch.cat(pv_list,  dim=0).to(device, dtype=torch.bfloat16)
        image_grid_thw = torch.cat(gt_list, dim=0).to(device)

        embeds = apply_latent_compression(model, pixel_values, image_grid_thw, args.latent_size)
        # embeds: [B, latent_size, D]
        all_embeds.append(embeds.float().cpu())

    all_embeds = torch.cat(all_embeds, dim=0)   # [N, latent_size, D]
    N, L, D = all_embeds.shape
    print(f"\nOracle embeddings shape: {all_embeds.shape}  (N={N}, latent_size={L}, D={D})")

    # ── Statistics ────────────────────────────────────────────────────────────

    # 1. Mean-pooled representation [N, D]
    mean_pooled = all_embeds.mean(dim=1)
    mean_pooled_n = F.normalize(mean_pooled, dim=-1)

    # 2. Pairwise cosine similarity
    sim_matrix = mean_pooled_n @ mean_pooled_n.t()   # [N, N]
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = sim_matrix[mask]
    print(f"\n── Pairwise cosine similarity (mean-pooled, N={N}) ──")
    print(f"  mean : {off_diag.mean():.4f}")
    print(f"  std  : {off_diag.std():.4f}")
    print(f"  min  : {off_diag.min():.4f}")
    print(f"  max  : {off_diag.max():.4f}")

    # 3. Inter-sample variance of mean-pooled embeddings
    centred = mean_pooled - mean_pooled.mean(dim=0)
    inter_var = centred.var(dim=0).mean().item()
    print(f"\n── Inter-sample variance (mean-pooled) ──")
    print(f"  mean per-dim variance : {inter_var:.6f}")

    # 4. Per-token-position inter-sample variance
    print(f"\n── Per-token-position inter-sample variance ──")
    for pos in range(L):
        tok = all_embeds[:, pos, :]            # [N, D]
        var_pos = (tok - tok.mean(dim=0)).var(dim=0).mean().item()
        print(f"  token[{pos}]: {var_pos:.6f}")

    # 5. Within-shape vs cross-shape similarity (if shape_names available)
    unique_shapes = list(set(shape_names))
    if len(unique_shapes) > 1:
        shape_to_idx = {s: [] for s in unique_shapes}
        for i, s in enumerate(shape_names):
            shape_to_idx[s].append(i)

        within_sims, cross_sims = [], []
        for i in range(N):
            for j in range(i + 1, N):
                s = sim_matrix[i, j].item()
                if shape_names[i] == shape_names[j]:
                    within_sims.append(s)
                else:
                    cross_sims.append(s)

        print(f"\n── Within-shape vs cross-shape cosine similarity ──")
        if within_sims:
            wt = torch.tensor(within_sims)
            print(f"  within-shape  mean={wt.mean():.4f}  std={wt.std():.4f}  n={len(wt)}")
        if cross_sims:
            ct = torch.tensor(cross_sims)
            print(f"  cross-shape   mean={ct.mean():.4f}  std={ct.std():.4f}  n={len(ct)}")

    # ── t-SNE plot ────────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np

        print("\nRunning t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, N // 4), random_state=args.seed)
        proj = tsne.fit_transform(mean_pooled.numpy())

        # Colour by shape name
        shape_set = sorted(set(shape_names))
        cmap = plt.get_cmap("tab20")
        color_map = {s: cmap(i / max(len(shape_set) - 1, 1)) for i, s in enumerate(shape_set)}
        colors = [color_map[s] for s in shape_names]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=12, alpha=0.7)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[s],
                               markersize=7, label=s) for s in shape_set[:20]]
        ax.legend(handles=handles, fontsize=6, ncol=2)
        ax.set_title(f"Oracle latent embeddings t-SNE (N={N}, latent_size={L})")
        out_path = os.path.join(args.output_dir, "oracle_tsne.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE saved to {out_path}")
    except ImportError:
        print("sklearn/matplotlib not available — skipping t-SNE")

    # ── Save raw embeddings for further analysis ──────────────────────────────
    save_path = os.path.join(args.output_dir, "oracle_embeddings.pt")
    torch.save({"embeddings": all_embeds, "shape_names": shape_names, "paths": inter_paths}, save_path)
    print(f"Raw embeddings saved to {save_path}")


if __name__ == "__main__":
    main()
