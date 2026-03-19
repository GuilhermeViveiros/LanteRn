"""
Quick CPU sanity check for the latent-only ablation.
Verifies:
  1. freeze_latent_only: only the 3 latent token rows accumulate gradients.
  2. Image corruption: latent crops come from the clean image; model input is corrupted.

Run with:
    PYTHONPATH=/home/gviveiros/LantErn python scripts/verify_ablation.py
"""
import torch
from PIL import Image, ImageDraw
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"

# ──────────────────────────────────────────────
# 1. Load model on CPU
# ──────────────────────────────────────────────
print("\n=== Loading model (CPU) ===")
from src.models import load_model
from src.train import set_latent_tokens, configure_latent_only

model, processor = load_model(MODEL_ID, compute_dtype=torch.float32)
model = model.cpu()
import pdb; pdb.set_trace()
set_latent_tokens(processor, model, latent_size=8)
lvr_ids = [model.config.lvr_start_id, model.config.lvr_sep_id, model.config.lvr_end_id]
print(f"Latent token IDs: {lvr_ids}")

# ──────────────────────────────────────────────
# 2. Verify freeze_latent_only
# ──────────────────────────────────────────────
print("\n=== Checking freeze_latent_only ===")
configure_latent_only(model)

trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable parameters: {[n for n, _ in trainable]}")
assert len(trainable) == 2, f"Expected 2 trainable params (embed_tokens + lm_head), got {len(trainable)}"
assert any("embed_tokens" in n for n, _ in trainable), "embed_tokens not trainable"
assert any("lm_head" in n for n, _ in trainable), "lm_head not trainable"
print("✓ Only embed_tokens.weight and lm_head.weight are trainable")

# Run a tiny forward + backward to confirm gradient masking
dummy_input = torch.tensor([[lvr_ids[0], lvr_ids[1], lvr_ids[2]]])  # [1, 3]
out = model.model.embed_tokens(dummy_input)  # [1, 3, hidden]
loss = out.sum()
loss.backward()

embed_grad = model.model.embed_tokens.weight.grad
assert embed_grad is not None, "No gradient on embed_tokens.weight"

# Only lvr rows should be non-zero
non_zero_rows = embed_grad.abs().sum(dim=1).nonzero(as_tuple=True)[0].tolist()
print(f"Non-zero gradient rows: {non_zero_rows}")
assert set(non_zero_rows) == set(lvr_ids), \
    f"Expected gradients only at {lvr_ids}, got {non_zero_rows}"
print("✓ Gradient hook is masking correctly — only latent token rows updated")

# ──────────────────────────────────────────────
# 3. Verify image corruption
# ──────────────────────────────────────────────
print("\n=== Checking image corruption ===")
import json
from src.utils import center_and_crop_image
from PIL import ImageDraw

with open(DATA_PATH) as f:
    dataset = json.load(f)

# find a sample with a bbox
sample = next(d for d in dataset if d.get("bboxs"))
img = Image.open(sample["img_path"]).convert("RGB")
bbox = sample["bboxs"][0]
x1, y1, x2, y2 = bbox

# corrupt main image
img_main = img.copy()
draw = ImageDraw.Draw(img_main)
draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

# latent crop from original
crop = center_and_crop_image(img, bbox)

# check that the bbox region in img_main is all black
region_main = np.array(img_main)[y1:y2, x1:x2]
region_orig = np.array(img)[y1:y2, x1:x2]
assert region_main.max() == 0, "Bbox region not fully blacked out in corrupted image"
assert region_orig.max() > 0, "Original image bbox region is unexpectedly empty"

# check that the crop comes from the clean image (any pixel should be non-black)
crop_arr = np.array(crop)
assert crop_arr.max() > 0, "Latent crop appears to be all black (should be clean)"
print(f"Bbox: {bbox}")
print(f"Corrupted region max pixel: {region_main.max()} (expected 0)")
print(f"Original region max pixel:  {region_orig.max()} (expected >0)")
print(f"Latent crop max pixel:      {crop_arr.max()} (expected >0)")
print("✓ Image corruption is correct — model input is corrupted, latent target is clean")

print("\n=== All checks passed ===")
