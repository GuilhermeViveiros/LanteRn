"""
Central location for all hardcoded paths used across LantErn.
Update these when running on a new cluster/storage mount.
"""
import os

# ---------------------------------------------------------------------------
# Storage roots
# ---------------------------------------------------------------------------
SCRATCH_ARTEMIS  = "/mnt/scratch-artemis/gviveiros/lantern"
DATA_ARTEMIS     = "/mnt/data-artemis/gviveiros/lantern"
SCRATCH_NYX      = "/mnt/scratch-nyx/gviveiros/lantern"
SCRATCH_HADES    = "/mnt/scratch-hades/nunogoncalves/LantErn"
DATA_HADES       = "/mnt/data-hades/gviveiros"

# ---------------------------------------------------------------------------
# VisCot data
# ---------------------------------------------------------------------------
VISCOT_DATA_PATH     = os.path.join(DATA_ARTEMIS, "LantErn_VisCot_data.json")
VISCOT_MC_TEST_PATH  = os.path.join(SCRATCH_ARTEMIS, "viscot_mc_test.jsonl")
VISCOT_IMAGE_ROOT    = DATA_ARTEMIS   # prefix for image paths in VisCot samples
VISCOT_IMAGE_ROOT_FALLBACK = SCRATCH_ARTEMIS  # some images were moved here

# Known bad sample to skip
TEXTVQA_BAD_SAMPLE = os.path.join(DATA_ARTEMIS, "textvqa/34084d4c3c347b83.jpg")

# ---------------------------------------------------------------------------
# Tetris analogy data
# ---------------------------------------------------------------------------
TETRIS_DATA_DIR   = os.path.join(SCRATCH_NYX, "analogy_data")
TETRIS_TRAIN_PATH = os.path.join(TETRIS_DATA_DIR, "train.json")
TETRIS_EVAL_PATH  = os.path.join(TETRIS_DATA_DIR, "eval.json")

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR       = os.path.join(SCRATCH_NYX, "checkpoints")
CHECKPOINTS_DIR_ARTEMIS = os.path.join(SCRATCH_ARTEMIS, "checkpoints")

# ---------------------------------------------------------------------------
# RL data
# ---------------------------------------------------------------------------
RL_DATA_PATH      = os.path.join(SCRATCH_HADES, "rl_dataset/lvr_data/virl39k.json")
RL_IMAGE_ROOT     = DATA_HADES

# ---------------------------------------------------------------------------
# HuggingFace cache snapshots
# ---------------------------------------------------------------------------
HF_VISUAL_COT_METADATA = (
    "/home/gviveiros/.cache/huggingface/hub/"
    "datasets--deepcs233--Visual-CoT/snapshots/"
    "223d2d8c1146fda2bb918801b8276c587b78b61c/metadata"
)
HF_MONET_SNAPSHOT = (
    "/home/gviveiros/.cache/huggingface/hub/"
    "datasets--NOVAglow646--Monet-SFT-125K/snapshots/"
    "c77a2df9624e3e1e229dda90139453181e018eb8/"
)
