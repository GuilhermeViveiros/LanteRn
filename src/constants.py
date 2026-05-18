"""
Central location for all hardcoded paths used across LantErn.
Update these when running on a new cluster/storage mount.
"""

import os

# ---------------------------------------------------------------------------
# Storage roots
# ---------------------------------------------------------------------------
SCRATCH_ARTEMIS = "/mnt/scratch-artemis/gviveiros/lantern"
DATA_ARTEMIS = "/mnt/data-artemis/gviveiros/lantern"
SCRATCH_NYX = "/e/project1/jureap131/gviveiros/lantern"
SCRATCH_JUPITER = "/e/project1/jureap126/gviveiros/lantern"
SCRATCH_HADES = "/mnt/scratch-hades/nunogoncalves/LantErn"
DATA_HADES = "/mnt/data-hades/gviveiros"


# ---------------------------------------------------------------------------
# VisCot data
# ---------------------------------------------------------------------------
def _resolve(*candidates):
    """Return the first path that exists on this machine."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # fall back to first so errors are informative


VISCOT_DATA_PATH = _resolve(
    os.path.join(SCRATCH_JUPITER, "LantErn_VisCot_data.json"),
    os.path.join(DATA_ARTEMIS, "LantErn_VisCot_data.json"),
)
VISCOT_MC_TEST_PATH = _resolve(
    os.path.join(SCRATCH_JUPITER, "viscot_mc_test.jsonl"),
    os.path.join(SCRATCH_ARTEMIS, "viscot_mc_test.jsonl"),
)
VISCOT_IMAGE_ROOT = _resolve(SCRATCH_JUPITER, DATA_ARTEMIS)
VISCOT_IMAGE_ROOT_FALLBACK = VISCOT_IMAGE_ROOT

# missing image - skip sample for now
TEXTVQA_BAD_SAMPLE = os.path.join(DATA_ARTEMIS, "textvqa/34084d4c3c347b83.jpg")

# ---------------------------------------------------------------------------
# Tetris analogy data
# ---------------------------------------------------------------------------
TETRIS_DATA_DIR = "/e/project1/jureap131/gviveiros/lantern/analogy_data"
TETRIS_TRAIN_PATH = os.path.join(TETRIS_DATA_DIR, "train.json")
TETRIS_EVAL_PATH = os.path.join(TETRIS_DATA_DIR, "eval.json")

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = os.path.join(SCRATCH_NYX, "checkpoints")
CHECKPOINTS_DIR_ARTEMIS = os.path.join(SCRATCH_ARTEMIS, "checkpoints")

# ---------------------------------------------------------------------------
# RL data
# ---------------------------------------------------------------------------
RL_DATA_PATH = os.path.join(SCRATCH_HADES, "rl_dataset/lvr_data/virl39k.json")
RL_IMAGE_ROOT = DATA_HADES

# ---------------------------------------------------------------------------
# HuggingFace cache snapshots
# ---------------------------------------------------------------------------
HF_VISUAL_COT_METADATA = (
    "/home/gviveiros/.cache/huggingface/hub/"
    "datasets--deepcs233--Visual-CoT/snapshots/"
    "223d2d8c1146fda2bb918801b8276c587b78b61c/metadata"
)
