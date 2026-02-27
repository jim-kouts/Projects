# config.py
# =========
# Central configuration file for the entire project.
# Instead of hardcoding paths and parameters in every script,
# we define them once here and import them wherever needed.
#
# Why is this useful?
# If you want to change the model directory or max_length,
# you change it in ONE place.

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
# Path(__file__) = this file (config.py)
# .parent        = utils/
# .parent.parent = project root
ROOT_DIR = Path(__file__).parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = ROOT_DIR / "data" / "raw" / "funsd_subset"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed" / "funsd_layoutlm"
PLOTS_DIR          = ROOT_DIR / "data" / "plots"
INFERENCE_OUT_DIR  = ROOT_DIR / "data" / "inference_output"

# ── Model paths ────────────────────────────────────────────────────────────────
MODELS_DIR        = ROOT_DIR / "models"
BEST_MODEL_DIR    = ROOT_DIR / "models" / "best_model"
FINAL_MODEL_DIR   = ROOT_DIR / "models" / "final_model"

# ── Model settings ─────────────────────────────────────────────────────────────
PRETRAINED_MODEL  = "microsoft/layoutlmv3-base"
MAX_TOKEN_LENGTH  = 512

# ── Training hyperparameters ───────────────────────────────────────────────────
# Hyperparameters: settings that control how training works.
# These are the defaults — they can be overridden via argparse.
DEFAULT_EPOCHS        = 5
DEFAULT_BATCH_SIZE    = 2
DEFAULT_LEARNING_RATE = 5e-5

# ── Label mapping ──────────────────────────────────────────────────────────────
# Defined once here so every script imports from the same place
ID2LABEL = {
    0: "O",
    1: "B-HEADER",
    2: "I-HEADER",
    3: "B-QUESTION",
    4: "I-QUESTION",
    5: "B-ANSWER",
    6: "I-ANSWER",
}

LABEL2ID  = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# ── API settings ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000