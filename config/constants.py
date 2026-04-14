from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anchor selection methods
ANCHOR_IRT_CLUSTERING = "irt_clustering"
ANCHOR_TOP_K = "top_k_discrimination"
ANCHOR_CORRECTNESS = "correctness_clustering"

# CLI / legacy names -> canonical method strings (see irt/anchors.py)
ANCHOR_METHOD_ALIASES: dict[str, str] = {
    "anchor-irt": ANCHOR_IRT_CLUSTERING,
    "anchor_irt": ANCHOR_IRT_CLUSTERING,
    "anchor": ANCHOR_CORRECTNESS,
}

# Calibration methods
METHOD_FIXED = "fixed"
METHOD_CONCURRENT = "concurrent"

CHAIN_DIRECT = "direct"

# Data source modes
MODE_LB_ALIASES = frozenset({"lb", "lb_only"})
MODE_MMLU_ALIASES = frozenset({"mmlu_split", "mmlu_fields"})

ERROR_METRICS = ["anchor_error", "irt_error", "gp_irt_error", "pirt_error"]

MIN_ANCHORS_PER_DATASET = 5
MAX_RETRIES = 3

# --- Sweep grids & outputs (scripts/run_experiments.py) ---

LB_DATASETS = [
    "MMLU",
    "ARC Challenge",
    "HellaSwag",
    "TruthfulQA",
    "Winogrande",
    "GSM8K",
]

LB_ANCHOR_COUNTS = [25, 50, 100, 200]
LB_ANCHOR_SWEEP_BASE_SEED = 31

LB_EXTENDED_MODEL_COUNTS = [5, 10, 25, 50, 100, 150, 200, 250, 300]
LB_EXTENDED_BASE_SEED = 101
LB_EXTENDED_N_ANCHORS = 100

MMLU_ANCHOR_COUNTS = [5, 10, 25, 50, 100]
MMLU_ANCHOR_SWEEP_SEEDS = [11, 12, 13, 14]

MMLU_EXTENDED_MODEL_COUNTS = [5, 10, 25, 50, 100, 150, 200, 250, 300]
MMLU_EXTENDED_N_ANCHORS = 50
MMLU_EXTENDED_BASE_SEED = 201

# Relative to repo root unless --base-dir is absolute
DEFAULT_BASE_DIR = Path("data") / "output"

PRESET_LB_STANDARD = "lb_standard"
PRESET_MMLU_FIELDS = "mmlu_fields"

OUTPUT_LB_ANCHOR_SWEEP = "lb_anchor_sweep"
OUTPUT_LB_REFMODEL_SWEEP = "lb_refmodel_sweep"
OUTPUT_MMLU_ANCHOR_SWEEP = "mmlu_anchor_sweep"
OUTPUT_MMLU_REFMODEL_SWEEP = "mmlu_refmodel_sweep"

DEFAULT_TEST_RATIO = 0.25
DEFAULT_EPOCHS = 2000
DEFAULT_EPOCHS_FIXED = 1000
DEFAULT_DIMS = [5]
DEFAULT_NUM_WORKERS = 4
DEFAULT_RANDOM_SEED_BASE = 1000
