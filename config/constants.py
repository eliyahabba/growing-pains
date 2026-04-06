from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anchor selection methods
ANCHOR_IRT_CLUSTERING = "irt_clustering"
ANCHOR_TOP_K = "top_k_discrimination"
ANCHOR_CORRECTNESS = "correctness_clustering"

# Calibration methods
METHOD_FIXED = "fixed"
METHOD_CONCURRENT = "concurrent"

CHAIN_DIRECT = "direct"

# Data source modes
MODE_LB_ALIASES = frozenset({"lb", "lb_only", "tinybenchmarks"})
MODE_MMLU_ALIASES = frozenset({"mmlu_split", "mmlu_fields"})

ERROR_METRICS = ["anchor_error", "irt_error", "gp_irt_error", "pirt_error"]

EXCLUDED_DATASETS = frozenset({
    "Summarization", "Copyright", "BOLD",
    "RealToxicityPrompts", "SyntheticReasoning", "Disinformation",
})

MIN_ANCHORS_PER_DATASET = 5
MAX_RETRIES = 3
