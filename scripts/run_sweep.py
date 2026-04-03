from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config.constants import (
    DEFAULT_BASE_DIR,
    HELM_EXTENDED_BASE_SEED,
    HELM_EXTENDED_MODEL_COUNTS,
    HELM_EXTENDED_N_ANCHORS,
    HELM_LITE_DATASETS,
    LB_ANCHOR_COUNTS,
    LB_ANCHOR_SWEEP_BASE_SEED,
    LB_DATASETS,
    LB_EXTENDED_BASE_SEED,
    LB_EXTENDED_MODEL_COUNTS,
    LB_EXTENDED_N_ANCHORS,
    LB_MODEL_SWEEP_BASE_SEED,
    LB_MODEL_SWEEP_COUNTS,
    MMLU_ANCHOR_COUNTS,
    MMLU_ANCHOR_SWEEP_SEEDS,
    MMLU_EXTENDED_BASE_SEED,
    MMLU_EXTENDED_MODEL_COUNTS,
    MMLU_EXTENDED_N_ANCHORS,
    OUTPUT_HELM_MODEL_EXTENDED,
    OUTPUT_LB_ANCHOR_SWEEP,
    OUTPUT_LB_MODEL_EXTENDED,
    OUTPUT_LB_MODEL_SWEEP,
    OUTPUT_MMLU_ANCHOR_SWEEP,
    OUTPUT_MMLU_MODEL_EXTENDED,
)

CHAIN = REPO_ROOT / "src/experiments/chain_linking/chain_linking_parallel.py"


def _env() -> dict[str, str]:
    e = os.environ.copy()
    e["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    return e


def _run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, env=_env(), cwd=REPO_ROOT)


def _lb_common(anchors: int) -> list[str]:
    return [
        "--n-base",
        "1",
        "--max-chain",
        "5",
        "--data-source-mode",
        "lb",
        "--n-anchors-per-dataset",
        str(anchors),
        "--dims",
        "5",
    ]


def _mmlu_common(anchors: int) -> list[str]:
    return [
        "--n-base",
        "8",
        "--max-chain",
        "10",
        "--data-source-mode",
        "mmlu_fields",
        "--n-anchors-per-dataset",
        str(anchors),
        "--dims",
        "5",
    ]


def _helm_common(anchors: int) -> list[str]:
    return [
        "--n-base",
        "1",
        "--max-chain",
        "10",
        "--data-source-mode",
        "helm_lite",
        "--n-anchors-per-dataset",
        str(anchors),
        "--dims",
        "5",
    ]


def cat_lb_anchor(base: Path, dry: bool) -> None:
    out = base / OUTPUT_LB_ANCHOR_SWEEP
    seed = LB_ANCHOR_SWEEP_BASE_SEED
    for target in LB_DATASETS:
        tseed = seed
        seed += 1
        for anchors in LB_ANCHOR_COUNTS:
            cmd = [
                sys.executable,
                str(CHAIN),
                "--output-dir",
                str(out),
                *_lb_common(anchors),
                "--shuffle-seed",
                str(tseed),
                "--seed",
                str(tseed),
                "--target-dataset",
                target,
                "--random-seed",
                str(1000 + anchors),
            ]
            _run(cmd, dry)


def cat_lb_model(base: Path, dry: bool) -> None:
    out = base / OUTPUT_LB_MODEL_SWEEP
    seed = LB_MODEL_SWEEP_BASE_SEED
    for target in LB_DATASETS:
        tseed = seed
        seed += 1
        for models in LB_MODEL_SWEEP_COUNTS:
            cmd = [
                sys.executable,
                str(CHAIN),
                "--output-dir",
                str(out),
                *_lb_common(LB_EXTENDED_N_ANCHORS),
                "--n-models-per-chain",
                str(models),
                "--shuffle-seed",
                str(tseed),
                "--seed",
                str(tseed),
                "--target-dataset",
                target,
                "--random-seed",
                str(2000 + models),
            ]
            _run(cmd, dry)


def cat_lb_extended(base: Path, n_seeds: int, dry: bool) -> None:
    out = base / OUTPUT_LB_MODEL_EXTENDED
    for seed_offset in range(n_seeds):
        for ti, target in enumerate(LB_DATASETS):
            run_seed = LB_EXTENDED_BASE_SEED + ti * n_seeds + seed_offset
            for models in LB_EXTENDED_MODEL_COUNTS:
                cmd = [
                    sys.executable,
                    str(CHAIN),
                    "--output-dir",
                    str(out),
                    *_lb_common(LB_EXTENDED_N_ANCHORS),
                    "--n-models-per-chain",
                    str(models),
                    "--shuffle-seed",
                    str(run_seed),
                    "--seed",
                    str(run_seed),
                    "--target-dataset",
                    target,
                    "--random-seed",
                    str(3000 + models),
                ]
                _run(cmd, dry)


def cat_mmlu_extended(base: Path, n_seeds: int, dry: bool) -> None:
    out = base / OUTPUT_MMLU_MODEL_EXTENDED
    for seed_offset in range(n_seeds):
        run_seed = MMLU_EXTENDED_BASE_SEED + seed_offset
        for models in MMLU_EXTENDED_MODEL_COUNTS:
            cmd = [
                sys.executable,
                str(CHAIN),
                "--output-dir",
                str(out),
                *_mmlu_common(MMLU_EXTENDED_N_ANCHORS),
                "--n-models-per-chain",
                str(models),
                "--shuffle-seed",
                str(run_seed),
                "--seed",
                str(run_seed),
                "--random-seed",
                str(4000 + models),
            ]
            _run(cmd, dry)


def cat_helm_extended(base: Path, n_seeds: int, dry: bool) -> None:
    out = base / OUTPUT_HELM_MODEL_EXTENDED
    for seed_offset in range(n_seeds):
        for ti, target in enumerate(HELM_LITE_DATASETS):
            run_seed = HELM_EXTENDED_BASE_SEED + ti * n_seeds + seed_offset
            for models in HELM_EXTENDED_MODEL_COUNTS:
                cmd = [
                    sys.executable,
                    str(CHAIN),
                    "--output-dir",
                    str(out),
                    *_helm_common(HELM_EXTENDED_N_ANCHORS),
                    "--n-models-per-chain",
                    str(models),
                    "--shuffle-seed",
                    str(run_seed),
                    "--seed",
                    str(run_seed),
                    "--target-dataset",
                    target,
                    "--random-seed",
                    str(5000 + models),
                ]
                _run(cmd, dry)


def cat_mmlu_anchor(base: Path, dry: bool) -> None:
    out = base / OUTPUT_MMLU_ANCHOR_SWEEP
    for run_seed in MMLU_ANCHOR_SWEEP_SEEDS:
        for anchors in MMLU_ANCHOR_COUNTS:
            cmd = [
                sys.executable,
                str(CHAIN),
                "--output-dir",
                str(out),
                *_mmlu_common(anchors),
                "--shuffle-seed",
                str(run_seed),
                "--seed",
                str(run_seed),
                "--random-seed",
                str(6000 + anchors),
            ]
            _run(cmd, dry)


def main() -> None:
    p = argparse.ArgumentParser(description="Anchor/model sweep orchestrator")
    p.add_argument(
        "--category",
        choices=["1a", "1b", "5", "6", "7", "8", "all"],
        default="all",
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    base = args.base_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    dry = args.dry_run
    cats = ["1a", "1b", "5", "6", "7", "8"] if args.category == "all" else [args.category]
    for c in cats:
        if c == "1a":
            cat_lb_anchor(base, dry)
        elif c == "1b":
            cat_lb_model(base, dry)
        elif c == "5":
            cat_lb_extended(base, args.n_seeds, dry)
        elif c == "6":
            cat_mmlu_extended(base, args.n_seeds, dry)
        elif c == "7":
            cat_helm_extended(base, args.n_seeds, dry)
        elif c == "8":
            cat_mmlu_anchor(base, dry)


if __name__ == "__main__":
    main()
