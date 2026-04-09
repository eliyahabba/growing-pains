from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
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

CHAIN = REPO_ROOT / "src/chain_experiment.py"
DEFAULT_SBATCH_SCRIPT = REPO_ROOT / "scripts/run_chain_job.sh"


@dataclass(frozen=True)
class SweepExec:
    """How each chain_experiment invocation is executed."""

    dry_run: bool
    dispatch: str  # "local" | "sbatch"
    sbatch_script: Path
    sbatch_extra: tuple[str, ...]


def _num_workers_args(n: int) -> list[str]:
    return ["--num-workers", str(n)]


def _env() -> dict[str, str]:
    e = os.environ.copy()
    e["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    return e


def _chain_cli_args(cmd: list[str]) -> list[str]:
    """Drop [python, chain_experiment.py]; keep flags for chain_experiment."""
    if len(cmd) < 3:
        raise ValueError(f"expected python + chain script + args, got {cmd!r}")
    return cmd[2:]


def _run(cmd: list[str], x: SweepExec) -> None:
    if x.dispatch == "sbatch":
        inner = _chain_cli_args(cmd)
        sbatch_cmd = ["sbatch", *x.sbatch_extra, str(x.sbatch_script), *inner]
        print(shlex.join(sbatch_cmd))
        if not x.dry_run:
            subprocess.run(sbatch_cmd, check=True, cwd=REPO_ROOT)
        return
    print(shlex.join(cmd))
    if not x.dry_run:
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


def cat_lb_anchor(base: Path, num_workers: int, x: SweepExec) -> None:
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
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def cat_lb_model(base: Path, num_workers: int, x: SweepExec) -> None:
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
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def cat_lb_extended(base: Path, n_seeds: int, num_workers: int, x: SweepExec) -> None:
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
                    *_num_workers_args(num_workers),
                ]
                _run(cmd, x)


def cat_mmlu_extended(base: Path, n_seeds: int, num_workers: int, x: SweepExec) -> None:
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
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def cat_helm_extended(base: Path, n_seeds: int, num_workers: int, x: SweepExec) -> None:
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
                    *_num_workers_args(num_workers),
                ]
                _run(cmd, x)


def cat_mmlu_anchor(base: Path, num_workers: int, x: SweepExec) -> None:
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
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Anchor/model sweep orchestrator. "
            "Use --dispatch sbatch on clusters: one Slurm job per experiment (like AdaptEval "
            "run_all_balanced.sh + run_chain_linking_unified.sh), each with its own memory limit."
        ),
    )
    p.add_argument(
        "--category",
        choices=["1a", "1b", "5", "6", "7", "8", "all"],
        default="all",
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument(
        "--dispatch",
        choices=("local", "sbatch"),
        default="local",
        help="local: run chain_experiment in this process tree. sbatch: submit one job per run via run_chain_job.sh.",
    )
    p.add_argument(
        "--sbatch-script",
        type=Path,
        default=DEFAULT_SBATCH_SCRIPT,
        help="Batch script wrapping a single chain_experiment.py (must start with #SBATCH lines).",
    )
    p.add_argument(
        "--sbatch-extra",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra args before the job script, e.g. --sbatch-extra --mem=8g (repeatable).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Forwarded to chain_experiment. Default: 4 with --dispatch sbatch, else 1.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    base = args.base_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    cats = ["1a", "1b", "5", "6", "7", "8"] if args.category == "all" else [args.category]
    nw = args.num_workers
    if nw is None:
        nw = 4 if args.dispatch == "sbatch" else 1
    x = SweepExec(
        dry_run=args.dry_run,
        dispatch=args.dispatch,
        sbatch_script=args.sbatch_script.resolve(),
        sbatch_extra=tuple(args.sbatch_extra),
    )
    for c in cats:
        if c == "1a":
            cat_lb_anchor(base, nw, x)
        elif c == "1b":
            cat_lb_model(base, nw, x)
        elif c == "5":
            cat_lb_extended(base, args.n_seeds, nw, x)
        elif c == "6":
            cat_mmlu_extended(base, args.n_seeds, nw, x)
        elif c == "7":
            cat_helm_extended(base, args.n_seeds, nw, x)
        elif c == "8":
            cat_mmlu_anchor(base, nw, x)


if __name__ == "__main__":
    main()
