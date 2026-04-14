"""Experiment orchestrator for reproducing paper results.

Usage:
    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --experiment lb_anchor_sweep --dry-run
    python scripts/run_experiments.py --experiment lb_anchor_sweep

On a SLURM cluster, add --dispatch sbatch to submit one job per run:
    python scripts/run_experiments.py --experiment lb_anchor_sweep --dispatch sbatch
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config.constants import (
    DEFAULT_BASE_DIR,
    LB_ANCHOR_COUNTS,
    LB_ANCHOR_SWEEP_BASE_SEED,
    LB_DATASETS,
    LB_EXTENDED_BASE_SEED,
    LB_EXTENDED_MODEL_COUNTS,
    LB_EXTENDED_N_ANCHORS,
    MMLU_ANCHOR_COUNTS,
    MMLU_ANCHOR_SWEEP_SEEDS,
    MMLU_EXTENDED_BASE_SEED,
    MMLU_EXTENDED_MODEL_COUNTS,
    MMLU_EXTENDED_N_ANCHORS,
    OUTPUT_LB_ANCHOR_SWEEP,
    OUTPUT_LB_REFMODEL_SWEEP,
    OUTPUT_MMLU_ANCHOR_SWEEP,
    OUTPUT_MMLU_REFMODEL_SWEEP,
)

CHAIN_MODULE = "src.chain_experiment"
DEFAULT_SBATCH_SCRIPT = Path(__file__).parent / "run_chain_job.sh"


# Experiment registry: name -> (description, paper figure)
EXPERIMENTS: dict[str, str] = {
    "lb_anchor_sweep": (
        "Anchor count sweep on the Open LLM Leaderboard (Figure 3). "
        "Runs anchor counts [25, 50, 100, 200] x 6 datasets."
    ),
    "lb_refmodel_sweep": (
        "Reference model count sweep on the Open LLM Leaderboard (Figure 4). "
        "Runs model counts [5, 10, 25, 50, 100, 150, 200, 250, 300] x 6 datasets x n_seeds."
    ),
    "mmlu_anchor_sweep": (
        "Anchor count sweep on MMLU (Figure 3). "
        "Runs anchor counts [5, 10, 25, 50, 100] x 4 seeds."
    ),
    "mmlu_refmodel_sweep": (
        "Reference model count sweep on MMLU (Figure 4). "
        "Runs model counts [5, 10, 25, 50, 100, 150, 200, 250, 300] x n_seeds."
    ),
}


@dataclass(frozen=True)
class ExecConfig:
    """Execution settings for each chain_experiment invocation."""
    dry_run: bool
    dispatch: str       # "local" | "sbatch"
    sbatch_script: Path
    sbatch_extra: tuple[str, ...]


def _num_workers_args(n: int) -> list[str]:
    return ["--num-workers", str(n)]


def _chain_cli_args(cmd: list[str]) -> list[str]:
    """Drop [python, -m, module]; return only chain_experiment flags."""
    if len(cmd) < 4:
        raise ValueError(f"expected python -m module + flags, got {cmd!r}")
    return cmd[3:]


def _run(cmd: list[str], x: ExecConfig) -> None:
    if x.dispatch == "sbatch":
        inner = _chain_cli_args(cmd)
        sbatch_cmd = ["sbatch", *x.sbatch_extra, str(x.sbatch_script), *inner]
        print(shlex.join(sbatch_cmd))
        if not x.dry_run:
            subprocess.run(sbatch_cmd, check=True)
        return
    print(shlex.join(cmd))
    if not x.dry_run:
        subprocess.run(cmd, check=True)


def _lb_common(anchors: int) -> list[str]:
    return [
        "--n-base", "1",
        "--max-chain", "5",
        "--data-source-mode", "lb",
        "--n-anchors-per-dataset", str(anchors),
        "--dims", "5",
    ]


def _mmlu_common(anchors: int) -> list[str]:
    return [
        "--n-base", "8",
        "--max-chain", "10",
        "--data-source-mode", "mmlu_fields",
        "--n-anchors-per-dataset", str(anchors),
        "--dims", "5",
    ]


def run_lb_anchor_sweep(base: Path, num_workers: int, x: ExecConfig) -> None:
    """Anchor count sweep on the Open LLM Leaderboard (Figure 3)."""
    out = base / OUTPUT_LB_ANCHOR_SWEEP
    seed = LB_ANCHOR_SWEEP_BASE_SEED
    for target in LB_DATASETS:
        tseed = seed
        seed += 1
        for anchors in LB_ANCHOR_COUNTS:
            cmd = [
                sys.executable, "-m", CHAIN_MODULE,
                "--output-dir", str(out),
                *_lb_common(anchors),
                "--shuffle-seed", str(tseed),
                "--seed", str(tseed),
                "--target-dataset", target,
                "--random-seed", str(1000 + anchors),
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def run_lb_refmodel_sweep(base: Path, n_seeds: int, num_workers: int, x: ExecConfig) -> None:
    """Reference model count sweep on the Open LLM Leaderboard (Figure 4)."""
    out = base / OUTPUT_LB_REFMODEL_SWEEP
    for seed_offset in range(n_seeds):
        for ti, target in enumerate(LB_DATASETS):
            run_seed = LB_EXTENDED_BASE_SEED + ti * n_seeds + seed_offset
            for models in LB_EXTENDED_MODEL_COUNTS:
                cmd = [
                    sys.executable, "-m", CHAIN_MODULE,
                    "--output-dir", str(out),
                    *_lb_common(LB_EXTENDED_N_ANCHORS),
                    "--n-models-per-chain", str(models),
                    "--shuffle-seed", str(run_seed),
                    "--seed", str(run_seed),
                    "--target-dataset", target,
                    "--random-seed", str(3000 + models),
                    *_num_workers_args(num_workers),
                ]
                _run(cmd, x)


def run_mmlu_anchor_sweep(base: Path, num_workers: int, x: ExecConfig) -> None:
    """Anchor count sweep on MMLU (Figure 3)."""
    out = base / OUTPUT_MMLU_ANCHOR_SWEEP
    for run_seed in MMLU_ANCHOR_SWEEP_SEEDS:
        for anchors in MMLU_ANCHOR_COUNTS:
            cmd = [
                sys.executable, "-m", CHAIN_MODULE,
                "--output-dir", str(out),
                *_mmlu_common(anchors),
                "--shuffle-seed", str(run_seed),
                "--seed", str(run_seed),
                "--random-seed", str(6000 + anchors),
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def run_mmlu_refmodel_sweep(base: Path, n_seeds: int, num_workers: int, x: ExecConfig) -> None:
    """Reference model count sweep on MMLU (Figure 4)."""
    out = base / OUTPUT_MMLU_REFMODEL_SWEEP
    for seed_offset in range(n_seeds):
        run_seed = MMLU_EXTENDED_BASE_SEED + seed_offset
        for models in MMLU_EXTENDED_MODEL_COUNTS:
            cmd = [
                sys.executable, "-m", CHAIN_MODULE,
                "--output-dir", str(out),
                *_mmlu_common(MMLU_EXTENDED_N_ANCHORS),
                "--n-models-per-chain", str(models),
                "--shuffle-seed", str(run_seed),
                "--seed", str(run_seed),
                "--random-seed", str(4000 + models),
                *_num_workers_args(num_workers),
            ]
            _run(cmd, x)


def list_experiments() -> None:
    print("Available experiments:\n")
    for name, desc in EXPERIMENTS.items():
        print(f"  {name}")
        print(f"    {desc}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run sweep experiments for the Growing Pains paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_experiments.py --list\n"
            "  python scripts/run_experiments.py --experiment lb_anchor_sweep --dry-run\n"
            "  python scripts/run_experiments.py --experiment lb_anchor_sweep\n"
            "  python scripts/run_experiments.py --experiment lb_anchor_sweep --dispatch sbatch\n"
        ),
    )
    p.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS),
        metavar="NAME",
        help=f"Experiment to run. One of: {', '.join(EXPERIMENTS)}",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Number of random seeds for reference model sweeps (default: 3).")
    p.add_argument(
        "--dispatch",
        choices=("local", "sbatch"),
        default="local",
        help="local: run in this process. sbatch: submit one Slurm job per run.",
    )
    p.add_argument(
        "--sbatch-script",
        type=Path,
        default=DEFAULT_SBATCH_SCRIPT,
        help="SLURM batch script wrapping chain_experiment.py.",
    )
    p.add_argument(
        "--sbatch-extra",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra sbatch args, e.g. --sbatch-extra --mem=8g (repeatable).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers forwarded to chain_experiment (default: 4 with sbatch, 1 locally).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")

    args = p.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        p.error("specify --experiment NAME or --list")

    base = args.base_dir
    if not base.is_absolute():
        base = base.resolve()

    nw = args.num_workers
    if nw is None:
        nw = 4 if args.dispatch == "sbatch" else 1

    x = ExecConfig(
        dry_run=args.dry_run,
        dispatch=args.dispatch,
        sbatch_script=args.sbatch_script.resolve(),
        sbatch_extra=tuple(args.sbatch_extra),
    )

    exp = args.experiment
    if exp == "lb_anchor_sweep":
        run_lb_anchor_sweep(base, nw, x)
    elif exp == "lb_refmodel_sweep":
        run_lb_refmodel_sweep(base, args.n_seeds, nw, x)
    elif exp == "mmlu_anchor_sweep":
        run_mmlu_anchor_sweep(base, nw, x)
    elif exp == "mmlu_refmodel_sweep":
        run_mmlu_refmodel_sweep(base, args.n_seeds, nw, x)


if __name__ == "__main__":
    main()
