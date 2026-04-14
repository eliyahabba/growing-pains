# Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Code for the paper "Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration" (Habba et al., 2026, preprint).

The rapid release of both language models and benchmarks makes it increasingly costly to evaluate every model on every dataset. In practice, models are often evaluated on different samples, making scores difficult to compare across studies. We propose a framework based on multidimensional Item Response Theory (IRT) that uses anchor items to calibrate new benchmarks to the evaluation suite while holding previously calibrated item parameters fixed. In large-scale experiments on more than 400 models, our framework predicts full-evaluation performance within 2–3 percentage points using only 100 anchor questions per dataset, with Spearman ρ ≥ 0.9 for ranking preservation, showing that it is possible to extend benchmark suites over time while preserving score comparability, at a constant evaluation cost per new dataset.

## Repository layout

```
config/
  constants.py              sweep grids, dataset lists, anchor methods
  data_sources.py           data source configuration builders
  data_source_config.json
  experiment_presets.yaml

src/
  chain_experiment.py       entry point — runs one chain calibration experiment
  calibration.py            base frame calibration, fixed-anchor calibration, anchor selection
  data_loading.py           dataset loading and grouping
  evaluation.py             MAE and Spearman rho evaluation, baselines
  io.py                     shared I/O helpers
  irt/                      IRT engine: MIRT 2PL training, anchor selection, theta estimation

scripts/
  run_experiments.py        orchestrator for anchor count and reference model sweeps

data/
  input/                    benchmark response files (tracked in git)
  output/                   experiment results written here (not in git)

tests/
  demo_test.py              fast synthetic IRT test + optional real-data test
```

## Setup

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Data

Benchmark response files (`lb.pickle`, `mmlu_fields.pickle`) are included in `data/input/`. The item-level response data is from Polo et al. (2024b), "tinyBenchmarks: evaluating LLMs with fewer examples" (ICML 2024).

## Getting Started

```bash
# Verify the IRT engine works on synthetic data (~5 seconds)
python tests/demo_test.py --fast

# Verify on real data (requires data/input/ to be populated)
python tests/demo_test.py --full

# Run a single chain calibration experiment
python src/chain_experiment.py --help
```

## Reproducing Experiments

List available experiments:

```bash
python scripts/run_experiments.py --list
```

Run an experiment (preview first, then execute):

```bash
python scripts/run_experiments.py --experiment lb_anchor_sweep --dry-run
python scripts/run_experiments.py --experiment lb_anchor_sweep
```

Available experiments: `lb_anchor_sweep`, `lb_refmodel_sweep`, `mmlu_anchor_sweep`, `mmlu_refmodel_sweep`.

Results are written to `data/output/`. On a SLURM cluster, add `--dispatch sbatch` to submit one job per run.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{habba2026growing,
  title={Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration},
  author={Habba, Eliya and Itzhak, Itay and Yehudai, Asaf and Perlitz, Yotam and Bandel, Elron and Shmueli-Scheuer, Michal and Choshen, Leshem and Stanovsky, Gabriel},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
