# Growing Pains: Psychometric Scale-Linking for LLM Benchmarking

Experiment code for the paper *Growing Pains: Psychometric Scale-Linking for Efficient and Continuous LLM Benchmarking*.

## Repository layout

```
config/
  constants.py          sweep grids, dataset lists, seeds
  data_source_config.json
  experiment_presets.yaml

src/
  irt/                  IRT engine: 2PL/MIRT training, anchor selection, theta estimation
  chain_experiment.py   entry point — runs one chain calibration experiment
  data_loading.py       dataset loading and grouping by skill/source
  calibration.py        base frame calibration, fixed-anchor calibration, anchor selection
  evaluation.py         run_validation and baseline routines (random, discriminative)
  io.py                 shared rounding helpers

scripts/
  run_sweep.py          Python orchestrator for all anchor/model-count sweeps

data/
  input/                place benchmark pickles/parquets here (not in git)
  output/               experiment results written here (not in git)

tests/
  demo_test.py          fast synthetic IRT test + optional real-data test
```

## Setup

```bash
cd /Users/ehabba/PycharmProjects/growing-pains
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
unset PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/src"
```

## Input data

Place benchmark response files under `data/input/` — see `data/input/README.md`.

## Running sweeps

```bash
python scripts/run_sweep.py --help
python scripts/run_sweep.py --category 1a --dry-run   # preview commands
python scripts/run_sweep.py --category 1a             # run LB anchor-count sweep
```

Sweep grids are defined in `config/constants.py`.

## Tests

```bash
# Fast: trains a tiny IRT on synthetic data (~5 s)
python tests/demo_test.py --fast

# Full: loads real data if present in data/input/
python tests/demo_test.py --full
```

## License

See parent repository.
