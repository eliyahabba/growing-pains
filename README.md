# Growing Pains: Psychometric Scale-Linking for LLM Benchmarking

Experiment code for the paper *Growing Pains: Psychometric Scale-Linking for Efficient and Continuous LLM Benchmarking*.

## Repository layout

```
config/          sweep parameters and dataset configs
scripts/         run_sweep.py — Python orchestrator (replaces shell scripts)
src/
  irt/           self-contained IRT library (2PL / MIRT training, anchor selection)
  experiments/
    chain_linking/   main sweep logic
    equating/        cross-dataset linking functions used by chain_linking
    utils/           shared I/O helpers
data/
  input/         place benchmark pickles/parquets here (not tracked in git)
  output/        experiment outputs written here (not tracked in git)
tests/
  demo_test.py   synthetic fast test + optional real-data full test
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

Place benchmark response files under `data/input/` (see `data/input/README.md`).

## Running sweeps

```bash
python scripts/run_sweep.py --help
python scripts/run_sweep.py --category 1a --dry-run   # preview commands
python scripts/run_sweep.py --category 1a             # run anchor-count sweep on LB
```

Sweep grids (datasets, anchor counts, model counts, seeds) are defined in `config/constants.py`.

## Tests

```bash
# Fast: trains a tiny IRT on synthetic data (~5 s)
python tests/demo_test.py --fast

# Full: loads real data if present in data/input/
python tests/demo_test.py --full
```

## License

See parent repository.
