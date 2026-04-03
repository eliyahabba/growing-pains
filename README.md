# Growing Pains: Psychometric Scale-Linking for LLM Benchmarking

Publication-ready extraction of the chain-linking experiment pipeline.

## Setup

```bash
cd growing-pains
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
# IRT training: uv pip install torch py-irt pyro-ppl
unset PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/src"
```

Sweep grids live in `config/constants.py`. If your shell sets `PYTHONPATH` globally, clear it before running (`unset PYTHONPATH`).

Place TinyBenchmarks pickles under `aggregated_data/tinybenchmarks/` and aggregated parquets under `aggregated_data/aggregated/` (see `aggregated_data/README.md`).

## Sweeps

Sweep grids and seeds are defined in `config/constants.py`. Run `python scripts/run_sweep.py --help`.

## Tests

```bash
python tests/demo_test.py --fast
```

## License

Same as parent AdaptEval project.
