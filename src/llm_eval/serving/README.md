### Serving (API & CLI)

Minimal API (FastAPI) and CLI commands for selection, scoring, and snapshots.

API run examples
----------------

Start the server:
```bash
uvicorn llm_eval.serving.api:app --reload
```

Select questions:
```bash
curl -s -X POST http://127.0.0.1:8000/select_questions \
  -H 'Content-Type: application/json' \
  -d '{
    "method": "irt",
    "k": 2,
    "model_name": "new",
    "matrix_path": "data/processed/matrix.parquet"
  }'
```

Create snapshot:
```bash
curl -s -X POST http://127.0.0.1:8000/snapshot \
  -H 'Content-Type: application/json' \
  -d '{
    "matrix_path": "data/processed/matrix.parquet",
    "out_path": "data/processed/snapshots.parquet"
  }'
```

Score model (aggregate + leaderboard):
```bash
curl -s -X POST http://127.0.0.1:8000/score_model \
  -H 'Content-Type: application/json' \
  -d '{
    "matrix_path": "data/processed/matrix.parquet",
    "out_path": "data/processed/snapshots.parquet"
  }'
```

Python (HTTPX):
```python
import httpx

client = httpx.Client()
resp = client.post("http://127.0.0.1:8000/select_questions", json={
  "method": "irt",
  "k": 2,
  "model_name": "new",
  "matrix_path": "data/processed/matrix.parquet",
})
print(resp.json())
```

CLI reference
-------------
- ingest: `llm-eval ingest --source local_files --path examples/tiny_dataset.csv --out data/processed/matrix.parquet`
- select: `llm-eval select --method irt --k 5 --model-name new --matrix data/processed/matrix.parquet --out out/selection.json`
- score: `llm-eval score --matrix data/processed/matrix.parquet --out data/processed/snapshots.parquet`
- simulate: `llm-eval simulate --method irt --k 8 --model-name candidate --matrix data/processed/matrix.parquet`
