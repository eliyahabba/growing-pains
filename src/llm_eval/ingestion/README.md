### Ingestion

Pods for data ingestion. Implement `IngestionSource` to add new sources.

Flow:
1. Source loads a raw DataFrame (long-format fields as available).
2. Normalization happens later in the pipeline (not here).

Included:
- `LocalCSVSource`: reads a CSV with the expected columns.
- `HuggingFaceDatasetSource`: stub for future integration.

Run examples
------------

Python (load CSV):
```python
from src.llm_eval.ingestion import LocalCSVSource

df = LocalCSVSource("examples/tiny_dataset.csv").load()
print(df.head())
```

CLI (ingest + save normalized matrix):
```bash
llm-eval ingest --source local_files --path examples/tiny_dataset.csv --out data/processed/matrix.parquet
```

CLI (merge multiple sources → unified matrix):
```bash
llm-eval ingest-multi \
  csv:examples/tiny_dataset.csv \
  hf:boolq:validation \
  --out data/processed/matrix.parquet \
  --dump-raw-dir data/raw_sources
```


