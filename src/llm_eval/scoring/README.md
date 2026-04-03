### Scoring & Snapshots

Time-aware scoring aggregates normalized scores across snapshots using configurable decay.

Components:
- `SnapshotManager`: create snapshots and compute decayed cumulative scores.
- `Leaderboard`: produce sorted leaderboard tables.

Run examples
------------

Python:
```python
from src.llm_eval.matrix import MatrixStorage
from src.llm_eval.scoring import SnapshotManager, Leaderboard

df = MatrixStorage("data/processed/matrix.parquet").load()
snap = SnapshotManager("data/processed/snapshots.parquet")
snapshot_id = snap.create_snapshot(df)
scores = snap.aggregate_scores(df)
lb = Leaderboard().from_scores(scores)
print(snapshot_id, lb.head())
```


