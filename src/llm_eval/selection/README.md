### Selection

Select informative questions for a new model.

Interfaces:
- `QuestionSelector.select(model_profile, k, matrix_df) -> list[str]`

Implementations:
- `TinyBenchmarksSelector`: unified IRT implementation. Uses `py-irt` if available for full statistical modeling, with a fallback to a fast heuristic method. Builds a representative subset of questions.
- `NaiveVarianceSelector`: chooses highest-variance questions across models.
- `MITVSelector`: interview-style with difficulty levels and optional diversity.

Two-stage workflow
------------------

1) Training stage (offline):

```bash
python -m llm_eval.training.cli train \
  --matrix data/processed/matrix.parquet \
  --out-params data/irt/item_params.parquet \
  --out-anchors data/irt/anchors.json
```

This produces reusable artifacts for selection.

2) Selection stage (online):

```python
from llm_eval.selection import TinyBenchmarksSelector, ModelProfile
sel = TinyBenchmarksSelector(
    item_params_path="data/irt/item_params.parquet",
    anchors_path="data/irt/anchors.json",
)
qs = sel.select(ModelProfile(model_name="new"), 10, matrix_df)
```

Run examples
------------

Python:
```python
import pandas as pd
from llm_eval.matrix import MatrixStorage
from llm_eval.selection import TinyBenchmarksSelector, ModelProfile

df = MatrixStorage("data/processed/matrix.parquet").load()
sel = TinyBenchmarksSelector()
qs = sel.select(ModelProfile(model_name="new"), 5, df)
print(qs)
```

CLI:
```bash
# The 'irt' method now points to the TinyBenchmarks implementation
llm-eval select --method irt --k 5 --model-name new --matrix data/processed/matrix.parquet --out out/selection.json
```


