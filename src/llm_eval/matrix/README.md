### Matrix

Builds the unified long-format matrix with schema validation via Pydantic models.  
Stores/loads Parquet using Arrow.

Key types:
- `ObservationRow`: typed row model with validators.
- `MatrixBuilder`: applies normalization and returns a DataFrame.
- `MatrixStorage`: Parquet IO.

Run examples
------------

Python:
```python
import pandas as pd
from src.llm_eval.config import load_yaml_config
from src.llm_eval.normalization import MetricRegistry
from src.llm_eval.matrix import MatrixBuilder, MatrixStorage

cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
reg = MetricRegistry(cfg)
builder = MatrixBuilder(reg)
raw = pd.read_csv("examples/tiny_dataset.csv")
matrix = builder.build(raw)
MatrixStorage("data/processed/matrix.parquet").save(matrix)
```


