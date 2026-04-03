### Normalization

**This is the centralized normalization system for the entire pipeline.**

The normalization system supports two approaches:

1. **Standard Normalization**: Per-score normalization to [0,100] range
2. **IRT Normalization**: Per-scenario optimal thresholds for binary classification [0,1]

**Note**: All downstream modules (including `tinyBenchmarks.training`) expect to receive 
pre-normalized data from this system. No additional normalization should be performed elsewhere.

## Standard Normalization

Metric registry maps `metric_name` to normalization strategy, producing `normalized_score ∈ [0,100]`.

Rules:
- Min-max with configured `min/max` and `higher_is_better`.
- Z-score→CDF requires `mean/std` via metadata. No generic fallback: metrics must be explicitly configured.

Configure per-metric in `src/llm_eval/config/metrics.yaml`.

## IRT Normalization

For IRT (Item Response Theory) models, we provide advanced binarization methods that optimize thresholds per scenario to preserve mean scores across models.

### Methods

**Direct Binarization (`'direct'`)**:
- Finds optimal thresholds in original score space
- Handles different score ranges per scenario naturally
- No normalization step - works with raw score distributions

**Normalized Binarization (`'normalized'`)**:
- First normalizes scores to [0,1] using statistics from ALL models
- Then finds optimal threshold in normalized space
- Makes thresholds more comparable across scenarios

### Key Features

Both methods follow the original implementation philosophy:
- **Per-scenario optimization**: Each scenario gets its own threshold
- **Mean preservation**: Thresholds minimize difference between original and binarized means
- **Model consistency**: Same threshold applies to all models within a scenario

### Usage

#### In MatrixBuilder

```python
from llm_eval.config import load_yaml_config
from llm_eval.normalization import MetricRegistry
from llm_eval.matrix import MatrixBuilder

cfg = load_yaml_config("defaults.yaml", "metrics.yaml")
registry = MetricRegistry(cfg)

# Standard normalization (default)
builder = MatrixBuilder(registry)

# IRT normalization with direct method
builder = MatrixBuilder(registry, use_irt_normalization=True, irt_method='direct')

# IRT normalization with normalized method
builder = MatrixBuilder(registry, use_irt_normalization=True, irt_method='normalized')

matrix_df = builder.build(raw_df)
```

#### In Pipeline

```python
# Standard normalization
python src/run_evaluation_pipeline.py

# IRT normalization with direct method
python src/run_evaluation_pipeline.py --irt

# IRT normalization with normalized method  
python src/run_evaluation_pipeline.py --irt --normalized
```

Run examples
------------

Standard normalization:
```python
from src.llm_eval.config import load_yaml_config
from src.llm_eval.normalization import MetricRegistry

cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
reg = MetricRegistry(cfg)
res = reg.normalize("binary_acc", raw_score=1.0, metadata={})
print(res.normalized)
```

IRT normalization demo:
```python
# Run the comprehensive demonstration
python examples/irt_normalization_demo.py
```