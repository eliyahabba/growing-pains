# TinyBenchmarks Training - AdaptEval Integration

This document explains how the TinyBenchmarks training methodology has been successfully adapted to work with the AdaptEval system's data structure and workflow.

## Integration with AdaptEval

The implementation successfully integrates TinyBenchmarks methodology with AdaptEval's pipeline:

- ✅ **Native Integration**: Works directly with AdaptEval's matrix format
- ✅ **Hierarchical Dataset Support**: Automatically detects and handles subscenarios (e.g., `legalbench.xxx`, `mmlu.xxx`)
- ✅ **Binary Data Handling**: Smart detection of already-binary data vs. continuous data needing thresholding
- ✅ **Pipeline Compatible**: Seamlessly integrates with `run_evaluation_pipeline.py`

## Core Algorithmic Steps

The implementation follows these key algorithmic steps from the notebook:

### 1. **Balance Weights Computation**
- **Purpose**: Give equal importance to subscenarios in multi-subscenario datasets
- **AdaptEval Implementation**: Automatically detects hierarchical datasets using naming patterns (e.g., `legalbench.abercrombie`)
- **Logic**: Apply formula `N/(n_sub*n_i)` for detected hierarchical datasets
- **Real Example**: Your data has `legalbench` (5 subscenarios), `math` (7 subscenarios), `mmlu` (5 subscenarios), `wmt_14` (5 subscenarios)

### 2. **Smart Binarization**  
- **Purpose**: Ensure data is in binary format for IRT training
- **AdaptEval Implementation**: Detects if data is already binary (0.0, 1.0) and skips thresholding
- **Logic**: For continuous data, finds optimal threshold per dataset
- **Real Example**: Your data is already binary, so no thresholding is needed

### 3. **Cross-Validation for Dimension Selection**
- **Purpose**: Choose optimal IRT model dimension
- **Logic**: Split models into train/val, use half questions as 'seen', evaluate on 'unseen'
- **Generalization**: Works with any number of models and questions

### 4. **IRT Model Training**
- **Purpose**: Learn item difficulty and discrimination parameters
- **Logic**: Use py-irt with optimal dimension from validation
- **Generalization**: Handles any question structure

### 5. **Lambda Computation for Blending**
- **Purpose**: Compute blending weights for anchor-IRT predictions
- **Logic**: `λ = b²/(v + b²)` where b=validation_error, v=variance
- **Generalization**: Computed per-dataset or globally

## Configuration

`TrainingConfig` maintains the core parameters while being flexible:

```python
@dataclass
class TrainingConfig:
    dims_search: list[int] = [5, 10]  # Dimensions to validate
    device: str = 'cuda'  # Training device
    epochs: int = 2000  # Training epochs
    lr: float = 0.1  # Learning rate
    val_stride: int = 5  # Validation stride over models
    number_item_per_scenario: int = 100  # For lambda scaling
```

## Input Data Flexibility

The implementation is designed to work with various data structures:

### Required Columns
- `model_name`: Identifier for each model/system
- `question_id`: Identifier for each question/item  
- `normalized_score`: Performance score (0.0-1.0)

### Optional Columns  
- `dataset`: Dataset name (enables per-dataset processing)
- `subscenario`: Sub-dataset name (enables balance weights)

### Supported Scenarios
1. **Single dataset, no subscenarios**: Uniform processing
2. **Multiple datasets, no subscenarios**: Per-dataset thresholds and lambda
3. **Multiple datasets with subscenarios**: Full balance weights + per-dataset processing

## Example Usage with AdaptEval Data

```python
import pandas as pd
from training import fit_2pl_parameters, TrainingConfig

# Load your AdaptEval matrix (already processed through the pipeline)
matrix_df = pd.read_parquet("src/data/processed/matrix_train.parquet")

# Configure training for your specific needs
config = TrainingConfig(
    dims_search=[5, 10],      # Standard dimensions from TinyBenchmarks
    epochs=2000,              # Full training  
    device='cuda',            # Use GPU if available
    val_stride=5,             # Use every 5th model for validation
    number_item_per_scenario=100  # For lambda scaling
)

# Train IRT model following TinyBenchmarks methodology
item_params = fit_2pl_parameters(matrix_df, config)

# Results include trained parameters + metadata
print(f"Best dimension: {item_params.attrs['best_dimension']}")
print(f"Lambda values: {item_params.attrs['lambdas_by_dataset']}")
print(f"Detected hierarchical datasets with balance weights applied")

# Your data structure:
# - 27 datasets (5 hierarchical: legalbench, math, mmlu, wmt_14)
# - Already binary data (0.0, 1.0)
# - 61 models, 11,975 questions
```

## Key Implementation Details

### Balance Weights
- Automatically detects multi-subscenario datasets via `subscenario` column
- Applies notebook formula `N/(n_sub*n_i)` only where needed
- Defaults to uniform weights (1.0) for simple datasets

### Binarization  
- Optimizes threshold per dataset using 100 candidate values
- Minimizes difference between binary and continuous model averages
- Handles missing data gracefully

### Dimension Validation
- Cross-validates over user-specified dimensions
- Uses every 5th model for validation (configurable)
- Evaluates on unseen questions (every other question)
- Supports any number of models/questions

### Lambda Computation
- Computes per-dataset if `dataset` column present
- Falls back to global computation otherwise
- Uses validation errors and dataset variance
- Applies notebook scaling factor

## Testing

Run the generalized training tests:

```bash
cd src/llm_eval/selection/tinyBenchmarks/
python test_notebook_compatibility.py
```

## CLI Usage

The CLI works with any compatible dataset:

```bash
# Train IRT model on your data
python -m main train-irt \
    --matrix path/to/your_matrix.parquet \
    --out path/to/item_params.parquet \
    --dims-search "5,10" \
    --device cuda \
    --epochs 2000
```

## Benefits of This Approach

1. **Data Agnostic**: Works with any evaluation dataset structure
2. **Methodology Faithful**: Preserves core TinyBenchmarks algorithms  
3. **Robust**: Handles edge cases and missing data gracefully
4. **Extensible**: Easy to add new features while maintaining compatibility
5. **Testable**: Clear separation of concerns enables targeted testing

This implementation gives you the power of TinyBenchmarks methodology without being locked into their specific dataset structure.
