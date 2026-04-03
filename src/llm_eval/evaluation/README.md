### Evaluation & Validation

The evaluation module provides comprehensive validation of question selection methods by comparing performance on selected questions vs. full evaluation.

**Key Components:**

#### Metrics (`metrics.py`)
- `EvaluationMetrics`: Dataclass containing RMSE, MAE, correlation, bias, relative error
- `compute_evaluation_metrics()`: Compare partial vs full scores with comprehensive metrics
- `compute_model_performance()`: Calculate aggregated performance for model on dataset/questions
- `estimate_error_to_full_eval()`: Legacy RMSE calculation (backward compatibility)

#### Selection Validator (`validator.py`)
- `SelectionValidator`: Main validation orchestrator
- `SelectionValidationResult`: Results for single validation run
- `ValidationSummary`: Aggregated results across multiple validations

#### Simulation (`simulation.py`)
- `simulate_selection_impact()`: Basic coverage analysis for selected questions
- `SimulationResult`: Coverage and selection results

**Usage Examples:**

```python
# Individual validation
from src.llm_eval.evaluation import SelectionValidator
from src.llm_eval.selection import IRT2PLSelector

validator = SelectionValidator(matrix_df)
result = validator.validate_selection(
    IRT2PLSelector(), "irt", "gpt-4", k=10, dataset_name="mmlu"
)
print(f"RMSE: {result.metrics.rmse:.3f}")
print(f"Correlation: {result.metrics.correlation:.3f}")
print(f"Bias: {result.metrics.bias:.3f}")

# Comprehensive validation
selectors = {"irt": IRT2PLSelector(), "naive": NaiveVarianceSelector()}
summary = validator.validate_all_models(selectors, k=10)
avg_metrics = summary.get_average_metrics()
results_df = summary.to_dataframe()

# Compare different k values for a model
comparison = validator.compare_selectors(
    selectors, k_values=[5, 10, 20], model_name="gpt-4"
)
```

**CLI Commands:**

```bash
# Validate single selector
llm-eval validate --matrix data.parquet --method irt --k 10 --model-name gpt-4

# Comprehensive validation
llm-eval validate-all --matrix data.parquet --k 10 --out results.csv

# Original simulation (coverage only)
llm-eval simulate --matrix data.parquet --method irt --k 8 --model-name candidate
```

**Output Metrics:**

- **RMSE**: Root Mean Square Error between selected and full scores
- **MAE**: Mean Absolute Error
- **Correlation**: Pearson correlation coefficient
- **Bias**: Systematic over/under-estimation (mean difference)
- **Relative Error**: RMSE as percentage of full score range
- **Coverage**: Percentage of total questions selected (from simulation)

The validation system helps answer: *"Do our selected questions provide a good estimate of full evaluation performance?"*




