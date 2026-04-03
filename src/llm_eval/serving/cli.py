

from pathlib import Path
import json
import typer
import pandas as pd

from llm_eval.config import load_yaml_config
from llm_eval.utils import get_logger, read_parquet_safely
from llm_eval.ingestion import LocalCSVSource
from llm_eval.ingestion.hf_datasets import HuggingFaceDatasetSource  # type: ignore
from llm_eval.normalization import MetricRegistry
from llm_eval.matrix import MatrixBuilder, MatrixStorage
from llm_eval.selection import NaiveVarianceSelector, MITVSelector, ModelProfile, TinyBenchmarksSelector
from llm_eval.scoring import SnapshotManager, Leaderboard
from llm_eval.evaluation import simulate_selection_impact, SelectionValidator


app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


@app.command()
def ingest(source: str = typer.Option("local_files"), path: str = typer.Option(...), out: str = typer.Option(...)):  # noqa: E501
    cfg = load_yaml_config(
        str(Path(__file__).parents[1] / "config" / "defaults.yaml"),
        str(Path(__file__).parents[1] / "config" / "metrics.yaml"),
    )
    if source != "local_files":
        raise typer.BadParameter("Only local_files supported in MVP")
    src = LocalCSVSource(path)
    raw = src.load()
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry)
    matrix = builder.build(raw)
    MatrixStorage(out).save(matrix)
    logger.info("Ingested and saved to %s", out)


@app.command()
def select(method: str = typer.Option("naive"), k: int = typer.Option(10), model_name: str = typer.Option(...), matrix: str = typer.Option(...), out: str = typer.Option(...)):  # noqa: E501
    df = MatrixStorage(matrix).load()
    profile = ModelProfile(model_name=model_name)
    if method == "naive":
        selector = NaiveVarianceSelector()
    elif method in {"irt", "py-irt", "py_irt", "tinyb", "tinybench", "tiny-benchmarks", "tiny_benchmarks"}:
        if TinyBenchmarksSelector is None:
            raise typer.BadParameter("TinyBenchmarks selector not available")
        selector = TinyBenchmarksSelector()
    elif method in {"mitv", "interview"}:
        selector = MITVSelector()
    else:
        raise typer.BadParameter("Unknown method")
    questions = selector.select(profile, k, df)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"questions": questions}, f)
    logger.info("Selection written to %s", out)


@app.command()
def ingest_multi(
    inputs: list[str] = typer.Argument(..., help="List of sources: csv:PATH or hf:DATASET[:SPLIT]"),
    out: str = typer.Option(..., help="Output Parquet path for unified matrix"),
    dump_raw_dir: str | None = typer.Option(None, help="Optional directory to dump per-source raw CSVs"),
):
    """Ingest multiple sources (CSV/HF), merge, normalize and write a unified matrix.

    Examples:
      llm-eval ingest-multi csv:examples/tiny_dataset.csv hf:boolq:validation --out data/processed/matrix.parquet
    """
    cfg = load_yaml_config(
        str(Path(__file__).parents[1] / "config" / "defaults.yaml"),
        str(Path(__file__).parents[1] / "config" / "metrics.yaml"),
    )
    raw_frames: list[pd.DataFrame] = []
    for item in inputs:
        if item.startswith("csv:"):
            path = item.split(":", 1)[1]
            df = LocalCSVSource(path).load()
            if dump_raw_dir:
                Path(dump_raw_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(df).to_csv(Path(dump_raw_dir) / (Path(path).stem + "_raw.csv"), index=False)
            raw_frames.append(df)
        elif item.startswith("hf:"):
            if HuggingFaceDatasetSource is None:
                raise typer.BadParameter("HF source requested but optional dependencies not installed. Install with extras: pip install .[hf]")
            rest = item.split(":", 1)[1]
            if ":" in rest:
                ds_name, split = rest.split(":", 1)
            else:
                ds_name, split = rest, "test"
            df = HuggingFaceDatasetSource(ds_name, split=split).load()
            if dump_raw_dir:
                Path(dump_raw_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(df).to_csv(Path(dump_raw_dir) / (f"{ds_name}_{split}_raw.csv"), index=False)
            raw_frames.append(df)
        else:
            raise typer.BadParameter(f"Unrecognized input format: {item}")

    if not raw_frames:
        raise typer.BadParameter("No valid inputs provided")

    merged = pd.concat(raw_frames, ignore_index=True, sort=False)
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry)
    matrix = builder.build(merged)
    MatrixStorage(out).save(matrix)
    logger.info("Ingested %d sources and saved unified matrix to %s", len(raw_frames), out)


@app.command()
def ingest_parquet(parquet: str = typer.Option(..., help="Path to aggregated Parquet (e.g., HELM)"), out: str = typer.Option(..., help="Output matrix Parquet path")):  # noqa: E501
    """Ingest an aggregated Parquet file into the standard matrix format.

    Expects columns like: dataset_name, hf_split, hf_index, model_name, model_family,
    evaluation_method_name, evaluation_score.
    """
    cfg = load_yaml_config(
        str(Path(__file__).parents[1] / "config" / "defaults.yaml"),
        str(Path(__file__).parents[1] / "config" / "metrics.yaml"),
    )
    df = read_parquet_safely(parquet)
    # Map input schema → raw long-format expected by MatrixBuilder
    # Required columns: metric_name, raw_score; common context fields mapped when present
    if not {"evaluation_method_name", "evaluation_score"}.issubset(df.columns):
        raise typer.BadParameter("Parquet must include 'evaluation_method_name' and 'evaluation_score'")
    raw = {}
    raw["metric_name"] = df["evaluation_method_name"].astype(str)
    raw["raw_score"] = df["evaluation_score"].astype(float)
    if "dataset_name" in df.columns:
        raw["dataset"] = df["dataset_name"].astype(str)
    if "hf_split" in df.columns:
        raw["split"] = df["hf_split"].astype(str)
    # Build question_id as dataset:split:index when available (split disambiguates identical indices across splits)
    if {"dataset_name", "hf_index", "hf_split"}.issubset(df.columns):
        raw["question_id"] = (
            df["dataset_name"].astype(str)
            + ":"
            + df["hf_split"].astype(str)
            + ":"
            + df["hf_index"].astype(str)
        )
    elif set(["dataset_name", "hf_index"]).issubset(df.columns):
        raw["question_id"] = df["dataset_name"].astype(str) + ":" + df["hf_index"].astype(str)
    elif "hf_index" in df.columns:
        raw["question_id"] = df["hf_index"].astype(str)
    else:
        raw["question_id"] = df.index.astype(str)
    if "model_name" in df.columns:
        raw["model_name"] = df["model_name"].astype(str)
    if "model_family" in df.columns:
        raw["model_family"] = df["model_family"].astype(str)

    import pandas as pd
    raw_df = pd.DataFrame(raw)
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry)
    matrix = builder.build(raw_df)
    MatrixStorage(out).save(matrix)
    logger.info("Ingested Parquet %s and saved matrix to %s", parquet, out)


@app.command()
def compare_selectors(
    matrix: str = typer.Option(..., help="Path to matrix Parquet"),
    k: int = typer.Option(10, help="Number of questions to select"),
    model_name: str = typer.Option("candidate", help="Target model name"),
    out: str = typer.Option(None, help="Optional JSON output path"),
    methods = typer.Option(None, help="Subset of methods, e.g. --methods naive --methods irt"),
):
    """Run multiple selection methods uniformly on an existing matrix and compare outputs."""
    df = MatrixStorage(matrix).load()
    profile = ModelProfile(model_name=model_name)

    available = {
        "naive": NaiveVarianceSelector(),
        "mitv": MITVSelector(),
    }
    # Register optional selectors under canonical keys
    if TinyBenchmarksSelector is not None:
        available["irt"] = TinyBenchmarksSelector()

    # Normalize method aliases to canonical keys
    alias_map = {
        "py_irt": "irt",
        "py-irt": "irt",
        "tinyb": "irt",
        "tiny-benchmarks": "irt",
        "tiny_benchmarks": "irt",
        "tinybench": "irt",
        "irt": "irt",
    }

    selected = methods if methods else list(available.keys())
    selected = [alias_map.get(m, m) for m in selected]
    results: dict[str, list[str]] = {}
    for name in selected:
        if name not in available:
            raise typer.BadParameter(f"Unknown or unavailable method: {name}")
        sel = available[name]
        results[name] = sel.select(profile, k, df)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f)
        logger.info("Comparison written to %s", out)
    else:
        typer.echo(json.dumps(results, indent=2))


@app.command()
def score(matrix: str = typer.Option(...), out: str = typer.Option(...)):
    df = MatrixStorage(matrix).load()
    snap = SnapshotManager(out)
    snapshot_id = snap.create_snapshot(df)
    scores = snap.aggregate_scores(df)
    lb = Leaderboard().from_scores(scores)
    MatrixStorage(out).save(df.assign(snapshot_id=snapshot_id))
    logger.info("Leaderboard top:\n%s", lb.head().to_string(index=False))


@app.command()
def simulate(matrix: str = typer.Option(...), method: str = typer.Option("irt"), k: int = typer.Option(8), model_name: str = typer.Option("candidate")):
    df = MatrixStorage(matrix).load()
    profile = ModelProfile(model_name=model_name)
    if method in {"irt", "py-irt", "py_irt", "tinyb", "tinybench"}:
        if TinyBenchmarksSelector is None:
            # This should ideally not be hit if deps are managed, but good practice
            raise typer.BadParameter("IRT selector requested but optional dependencies not installed. Install with extras: pip install .[irt]")
        selector = TinyBenchmarksSelector()
    elif method in {"mitv", "interview"}:
        selector = MITVSelector()
    else:
        selector = NaiveVarianceSelector()
    res = simulate_selection_impact(selector, profile, k, df)
    logger.info("Selected %d questions (coverage=%.2f)", len(res.selected_questions), res.coverage)


@app.command()
def validate(
    matrix: str = typer.Option(..., help="Path to matrix Parquet"),
    method: str = typer.Option("irt", help="Selection method to validate"),
    k: int = typer.Option(10, help="Number of questions to select"),
    model_name: str = typer.Option("candidate", help="Target model name"),
    dataset: str = typer.Option(None, help="Specific dataset to validate on"),
    out: str = typer.Option(None, help="Optional CSV output path for results"),
):
    """Validate a selection method by comparing full vs selected question performance."""
    df = MatrixStorage(matrix).load()
    
    # Get selector
    available = {
        "naive": NaiveVarianceSelector(),
        "mitv": MITVSelector(),
    }
    if TinyBenchmarksSelector is not None:
        available["irt"] = TinyBenchmarksSelector()
        available["py-irt"] = TinyBenchmarksSelector() # for backwards compatibility
        
    if method not in available:
        raise typer.BadParameter(f"Unknown method: {method}. Available: {list(available.keys())}")
    
    selector = available[method]
    validator = SelectionValidator(df)
    
    result = validator.validate_selection(selector, method, model_name, k, dataset)
    
    # Print results
    typer.echo(f"Validation Results for {method} selector:")
    typer.echo(f"  Model: {result.model_name}")
    typer.echo(f"  Dataset: {result.dataset_name}")
    typer.echo(f"  Questions selected: {len(result.selected_questions)}")
    typer.echo(f"  Full score: {result.full_score:.3f}")
    typer.echo(f"  Selected score: {result.selected_score:.3f}")
    typer.echo(f"  RMSE: {result.metrics.rmse:.3f}")
    typer.echo(f"  MAE: {result.metrics.mae:.3f}")
    typer.echo(f"  Correlation: {result.metrics.correlation:.3f}")
    typer.echo(f"  Bias: {result.metrics.bias:.3f}")
    typer.echo(f"  Relative Error: {result.metrics.relative_error:.1f}%")
    
    if out:
        import pandas as pd
        result_data = {
            "model_name": [result.model_name],
            "dataset_name": [result.dataset_name],
            "selector_name": [result.selector_name],
            "k": [result.k],
            "n_selected": [len(result.selected_questions)],
            "full_score": [result.full_score],
            "selected_score": [result.selected_score],
            "rmse": [result.metrics.rmse],
            "mae": [result.metrics.mae],
            "correlation": [result.metrics.correlation],
            "bias": [result.metrics.bias],
            "relative_error": [result.metrics.relative_error],
        }
        pd.DataFrame(result_data).to_csv(out, index=False)
        logger.info("Validation results written to %s", out)


@app.command()
def validate_all(
    matrix: str = typer.Option(..., help="Path to matrix Parquet"),
    k: int = typer.Option(10, help="Number of questions to select"),
    datasets = typer.Option(None, help="Specific datasets to validate on"),
    max_models: int = typer.Option(5, help="Maximum number of models to validate"),
    out: str = typer.Option(None, help="Optional CSV output path for results"),
):
    """Run comprehensive validation across all selectors, models, and datasets."""
    df = MatrixStorage(matrix).load()
    
    # Setup selectors
    selectors = {
        "naive": NaiveVarianceSelector(),
        "mitv": MITVSelector(),
    }
    if TinyBenchmarksSelector is not None:
        selectors["irt"] = TinyBenchmarksSelector()
    
    validator = SelectionValidator(df)
    
    # Limit models for performance
    models = df["model_name"].unique()[:max_models]
    dataset_list = datasets if datasets else df["dataset"].unique()[:3]  # Limit datasets too
    
    typer.echo(f"Running validation on {len(models)} models, {len(dataset_list)} datasets, {len(selectors)} selectors...")
    
    all_results = []
    for model_name in models:
        for dataset_name in dataset_list:
            for selector_name, selector in selectors.items():
                try:
                    result = validator.validate_selection(selector, selector_name, model_name, k, dataset_name)
                    all_results.append(result)
                    typer.echo(f"  ✓ {selector_name} | {model_name} | {dataset_name}: RMSE={result.metrics.rmse:.3f}")
                except Exception as e:
                    typer.echo(f"  ✗ {selector_name} | {model_name} | {dataset_name}: {e}")
                    continue
    
    # Generate summary
    from llm_eval.evaluation import ValidationSummary
    summary = ValidationSummary(results=all_results)
    avg_metrics = summary.get_average_metrics()
    
    typer.echo("\n=== VALIDATION SUMMARY ===")
    typer.echo(f"Total validations: {len(all_results)}")
    typer.echo(f"Average RMSE: {avg_metrics.get('avg_rmse', 'N/A'):.3f}")
    typer.echo(f"Average MAE: {avg_metrics.get('avg_mae', 'N/A'):.3f}")
    typer.echo(f"Average Correlation: {avg_metrics.get('avg_correlation', 'N/A'):.3f}")
    typer.echo(f"Average Bias: {avg_metrics.get('avg_bias', 'N/A'):.3f}")
    
    # Show best performers
    results_df = summary.to_dataframe()
    best_rmse = results_df.groupby('selector_name')['rmse'].mean().sort_values()
    
    typer.echo("\nBest selectors by RMSE:")
    for selector, rmse in best_rmse.items():
        typer.echo(f"  {selector}: {rmse:.3f}")
    
    if out:
        results_df.to_csv(out, index=False)
        logger.info("Validation results written to %s", out)


if __name__ == "__main__":
    app()


