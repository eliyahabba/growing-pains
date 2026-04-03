from __future__ import annotations

from pathlib import Path
import json
import typer
import pandas as pd

from llm_eval.matrix import MatrixStorage
from training import fit_2pl_parameters, TrainingConfig
from anchors import find_anchor_items, AnchorConfig
from estimation import estimate_theta_from_anchors, expected_correctness, blend_anchor_and_irt, EstimationConfig


app = typer.Typer(add_completion=False, help="TinyBenchmarks workflow: train IRT, find anchors, estimate.")


@app.command()
def train_irt(
    matrix: str = typer.Option(..., help="Path to matrix Parquet produced by MatrixBuilder"),
    out: str = typer.Option(..., help="Output path for learned item params Parquet (a,b per question_id)"),
    dims_search: str = typer.Option("5,10", help="Comma-separated list for D search, e.g. '5,10'"),
    device: str = typer.Option("cuda", help="'cuda' or 'cpu'"),
    epochs: int = typer.Option(2000, help="Number of training epochs"),
    lr: float = typer.Option(0.1, help="Learning rate"),
    random_state: int = typer.Option(42, help="Random seed"),
    val_stride: int = typer.Option(5, help="Validation stride over models for D search"),
    number_item_per_scenario: int = typer.Option(100, help="For lambda heuristic like notebook"),
):
    """Train IRT model exactly following the TinyBenchmarks notebook workflow."""
    df = MatrixStorage(matrix).load()
    cfg = TrainingConfig(
        dims_search=[int(x) for x in dims_search.split(",")],
        device=device,
        epochs=epochs,
        lr=lr,
        random_state=random_state,
        val_stride=val_stride,
        number_item_per_scenario=number_item_per_scenario,
    )
    params = fit_2pl_parameters(df, cfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params.to_parquet(out_path)
    
    # Print summary information
    typer.echo(f"Saved item parameters to {out}")
    if hasattr(params, 'attrs'):
        if 'best_dimension' in params.attrs:
            typer.echo(f"Best dimension: {params.attrs['best_dimension']}")
        if 'lambdas_by_dataset' in params.attrs:
            typer.echo(f"Lambda values: {params.attrs['lambdas_by_dataset']}")
    typer.echo(f"Trained on {len(params)} items")


@app.command()
def anchors(
    item_params: str = typer.Option(..., help="Path to item params Parquet (from train-irt)"),
    out: str = typer.Option(..., help="Output JSON path with anchors list"),
    number_items: int = typer.Option(100, help="Total number of anchor items (from notebook)"),
    method: str = typer.Option("irt_clustering", help="Selection method: 'irt_clustering', 'correctness_clustering', or 'difficulty_binning'"),

):
    params = pd.read_parquet(item_params)
    
    # Extract balance weights from training metadata if available
    balance_weights = None
    if hasattr(params, 'attrs') and 'balance_weights' in params.attrs:
        balance_weights = params.attrs['balance_weights']
        typer.echo(f"Using balance weights from training metadata")
    
    # Configure based on method
    acfg = AnchorConfig(
        method=method,
        number_items=number_items,
        balance_weights=balance_weights
    )
    
    # Use per-dataset selection if dataset column is available
    if "dataset" in params.columns:
        from anchors import find_anchor_items_by_dataset
        # For main.py, we don't have matrix_df, so pass None (works for irt_clustering and difficulty_binning)
        anchors_by_dataset = find_anchor_items_by_dataset(params, "dataset", number_items, method, matrix_df=None)
        # Combine all anchors from all datasets
        anchor_ids = []
        for dataset, anchors in anchors_by_dataset.items():
            anchor_ids.extend(anchors)
    else:
        anchor_ids = find_anchor_items(params, acfg)
    
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"anchors": anchor_ids}, f)
    typer.echo(f"Saved {len(anchor_ids)} anchors to {out} (method: {method})")


@app.command()
def estimate(
    item_params: str = typer.Option(..., help="Path to item params Parquet (from train-irt)"),
    anchors_path: str = typer.Option(..., help="Path to anchors JSON (from anchors)"),
    out: str = typer.Option(..., help="Output Parquet with predictions per question_id and theta"),
    # Option A: Provide anchor responses directly
    anchor_responses_csv: str | None = typer.Option(None, help="CSV with columns: question_id,response(0/1)"),
    # Option B: Derive anchor responses from matrix for a specific model
    matrix: str | None = typer.Option(None, help="Path to matrix Parquet containing the candidate model"),
    model_name: str | None = typer.Option(None, help="Candidate model name in the matrix"),
    threshold: float = typer.Option(50.0, help="Binarization threshold when deriving responses from matrix"),
    blend_lambda: bool = typer.Option(False, help="Blend anchor and IRT predictions using gp-IRT style"),
):
    params = pd.read_parquet(item_params)
    lambdas_by_dataset = params.attrs.get("lambdas_by_dataset", {}) if hasattr(params, "attrs") else {}
    with open(anchors_path, "r") as f:
        data = json.load(f)
    anchor_ids = [str(x) for x in data.get("anchors", [])]
    if not anchor_ids:
        raise typer.BadParameter("No anchors found in anchors JSON")

    # Prepare anchor responses series
    if anchor_responses_csv:
        ar = pd.read_csv(anchor_responses_csv)
        if not {"question_id", "response"}.issubset(ar.columns):
            raise typer.BadParameter("anchor_responses_csv must have columns question_id,response")
        responses = pd.Series(ar["response"].astype(int).values, index=ar["question_id"].astype(str).values)
    else:
        if matrix is None or model_name is None:
            raise typer.BadParameter("Provide either anchor_responses_csv or both matrix and model_name")
        df = MatrixStorage(matrix).load()
        sub = df[(df["model_name"].astype(str) == str(model_name)) & (df["question_id"].astype(str).isin(anchor_ids))]
        if sub.empty:
            raise typer.BadParameter("No matching anchor responses found for the given model_name in matrix")
        correct = (sub["normalized_score"].astype(float) >= float(threshold)).astype(int)
        responses = pd.Series(correct.values, index=sub["question_id"].astype(str).values)

    theta = estimate_theta_from_anchors(params, responses)
    preds_irt = expected_correctness(params, theta)
    # Simple anchor-only estimate per item: project anchor mean to all items (baseline)
    anchor_mean = float(responses.astype(float).mean()) if len(responses) else 0.5
    preds_anchor = pd.Series(anchor_mean, index=preds_irt.index)
    if blend_lambda and lambdas_by_dataset:
        item_to_dataset = None
        # If item_params preserved dataset column, use it; otherwise skip
        if "dataset" in params.columns:
            item_to_dataset = params["dataset"]
        preds = blend_anchor_and_irt(preds_anchor, preds_irt, lambdas_by_dataset, item_to_dataset)
    else:
        preds = preds_irt
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"question_id": preds.index.astype(str), "predicted_p": preds.values})
    res.to_parquet(out_path)
    typer.echo(f"Estimated theta={theta:.4f}; saved predictions to {out}")


if __name__ == "__main__":
    app()














