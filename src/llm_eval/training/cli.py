# from __future__ import annotations
#
# import typer
# import pandas as pd
# from pathlib import Path
#
# from llm_eval.matrix import MatrixStorage
# from llm_eval.selection.tinyBenchmarks.training import TrainingConfig
# from .irt import (
#     train_item_parameters,
#     select_anchors,
#     save_item_parameters,
#     save_anchors,
# )
#
#
# app = typer.Typer(add_completion=False, help="Training stage: fit IRT parameters and compute anchors.")
#
#
# @app.command()
# def train(
#     matrix: str = typer.Option(..., help="Path to matrix parquet"),
#     out_params: str = typer.Option(..., help="Output parquet for item parameters (a,b per question_id)"),
#     out_anchors: str = typer.Option(..., help="Output JSON for anchors list"),
#     model_type: str = typer.Option("multidim_2pl"),
#     num_epochs: int = typer.Option(2000),
#     seed: int = typer.Option(42),
#     dims: int = typer.Option(10),
#     lr: float = typer.Option(0.1),
#     lr_decay: float = typer.Option(0.9999),
#     dropout: float = typer.Option(0.5),
#     hidden: int = typer.Option(100),
#     priors: str = typer.Option("hierarchical"),
#     deterministic: bool = typer.Option(True),
#     log_every: int = typer.Option(200),
#     device: str | None = typer.Option(None),
#     number_items: int = typer.Option(100, help="Number of anchor items per dataset (from notebook)"),
# ):
#     df = MatrixStorage(matrix).load()
#     cfg = TrainingConfig(
#         model_type=model_type,
#         num_epochs=num_epochs,
#         seed=seed,
#         dims=dims,
#         lr=lr,
#         lr_decay=lr_decay,
#         dropout=dropout,
#         hidden=hidden,
#         priors=priors,
#         deterministic=deterministic,
#         log_every=log_every,
#         device=device,
#     )
#     params = train_item_parameters(df, config=cfg)
#     save_item_parameters(params, out_params)
#     anchors = select_anchors(params, number_items=number_items)
#     save_anchors(anchors, out_anchors)
#     typer.echo(f"Saved item params -> {out_params}; anchors -> {out_anchors}")
#
#
# if __name__ == "__main__":
#     app()
#
#
