

from pathlib import Path
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from llm_eval.matrix import MatrixStorage
from llm_eval.selection import TinyBenchmarksSelector, NaiveVarianceSelector, ModelProfile
from llm_eval.scoring import SnapshotManager, Leaderboard


app = FastAPI(title="AdaptEval API")


class SelectRequest(BaseModel):
    method: str = "irt"
    k: int = 10
    model_name: str
    matrix_path: str


class SelectResponse(BaseModel):
    questions: List[str]


@app.post("/select_questions", response_model=SelectResponse)
def select_questions(req: SelectRequest) -> SelectResponse:
    df = MatrixStorage(req.matrix_path).load()
    profile = ModelProfile(model_name=req.model_name)
    selector = TinyBenchmarksSelector() if req.method == "irt" else NaiveVarianceSelector()
    questions = selector.select(profile, req.k, df)
    return SelectResponse(questions=questions)


class ScoreRequest(BaseModel):
    matrix_path: str
    out_path: str


@app.post("/score_model")
def score_model(req: ScoreRequest) -> dict:
    df = MatrixStorage(req.matrix_path).load()
    snap = SnapshotManager(req.out_path)
    snapshot_id = snap.create_snapshot(df)
    scores = snap.aggregate_scores(df)
    lb = Leaderboard().from_scores(scores)
    return {"snapshot_id": snapshot_id, "leaderboard": lb.to_dict(orient="records")}


class SnapshotRequest(BaseModel):
    matrix_path: str
    out_path: str


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict:
    df = MatrixStorage(req.matrix_path).load()
    snap = SnapshotManager(req.out_path)
    snapshot_id = snap.create_snapshot(df)
    return {"snapshot_id": snapshot_id}


