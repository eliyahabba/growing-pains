

from dataclasses import dataclass
import pandas as pd

from llm_eval.selection import QuestionSelector, ModelProfile


@dataclass
class SimulationResult:
    selected_questions: list[str]
    coverage: float


def simulate_selection_impact(selector: QuestionSelector, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> SimulationResult:
    questions = selector.select(model, k, matrix_df)
    unique = matrix_df["question_id"].nunique()
    coverage = len(set(questions)) / float(unique) if unique > 0 else 0.0
    return SimulationResult(selected_questions=questions, coverage=coverage)


