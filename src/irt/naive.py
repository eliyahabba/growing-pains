

import pandas as pd

from irt.interfaces import QuestionSelector, ModelProfile


class NaiveVarianceSelector(QuestionSelector):
    """Select questions with highest score variance across models."""

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        grouped = matrix_df.groupby("question_id")["normalized_score"].agg("var").fillna(0.0)
        top = grouped.sort_values(ascending=False).head(k)
        return list(top.index)


