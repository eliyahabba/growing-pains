

from dataclasses import dataclass
import pandas as pd


@dataclass
class Leaderboard:
    def from_scores(self, scores_df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        lb = scores_df.sort_values("cumulative_score", ascending=False)
        if top_k is not None:
            lb = lb.head(top_k)
        return lb.reset_index(drop=True)


