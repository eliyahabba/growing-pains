

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from irt.interfaces import QuestionSelector, ModelProfile
from irt.cold_start import simple_cold_start_theta


def _entropy_bits(p: float) -> float:
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    q = 1.0 - p
    return -p * math.log2(p) - q * math.log2(q)


@dataclass
class MITVSelector(QuestionSelector):
    """Interview-style selector inspired by MITV.

    This selector assigns a difficulty level (1-10) to each question based on
    cross-model performance in the matrix, then ranks questions by expected
    information at the target model's estimated ability. Expected information
    is approximated by Bernoulli entropy, weighted by proximity of difficulty
    to a target difficulty derived from ability.

    - If a column `difficulty_level` exists, it is used directly (values expected 1..10).
    - Otherwise, levels are computed from mean correctness across models.
    - Optionally, selection can be diversified across a grouping column, e.g. `category`.
    """

    alpha_distance: float = 0.6
    diversify_by: str | None = None
    level_prob_mapping: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        if self.level_prob_mapping is None:
            # Monotonic mapping: easier levels → higher expected correctness.
            # Roughly aligned to examples in the description (L3≈0.7, L5≈0.5, L7≈0.3)
            self.level_prob_mapping = {
                1: 0.90,
                2: 0.80,
                3: 0.70,
                4: 0.60,
                5: 0.50,
                6: 0.40,
                7: 0.30,
                8: 0.20,
                9: 0.10,
                10: 0.05,
            }

    def _compute_levels(self, matrix_df: pd.DataFrame) -> pd.Series:
        if "difficulty_level" in matrix_df.columns:
            lvl = matrix_df.groupby("question_id")["difficulty_level"].first().astype(int)
            return lvl
        # Fallback: derive from cross-model mean probability
        df = matrix_df.copy()
        if "normalized_score" not in df.columns:
            raise ValueError("matrix_df must have normalized_score in [0,100]")
        df["p"] = (df["normalized_score"].astype(float) / 100.0).clip(1e-6, 1 - 1e-6)
        p_mean = df.groupby("question_id")["p"].mean()
        # Map mean correctness to 10 levels: high p → easy → lower level
        # level = 1 + floor((1 - p_mean) * 10), clipped to 1..10
        levels = (1 + np.floor((1.0 - p_mean) * 10.0)).astype(int)
        levels = levels.clip(lower=1, upper=10)
        return levels

    def _target_level_from_theta(self, theta: float) -> int:
        # Center around level 5; shift with ability. Larger theta → harder target level.
        # Map theta roughly in [-1.0, +1.0] → shift of [-2, +2]
        shift = int(max(min(round(theta * 2.0), 3), -3))
        lvl = 5 + shift
        return int(max(min(lvl, 10), 1))

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        levels = self._compute_levels(matrix_df)
        theta = simple_cold_start_theta(model)
        target_level = self._target_level_from_theta(theta)

        # Build per-question score
        lvl_prob = pd.Series({
            l: float(self.level_prob_mapping.get(l, 0.5))  # type: ignore[attr-defined]
            for l in range(1, 11)
        })
        q_levels = levels.reindex(levels.index)
        p_level = q_levels.map(lvl_prob)
        info = p_level.map(_entropy_bits)
        distance = (q_levels - target_level).abs()
        weight = info * np.exp(-self.alpha_distance * distance.astype(float))
        scored = pd.DataFrame({"question_id": levels.index, "level": q_levels.values, "score": weight.values})

        # Optional diversity by a column (e.g., category) if present
        if self.diversify_by is not None and self.diversify_by in matrix_df.columns:
            bycol = self.diversify_by
            # Get one row per question with its group value
            q_group = matrix_df.groupby("question_id")[bycol].first()
            scored = scored.join(q_group, on="question_id")
            scored = scored.sort_values([bycol, "score"], ascending=[True, False])
            # Round-robin take across groups
            groups = [g for _, g in scored.groupby(bycol)]
            picks: list[str] = []
            idx = 0
            while len(picks) < k and any(len(g) > 0 for g in groups):
                gi = idx % len(groups)
                if len(groups[gi]) > 0:
                    row = groups[gi].iloc[0]
                    picks.append(str(row["question_id"]))
                    groups[gi] = groups[gi].iloc[1:]
                idx += 1
            return picks[:k]

        # Default: take top-k by score
        top = scored.sort_values("score", ascending=False).head(k)
        return [str(q) for q in top["question_id"].tolist()]


