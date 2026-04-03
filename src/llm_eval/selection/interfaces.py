

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class ModelProfile:
    model_name: str
    model_family: str | None = None
    model_size_params: str | None = None


class QuestionSelector(Protocol):
    def select(self, model: ModelProfile, k: int, matrix_df) -> list[str]:  # pragma: no cover - interface
        ...


