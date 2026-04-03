

from dataclasses import dataclass
from typing import Protocol, Mapping, Any


@dataclass
class NormalizationResult:
    normalized: float
    method: str
    params: Mapping[str, Any]


class AbstractNormalizer(Protocol):
    """Interface for metric-specific normalization.

    Each metric should have its own concrete normalizer. Avoid generic
    fallbacks so that misconfigurations surface as errors early.
    """

    def normalize(self, metric_name: str, raw_score: float, metadata: Mapping[str, Any]) -> NormalizationResult:  # noqa: E501
        ...


