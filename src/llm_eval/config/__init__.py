

from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class MetricConfig(BaseModel):
    higher_is_better: bool
    min: Optional[float] = None
    max: Optional[float] = None
    method: str = "minmax"  # minmax|zscore|quantile


class SplitRatios(BaseModel):
    train_ratio: float = 0.6
    link_ratio: float = 0.2
    test_ratio: float = 0.2

    @model_validator(mode="after")
    def _validate_positive(self) -> "SplitRatios":
        ratios = {
            "train_ratio": self.train_ratio,
            "link_ratio": self.link_ratio,
            "test_ratio": self.test_ratio,
        }
        for key, val in ratios.items():
            if val < 0:
                raise ValueError(f"{key} must be non-negative")
        total = sum(ratios.values())
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value")
        return self

    def normalized(self) -> "SplitRatios":
        total = self.train_ratio + self.link_ratio + self.test_ratio
        return SplitRatios(
            train_ratio=self.train_ratio / total,
            link_ratio=self.link_ratio / total,
            test_ratio=self.test_ratio / total,
        )

    def as_dict(self) -> Dict[str, float]:
        norm = self.normalized()
        return {
            "train_ratio": norm.train_ratio,
            "link_ratio": norm.link_ratio,
            "test_ratio": norm.test_ratio,
        }


class SplitSettings(BaseModel):
    default: SplitRatios = Field(default_factory=SplitRatios)
    overrides: Dict[str, SplitRatios] = Field(default_factory=dict)

    def for_skill(self, skill: str) -> SplitRatios:
        return self.overrides.get(skill, self.default)


class AppConfig(BaseSettings):
    storage_dir: str = "data/processed"
    decay: float = 0.9
    metrics: Dict[str, MetricConfig] = {}
    split_settings: SplitSettings = Field(default_factory=SplitSettings)

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",
    }


def load_yaml_config(defaults_path: str, metrics_path: str) -> AppConfig:
    import yaml

    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f) or {}
    with open(metrics_path, "r") as f:
        metrics_data = yaml.safe_load(f) or {}

    metrics = metrics_data
    if "metrics" in defaults:
        # allow override via defaults
        metrics = defaults.get("metrics", metrics)

    split_cfg = defaults.get("split", {})
    split_settings = SplitSettings(
        default=SplitRatios(**split_cfg.get("default", {})) if split_cfg.get("default") else SplitRatios(),
        overrides={k: SplitRatios(**v) for k, v in split_cfg.get("overrides", {}).items()}
        if split_cfg.get("overrides")
        else {},
    )

    cfg = AppConfig(
        storage_dir=defaults.get("defaults", {}).get("storage_dir", "data/processed"),
        decay=defaults.get("defaults", {}).get("decay", 0.9),
        metrics={k: MetricConfig(**v) for k, v in metrics.items()},
        split_settings=split_settings,
    )
    return cfg


