"""Single-experiment settings; no ablation or sweep runner."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    train_year: int = 2014
    test_year: int = 2015
    threshold: float = 0.2
    num_clusters: int = 13
    topk: int = 3
    seeds: tuple[int, ...] = (4507, 2287, 9675, 4013, 3658)
    split_seed: int = 42
    validation_fraction: float = 0.2
    n_bins: int = 32
    preprocessing: str = "notebook_yearwise"
    epochs: int = 20
    batch_size: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3

    def __post_init__(self):
        if self.test_year <= self.train_year:
            raise ValueError("test_year must be later than train_year")
        if not 0 <= self.threshold < 1:
            raise ValueError("threshold must be in [0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.preprocessing not in {"notebook_yearwise", "train_only"}:
            raise ValueError("preprocessing must be notebook_yearwise or train_only")
        for key in ("num_clusters", "topk", "epochs", "batch_size", "patience"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        if not self.learning_rate > 0 or not self.weight_decay >= 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty list of unique integers")
        for value in (*self.seeds, self.split_seed):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError("seeds must be integers in [0, 2**32)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        return cls(**values)

