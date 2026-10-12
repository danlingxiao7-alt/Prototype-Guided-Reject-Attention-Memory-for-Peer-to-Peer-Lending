"""Train one temporal experiment and save a self-contained model per seed."""

from importlib.metadata import version
import json
from pathlib import Path
import platform
import numpy as np
import torch

from . import __version__
from .data import FeatureBinner, load_data
from .memory import build_prototypes, filter_rejects, fit_risk_scorer
from .metrics import evaluate_probs
from .model import RawRejectMemory
from .training import predict_batches, resolve_device, set_seed, train_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(accepted_path, rejected_path, output_dir, config, device="auto"):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory")
    device = resolve_device(device)
    print(f"Preparing {config.train_year} → {config.test_year} ({config.preprocessing})", flush=True)
    data = load_data(accepted_path, rejected_path, config)
    set_seed(config.split_seed)
    scorer, scorer_info = fit_risk_scorer(data.X_train, data.y_train, data.X_val, data.y_val, config.split_seed)
    candidates = filter_rejects(scorer, data.X_rejected, config.threshold)
    if len(candidates) < config.num_clusters:
        raise ValueError(f"Only {len(candidates)} candidates for {config.num_clusters} clusters")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "pram-lending": __version__, "device": str(device),
                   **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "entmax")}}
    write_json(output / "environment.json", environment)
    print(f"Data counts: {data.counts}; prototype candidates: {len(candidates)}", flush=True)
    runs = []
    for seed in config.seeds:
        set_seed(seed)
        prototypes = build_prototypes(candidates, config.num_clusters, config.topk, seed)
        model, history, info = train_model(data, prototypes, config, device)
        probabilities = predict_batches(model, torch.tensor(prototypes), data.X_test, config.batch_size)
        metrics = evaluate_probs(data.y_test, probabilities)
        folder = output / f"seed_{seed}"
        folder.mkdir()
        checkpoint = {
            "format_version": 1, "config": config.to_dict(), "seed": seed,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "prototypes": torch.tensor(prototypes, dtype=torch.float32),
            "train_binner": data.train_binner.to_dict(), "test_binner": data.test_binner.to_dict(),
        }
        torch.save(checkpoint, folder / "model.pt")
        write_json(folder / "history.json", history)
        run = {"seed": seed, "prototype_count": len(prototypes), **info, "metrics": metrics}
        write_json(folder / "metrics.json", run)
        runs.append(run)
        print(f"Seed {seed}: AUC={metrics['auc']:.6f}, KS={metrics['ks']:.6f}, best epoch={info['best_epoch']}", flush=True)
    summary = {
        "config": config.to_dict(), "counts": data.counts, "candidate_count": len(candidates),
        "risk_scorer": scorer_info,
        "lr_reference_metrics": evaluate_probs(data.y_test, scorer.predict_proba(data.X_test)[:, 1]),
        "runs": runs,
        "aggregate": {name: {"mean": float(np.mean([run["metrics"][name] for run in runs])),
                             "std": float(np.std([run["metrics"][name] for run in runs], ddof=0))}
                      for name in runs[0]["metrics"]},
    }
    write_json(output / "summary.json", summary)
    return summary


class Predictor:
    """Predict one checkpoint. No ensembling or implicit test-score selection."""

    def __init__(self, model, prototypes, binner, device):
        self.model = model.to(device).eval()
        self.prototypes = prototypes.to(device)
        self.binner = binner

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", binning=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        mode = state["config"]["preprocessing"]
        if binning is None:
            if mode == "notebook_yearwise":
                raise ValueError("For notebook_yearwise checkpoints, explicitly choose binning='train' "
                                 "for new applicants or binning='test' for the historical test-year protocol")
            binning = "train"
        if binning not in {"train", "test"}:
            raise ValueError("binning must be train or test")
        model = RawRejectMemory()
        model.load_state_dict(state["state_dict"])
        return cls(model, state["prototypes"], FeatureBinner.from_dict(state[f"{binning}_binner"]),
                   resolve_device(device))

    def predict_proba(self, frame, batch_size=10000):
        return predict_batches(self.model, self.prototypes, self.binner.transform(frame), batch_size)

