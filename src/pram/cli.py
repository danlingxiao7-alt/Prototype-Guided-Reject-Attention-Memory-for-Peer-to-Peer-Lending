import argparse
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd

from . import __version__
from .config import ExperimentConfig
from .demo import make_demo_data
from .pipeline import Predictor, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="PRAM main prediction pipeline (no ablation experiments)")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and evaluate a temporal experiment")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--accepted", required=True, type=Path)
    train.add_argument("--rejected", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...; auto uses CUDA 0 or CPU")
    train.add_argument("--preprocessing", choices=("train_only", "notebook_yearwise"))
    predict = commands.add_parser("predict", help="Predict from canonical, unbinned feature CSV")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=10000)
    predict.add_argument("--binning", choices=("train", "test"), help="Required for notebook_yearwise checkpoints")
    demo = commands.add_parser("make-demo", help="Write synthetic data for a smoke test")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = ExperimentConfig.from_json(args.config)
            if args.preprocessing:
                config = replace(config, preprocessing=args.preprocessing)
            summary = run_experiment(args.accepted, args.rejected, args.output, config, args.device)
            print(json.dumps(summary["aggregate"], indent=2))
        elif args.command == "predict":
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}")
            frame = pd.read_csv(args.input)
            if {"default_probability", "predicted_default"}.intersection(frame.columns):
                raise ValueError("Input already contains prediction output columns")
            predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device, binning=args.binning)
            probabilities = predictor.predict_proba(frame, args.batch_size)
            frame["default_probability"] = probabilities
            frame["predicted_default"] = (probabilities >= 0.5).astype(int)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"Saved {len(frame)} predictions to {args.output}")
        else:
            print(f"Synthetic data saved to {make_demo_data(args.output)}")
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")

