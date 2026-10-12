import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from pram.config import ExperimentConfig
from pram.data import clean_accepted
from pram.demo import make_demo_data
from pram.metrics import evaluate_probs
from pram.pipeline import Predictor, run_experiment


@pytest.mark.parametrize("mode", ["train_only", "notebook_yearwise"])
def test_training_checkpoint_and_csv_prediction_round_trip(tmp_path, mode):
    folder = make_demo_data(tmp_path / "data", rows_per_year=50)
    config = ExperimentConfig(preprocessing=mode, seeds=(42,), num_clusters=3,
                              epochs=2, batch_size=13, patience=2)
    # 40 training rows with batch_size=13 leaves a singleton last batch.
    summary = run_experiment(folder / "accepted.csv", folder / "rejected.csv",
                             tmp_path / "run", config, device="cpu")
    checkpoint = tmp_path / "run/seed_42/model.pt"
    if mode == "notebook_yearwise":
        with pytest.raises(ValueError, match="explicitly choose"):
            Predictor.from_checkpoint(checkpoint)
    predictor = Predictor.from_checkpoint(checkpoint, binning="test")
    test = clean_accepted(pd.read_csv(folder / "accepted.csv"), config.test_year,
                          legacy_employment=mode == "notebook_yearwise")
    probabilities = predictor.predict_proba(test, batch_size=13)
    metrics = evaluate_probs(test["loan_status"], probabilities)
    assert metrics == summary["runs"][0]["metrics"]
    np.testing.assert_allclose(predictor.predict_proba(test.iloc[:1]), probabilities[:1], atol=1e-6)
    restored_summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert restored_summary["aggregate"]["auc"]["std"] == 0
    assert restored_summary["counts"]["train"] == 40
    predicted_path = tmp_path / "predictions.csv"
    subprocess.run([sys.executable, "-m", "pram", "predict", "--checkpoint", str(checkpoint),
                    "--binning", "test", "--input", str(folder / "applicants.csv"),
                    "--output", str(predicted_path)], check=True)
    result = pd.read_csv(predicted_path)
    assert len(result) == 12
    assert result["default_probability"].between(0, 1).all()
    np.testing.assert_allclose(result["default_probability"],
                               predictor.predict_proba(pd.read_csv(folder / "applicants.csv")), atol=1e-7)
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(folder / "accepted.csv", folder / "rejected.csv", tmp_path / "run", config)


def test_repeated_seed_is_deterministic(tmp_path):
    folder = make_demo_data(tmp_path / "data", rows_per_year=40)
    config = ExperimentConfig(preprocessing="train_only", seeds=(42,), num_clusters=2,
                              epochs=1, batch_size=64)
    first = run_experiment(folder / "accepted.csv", folder / "rejected.csv", tmp_path / "a", config, "cpu")
    second = run_experiment(folder / "accepted.csv", folder / "rejected.csv", tmp_path / "b", config, "cpu")
    assert first == second

