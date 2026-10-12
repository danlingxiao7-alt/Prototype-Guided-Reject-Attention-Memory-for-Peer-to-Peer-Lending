from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from entmax import sparsemax
from sklearn.preprocessing import KBinsDiscretizer

from pram.config import ExperimentConfig
from pram.data import FEATURES, FeatureBinner, clean_accepted, clean_rejected, prepare_data
from pram.demo import make_demo_data
from pram.memory import build_prototypes, filter_rejects
from pram.model import RawRejectMemory
from pram.training import set_seed


@pytest.fixture
def raw_data(tmp_path):
    folder = make_demo_data(tmp_path / "data", rows_per_year=60)
    return pd.read_csv(folder / "accepted.csv"), pd.read_csv(folder / "rejected.csv")


def test_portable_binning_matches_sklearn_at_edges_and_outliers():
    raw = pd.DataFrame({"fico_score": [500., 600., 700.], "dti": [0., 10., 30.],
                        "loan_amnt": [5000., 5000., 5000.], "emp_length": [2., 5., 10.]})
    with pytest.warns(UserWarning, match="constant"):
        binner = FeatureBinner.fit(raw, 4)
    expected_input = raw[list(FEATURES)].to_numpy().copy()
    expected_input[:, 1] = np.log1p(expected_input[:, 1])
    sklearn_binner = KBinsDiscretizer(n_bins=4, strategy="uniform", encode="ordinal", subsample=None)
    with pytest.warns(UserWarning, match="constant"):
        sklearn_binner.fit(expected_input[:, :3])
    extra = pd.DataFrame({"fico_score": [400., 550., 800.], "dti": [1., 60., 90.],
                          "loan_amnt": [100., 5000., 9000.], "emp_length": [0., 2., 10.]})
    values = extra[list(FEATURES)].to_numpy().copy()
    values[:, 1] = np.log1p(values[:, 1])
    expected = np.column_stack([sklearn_binner.transform(values[:, :3]), values[:, 3]])
    restored = FeatureBinner.from_dict(json.loads(json.dumps(binner.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.transform(extra), expected)


def test_train_only_binning_is_unaffected_by_future_data(raw_data):
    accepted, rejected = raw_data
    config = ExperimentConfig(preprocessing="train_only", num_clusters=3)
    before = prepare_data(accepted, rejected, config)
    changed_a, changed_r = accepted.copy(), rejected.copy()
    changed_a.loc[changed_a["issue_d"] == "Jan-2015", "loan_amnt"] *= 100
    changed_r.loc[changed_r["Application Date"] == "2015-02-01", "Amount Requested"] *= 100
    after = prepare_data(changed_a, changed_r, config)
    assert before.train_binner.to_dict() == after.train_binner.to_dict()
    assert after.train_binner.to_dict() == after.test_binner.to_dict()
    np.testing.assert_array_equal(before.X_train, after.X_train)
    legacy = prepare_data(changed_a, changed_r, replace(config, preprocessing="notebook_yearwise"))
    assert legacy.train_binner.to_dict() != legacy.test_binner.to_dict()


def test_legacy_and_normalized_employment_are_explicit(raw_data):
    accepted, _ = raw_data
    accepted.loc[:2, "emp_length"] = ["1 year", "< 1 year", "1 YEAR"]
    legacy = clean_accepted(accepted, 2012, legacy_employment=True)
    normalized = clean_accepted(accepted, 2012, legacy_employment=False)
    assert len(normalized) == len(legacy) + 2


def test_legacy_reject_deduplication_precedes_percentage_parsing():
    raw = pd.DataFrame({"Application Date": ["2014-01-01"] * 3, "Risk_Score": [600] * 3,
                        "Debt-To-Income Ratio": ["10%", "10.0%", "10%"],
                        "Amount Requested": [5000] * 3, "Employment Length": ["2 years"] * 3})
    legacy = clean_rejected(raw, 2014)
    assert len(legacy) == 2
    assert legacy.dti.tolist() == [10., 10.]
    assert len(clean_rejected(raw, 2014, legacy_employment=False)) == 1


def test_prototypes_are_nearest_real_points():
    values = np.array([[0., 0.], [1., 0.], [9., 0.], [10., 0.]], dtype=np.float32)
    prototypes = build_prototypes(values, num_clusters=1, topk=2, seed=42)
    np.testing.assert_array_equal(prototypes, values[[1, 2]])
    with pytest.raises(ValueError, match="candidates"):
        build_prototypes(values, num_clusters=5)


def test_filter_uses_strict_threshold_and_descending_risk():
    class Scorer:
        def predict_proba(self, values):
            return np.column_stack([1 - values[:, 0], values[:, 0]])
    candidates = np.array([[0.2], [0.7], [0.5]])
    np.testing.assert_array_equal(filter_rejects(Scorer(), candidates, 0.2), [[0.7], [0.5]])
    with pytest.raises(ValueError, match="No rejected"):
        filter_rejects(Scorer(), candidates, 0.9)


def test_attention_matches_main_formula_and_has_finite_gradients():
    set_seed(42)
    model = RawRejectMemory()
    x, prototypes = torch.randn(1, 4), torch.randn(7, 4)
    logits, attention, _ = model(x, prototypes)
    query, rejects = model.input_proj(x), model.reject_proj(prototypes)
    expected_attention = sparsemax(query @ model.W_k(rejects).T / 8, dim=-1)
    expected = model.clf(0.7 * query + 0.3 * (expected_attention @ model.W_v(rejects))).squeeze(1)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 1))
    assert (attention >= 0).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.gate[0].weight.grad is None


@pytest.mark.parametrize("kwargs", [{"threshold": 1}, {"seeds": ()}, {"num_clusters": 0},
                                    {"preprocessing": "unknown"}, {"test_year": 2014}])
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)

