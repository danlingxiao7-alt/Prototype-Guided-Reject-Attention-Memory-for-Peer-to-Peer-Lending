# PRAM: Prototype-Guided Reject Attention Memory

[中文说明](README.zh-CN.md)

An installable Python package extracted from the **main K-means prediction pipeline** in `Test_20epoh_Kmeans.ipynb`. Rejected applicants supply an unlabeled memory bank; they are not assigned training labels. This repository contains no ablation runs, parameter sweeps, or hard-coded result plots.

**Scope:** this is a refactoring of the supplied notebook, not a claim of exact reproduction of any published table. The original training data and trained weights are not distributed. See [reproducibility notes](docs/REPRODUCIBILITY.md) for preserved behavior and explicit differences.

## Method

1. Clean accepted and rejected applications into four shared features.
2. Split training-year accepted applicants into 80% training and 20% validation; evaluate on the next year.
3. Select logistic-regression regularization by validation AUC and retain rejects with estimated default probability strictly above `threshold`.
4. Run K-means and retain up to three actual applicants nearest each cluster centroid.
5. Train the attention-memory predictor: 64-dimensional representations, one attention head, **sparsemax**, and fixed fusion `0.7 * query + 0.3 * memory`.
6. Use BCE loss, Adam, and validation-loss early stopping; report AUC, KS, accuracy, precision, recall, and F1 across seeds.

The fitted logistic risk scorer's test metrics are also reported as `lr_reference_metrics`, as in the main notebook cells. There is no separate baseline-comparison framework.

## Installation

Python 3.10 or newer is required. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows activate with `.venv\Scripts\activate`. CPU is supported. For GPU training, install a PyTorch build appropriate for your CUDA installation. `--device auto` selects `cuda:0` if available, otherwise CPU; choose `--device cpu` explicitly for CPU execution.

## Quick smoke test (synthetic data)

```bash
pram make-demo --output data/demo
pram train --config configs/demo.json \
  --accepted data/demo/accepted.csv \
  --rejected data/demo/rejected.csv \
  --output runs/demo --device cpu
pram predict --checkpoint runs/demo/seed_42/model.pt \
  --input data/demo/applicants.csv --output runs/demo/predictions.csv
```

The demo runs two epochs and is only a software check. Its metrics have no scientific interpretation. Use a new output directory for subsequent runs; existing files are not overwritten. All commands also work as `python -m pram ...`.

## Lending Club experiments

Supply your own accepted and rejected CSV files (`.csv.gz` is also supported). Only these columns are read:

| File | Required columns |
| --- | --- |
| Accepted | `issue_d`, `fico_range_high`, `fico_range_low`, `dti`, `loan_amnt`, `emp_length`, `loan_status` |
| Rejected | `Application Date`, `Risk_Score`, `Debt-To-Income Ratio`, `Amount Requested`, `Employment Length` |

For accepted loans, `Fully Paid` maps to 0; `Charged Off` and `Default` map to 1. Other statuses, missing feature values, duplicate retained rows, and negative DTI values are excluded. FICO is the mean of the accepted loan's two FICO bounds. DTI uses percentage-point units (`15` or `15%` for rejects, meaning 15%, not 0.15); it is transformed by `log1p`. FICO, log-DTI, and loan amount are placed in 32 uniform bins, and employment length remains numeric. Invalid dates or nonnumeric/infinite features are excluded during training.

```bash
pram train --config configs/2014-2015.json \
  --accepted /path/to/accepted_2007_to_2018Q4.csv.gz \
  --rejected /path/to/rejected_2007_to_2018Q4.csv.gz \
  --output runs/2014-2015 --device auto
```

| Configuration | Training → test | K-means clusters | Seeds |
| --- | --- | ---: | --- |
| `2012-2013.json` | 2012 → 2013 | 8 | 6225, 4507, 3528, 1292, 435 |
| `2013-2014.json` | 2013 → 2014 | 12 | 5882, 5636, 6202, 8280, 410 |
| `2014-2015.json` | 2014 → 2015 | 13 | 4507, 2287, 9675, 4013, 3658 |

All three examples set `threshold=0.2`, one of the original notebook's scanned values. **This is an example, not a verified optimal or paper-final threshold.** Set it to your intended value in the JSON before running. No threshold is chosen using test results. Other shared defaults are 20 epochs, batch size 10,000, learning rate 0.001, weight decay 0.0001, patience 3, split seed 42, and three prototypes per cluster.

### Two explicit preprocessing protocols

- **`notebook_yearwise`** (the three historical configs): preserves separate per-year bin fitting on that year's accepted and rejected records, including test-year covariates. It also preserves the original case-sensitive employment mapping. This is the notebook's historical/transductive protocol; it uses future covariate information and is not an untouched future-data evaluation.
- **`train_only`** (the demo): fits bins only on training-split accepted records, applies the same bins to validation, rejects, and test, and normalizes employment strings such as `1 year` and `< 1 year`. It is suitable for predicting new applicants with a fixed preprocessing transform. Its results are not interchangeable with the historical protocol.

To run the latter on real data, add `--preprocessing train_only` to the training command. The choice is recorded in each checkpoint and `config.json`.

## Saved outputs

```text
runs/2014-2015/
├── config.json          # resolved experiment settings
├── environment.json     # dependency versions and execution device
├── summary.json         # per-seed metrics, means, and population std (ddof=0)
└── seed_4507/           # one folder per configured seed
    ├── model.pt         # weights, memory prototypes, both binning transforms
    ├── metrics.json     # metrics and best epoch
    └── history.json     # training and validation losses
```

Metrics are on a 0–1 scale. Predictions use a 0.5 default threshold. The summary averages **metrics** across independent seeds; it is not an ensemble. A predictor uses whichever seed checkpoint you explicitly supply.

## Predict new applicants

Prediction accepts canonical **unbinned** features in a CSV. Use numeric employment years; compute `fico_score` yourself when starting from accepted-loan raw columns. Extra columns (such as an applicant ID) are preserved.

```csv
applicant_id,fico_score,dti,loan_amnt,emp_length
example-1,682,15.2,10000,5
example-2,622,28.5,15000,2
```

Missing/nonfinite values and negative DTI raise an error rather than silently dropping rows. Do not pass already logged or binned data. Output adds `default_probability` and `predicted_default` in the original row order.

```bash
pram predict --checkpoint runs/2014-2015/seed_4507/model.pt \
  --binning train --input applicants.csv --output predictions.csv
```

For `notebook_yearwise` checkpoints, the binning choice is mandatory: `--binning train` uses the training-year transform for new applicants; `--binning test` uses the saved historical test-year transform when reproducing that test protocol. No bins are fitted during prediction. For `train_only`, the two transforms are identical and `--binning` can be omitted. Prefer training with `train_only` for new-data use.

```python
import pandas as pd
from pram.pipeline import Predictor

predictor = Predictor.from_checkpoint(
    "runs/demo/seed_42/model.pt", device="cpu"
)
probabilities = predictor.predict_proba(pd.read_csv("data/demo/applicants.csv"))
```

## Development

```bash
python -m pytest -q
python -m build
```

Tests cover the main attention equation, real-point prototype selection, threshold semantics, binning boundaries and serialization, independence from future covariates in `train_only`, singleton batches, deterministic CPU runs, checkpoint reload, and CSV prediction. GitHub Actions runs the tests on Python 3.10 and 3.12. Large files are read with selected columns but are not streamed; full dataset runs require sufficient RAM.

See the [local validation record](docs/VALIDATION.md) for the tested environment and the scope of the notebook parity check.

```text
src/pram/
  config.py       experiment settings
  data.py         cleaning, splits, binning
  memory.py       logistic risk filter and K-means prototypes
  model.py        attention-memory network
  training.py     training and batched inference
  metrics.py      evaluation metrics
  pipeline.py     experiments and checkpoint loading
  cli.py          command-line interface
  demo.py         synthetic data generator
configs/          three temporal examples and a small demo
tests/            functional tests
docs/             extraction and reproducibility notes
```

## Publishing this repository

Upload this directory's source files to your GitHub repository. `data/`, `runs/`, checkpoints, and build outputs are ignored; do not include your Lending Club files. No software license was chosen on the authors' behalf. Add your chosen license and verified paper citation before granting reuse rights or presenting the repository as an official publication release.

