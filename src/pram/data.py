"""Lending Club cleaning, temporal splits, and portable binning state."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from .config import ExperimentConfig

FEATURES = ("fico_score", "dti", "loan_amnt", "emp_length")
ACCEPTED_COLUMNS = ("issue_d", "fico_range_high", "fico_range_low", "dti",
                    "loan_amnt", "emp_length", "loan_status")
REJECTED_COLUMNS = ("Application Date", "Risk_Score", "Debt-To-Income Ratio",
                    "Amount Requested", "Employment Length")
EMPLOYMENT = {**{f"{i} years": i for i in range(2, 10)},
              "10+ years": 10, "1 YEAR": 1, "< 1 YEAR": 0}


def _employment(values, legacy):
    if legacy:
        return values.map(EMPLOYMENT)
    mapping = {key.lower(): value for key, value in EMPLOYMENT.items()}
    return values.astype("string").str.strip().str.lower().map(mapping)


def _year(values):
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.year


def _clean_numeric(frame, *, deduplicate=True):
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if deduplicate:
        frame = frame.drop_duplicates()
    return frame.loc[frame["dti"] >= 0].reset_index(drop=True)


def clean_accepted(raw, year, *, legacy_employment=True):
    frame = raw.loc[_year(raw["issue_d"]) == year].copy()
    frame["fico_score"] = (pd.to_numeric(frame["fico_range_high"], errors="coerce")
                           + pd.to_numeric(frame["fico_range_low"], errors="coerce")) / 2
    frame["loan_status"] = frame["loan_status"].map(
        {"Fully Paid": 0, "Charged Off": 1, "Default": 1})
    frame["emp_length"] = _employment(frame["emp_length"], legacy_employment)
    return _clean_numeric(frame[[*FEATURES, "loan_status"]].copy())


def clean_rejected(raw, year, *, legacy_employment=True):
    frame = raw.loc[_year(raw["Application Date"]) == year].copy()
    frame = frame.rename(columns={
        "Risk_Score": "fico_score", "Debt-To-Income Ratio": "dti",
        "Amount Requested": "loan_amnt", "Employment Length": "emp_length"})
    frame["emp_length"] = _employment(frame["emp_length"], legacy_employment)
    frame = frame[list(FEATURES)].copy()
    if legacy_employment:
        # The notebook deduplicates before parsing percentage strings. Preserve
        # distinct source rows such as '10%' and '10.0%' in historical mode.
        frame = frame.dropna().drop_duplicates()
    frame["dti"] = frame["dti"].astype("string").str.rstrip("%")
    return _clean_numeric(frame, deduplicate=not legacy_employment)


def feature_array(frame):
    """Validate canonical, unbinned features without discarding prediction rows."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite numbers; missing values are not imputed")
    if (values[:, 1] < 0).any():
        raise ValueError("dti must be nonnegative, in percentage points before log1p")
    values[:, 1] = np.log1p(values[:, 1])
    return values


@dataclass
class FeatureBinner:
    """Uniform bins for FICO, log1p(DTI), amount; employment is unchanged.

    Only interior cut points are stored, so JSON remains finite even when a
    feature is constant. searchsorted(right) matches sklearn ordinal encoding.
    """
    cut_points: list[list[float]]

    @classmethod
    def fit(cls, frame, n_bins=32):
        values = feature_array(frame)
        if not len(values):
            raise ValueError("Cannot fit bins on an empty dataset")
        estimator = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                    strategy="uniform", subsample=None)
        estimator.fit(values[:, :3])
        return cls([edges[1:-1].tolist() for edges in estimator.bin_edges_])

    def transform(self, frame):
        values = feature_array(frame)
        for col, edges in enumerate(self.cut_points):
            values[:, col] = np.searchsorted(edges, values[:, col], side="right")
        return values.astype(np.float32)

    def to_dict(self):
        return {"features": list(FEATURES), "cut_points": self.cut_points}

    @classmethod
    def from_dict(cls, state):
        if state["features"] != list(FEATURES) or len(state["cut_points"]) != 3:
            raise ValueError("Incompatible saved feature schema")
        return cls(state["cut_points"])


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_rejected: np.ndarray
    train_binner: FeatureBinner
    test_binner: FeatureBinner
    counts: dict


def prepare_data(accepted, rejected, config: ExperimentConfig):
    legacy = config.preprocessing == "notebook_yearwise"
    accepted_train = clean_accepted(accepted, config.train_year, legacy_employment=legacy)
    accepted_test = clean_accepted(accepted, config.test_year, legacy_employment=legacy)
    rejected_train = clean_rejected(rejected, config.train_year, legacy_employment=legacy)
    for name, frame in (("training accepted", accepted_train), ("test accepted", accepted_test)):
        if frame["loan_status"].nunique() != 2:
            raise ValueError(f"{name} data must contain both label classes after cleaning")
    if rejected_train.empty:
        raise ValueError("No training-year rejected applicants remain after cleaning")
    train_idx, val_idx = train_test_split(
        np.arange(len(accepted_train)), test_size=config.validation_fraction,
        random_state=config.split_seed, stratify=accepted_train["loan_status"])
    if legacy:
        train_binner = FeatureBinner.fit(pd.concat([accepted_train, rejected_train]), config.n_bins)
        rejected_test = clean_rejected(rejected, config.test_year, legacy_employment=True)
        test_binner = FeatureBinner.fit(pd.concat([accepted_test, rejected_test]), config.n_bins)
    else:
        train_binner = FeatureBinner.fit(accepted_train.iloc[train_idx], config.n_bins)
        test_binner = train_binner
    X = train_binner.transform(accepted_train)
    labels = accepted_train["loan_status"].to_numpy(dtype=np.float32)
    return PreparedData(
        X[train_idx], labels[train_idx], X[val_idx], labels[val_idx],
        test_binner.transform(accepted_test),
        accepted_test["loan_status"].to_numpy(dtype=np.float32),
        train_binner.transform(rejected_train), train_binner, test_binner,
        {"train": len(train_idx), "validation": len(val_idx), "test": len(accepted_test),
         "rejected_train": len(rejected_train),
         "rejected_test_used_for_binning": len(rejected_test) if legacy else 0})


def load_data(accepted_path, rejected_path, config):
    accepted = pd.read_csv(accepted_path, usecols=list(ACCEPTED_COLUMNS), low_memory=False)
    rejected = pd.read_csv(rejected_path, usecols=list(REJECTED_COLUMNS), low_memory=False)
    return prepare_data(accepted, rejected, config)

