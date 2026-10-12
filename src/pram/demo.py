"""Small synthetic fixtures for software checks, not research results."""

from pathlib import Path
import numpy as np
import pandas as pd


def make_demo_data(output_dir, *, seed=42, rows_per_year=160):
    folder = Path(output_dir)
    if any((folder / name).exists() for name in ("accepted.csv", "rejected.csv", "applicants.csv")):
        raise ValueError("Demo files already exist; choose an empty directory")
    rng = np.random.default_rng(seed)
    accepted, rejected = [], []
    for year in (2012, 2013, 2014, 2015):
        n = rows_per_year
        fico = rng.integers(580, 810, n)
        dti = rng.uniform(1, 55, n)
        amount = rng.integers(1000, 35000, n)
        probability = 1 / (1 + np.exp(-((680 - fico) / 45 + (dti - 25) / 12)))
        labels = rng.binomial(1, probability)
        labels[:2] = [0, 1]
        accepted.append(pd.DataFrame({
            "issue_d": f"Jan-{year}", "fico_range_low": fico, "fico_range_high": fico + 4,
            "dti": dti, "loan_amnt": amount, "emp_length": rng.choice(["2 years", "5 years", "10+ years"], n),
            "loan_status": np.where(labels, "Charged Off", "Fully Paid")}))
        rejected.append(pd.DataFrame({
            "Application Date": f"{year}-02-01", "Risk_Score": rng.integers(450, 730, n),
            "Debt-To-Income Ratio": [f"{v:.3f}%" for v in rng.uniform(10, 90, n)],
            "Amount Requested": rng.integers(1000, 40000, n),
            "Employment Length": rng.choice(["2 years", "5 years", "10+ years"], n)}))
    a, r = pd.concat(accepted, ignore_index=True), pd.concat(rejected, ignore_index=True)
    folder.mkdir(parents=True, exist_ok=True)
    a.to_csv(folder / "accepted.csv", index=False)
    r.to_csv(folder / "rejected.csv", index=False)
    applicants = a.tail(12).copy()
    applicants["fico_score"] = (applicants["fico_range_low"] + applicants["fico_range_high"]) / 2
    applicants["emp_length"] = applicants["emp_length"].map({"2 years": 2, "5 years": 5, "10+ years": 10})
    applicants[["fico_score", "dti", "loan_amnt", "emp_length"]].to_csv(folder / "applicants.csv", index=False)
    return folder

