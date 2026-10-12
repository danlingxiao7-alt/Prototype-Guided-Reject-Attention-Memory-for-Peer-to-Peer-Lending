"""Risk filtering and K-means exemplar selection from unlabeled rejects."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_GRID = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


def fit_risk_scorer(X_train, y_train, X_val, y_val, seed=42):
    best_model, best_auc = None, -np.inf
    for C in C_GRID:
        model = LogisticRegression(C=C, max_iter=10000, solver="lbfgs", random_state=seed)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_model, best_auc = model, auc
    return best_model, {"C": float(best_model.C), "validation_auc": float(best_auc)}


def filter_rejects(scorer, X_rejected, threshold):
    probabilities = scorer.predict_proba(X_rejected)[:, 1]
    # pandas uses quicksort in the notebook; preserve its descending ordering.
    import pandas as pd
    order = pd.Series(probabilities).sort_values(ascending=False).index.to_numpy()
    selected = order[probabilities[order] > threshold]
    if not len(selected):
        raise ValueError("No rejected applicants exceed threshold; check data or lower threshold")
    return X_rejected[selected]


def build_prototypes(candidates, num_clusters, topk=3, seed=42):
    values = np.asarray(candidates, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Prototype candidates must be a finite two-dimensional array")
    if len(values) < num_clusters:
        raise ValueError(f"Only {len(values)} candidates for {num_clusters} clusters")
    if num_clusters < 1 or topk < 1:
        raise ValueError("num_clusters and topk must be positive")
    labels = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10).fit_predict(values)
    indices = []
    for cluster in range(num_clusters):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        points = values[members]
        distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
        indices.extend(members[np.argsort(distances)[:topk]].tolist())
    return values[indices].copy()

