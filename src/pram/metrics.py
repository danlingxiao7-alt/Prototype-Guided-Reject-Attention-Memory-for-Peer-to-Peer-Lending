import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def evaluate_probs(y_true, probabilities):
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUC and KS require both classes in evaluation labels")
    predictions = np.asarray(probabilities) >= 0.5
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return {"accuracy": float(accuracy_score(y_true, predictions)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "auc": float(roc_auc_score(y_true, probabilities)),
            "ks": float(np.max(tpr - fpr))}

