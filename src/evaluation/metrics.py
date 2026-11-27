import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

THRESHOLD = 0.5


def classification_metrics(y_true, proba) -> dict:
    proba = np.asarray(proba)
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
