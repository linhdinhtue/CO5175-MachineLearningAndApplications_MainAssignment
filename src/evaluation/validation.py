"""Cross-validation diagnostics collected inside each worker, before clones disappear.

The diagnostic scorer records convergence warnings; it never selects the winner.
Selection remains ROC-AUC, exactly as in the original experiment.
"""
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline


class AuditedPipeline(Pipeline):
    def fit(self, X, y=None, **params):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            super().fit(X, y, **params)
        self.fit_warnings_ = [
            {"category": w.category.__name__, "message": str(w.message)} for w in caught
        ]
        self.n_convergence_warnings_ = sum(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        # Keep all warning messages visible in the experiment log as well.
        for w in caught:
            warnings.warn(str(w.message), w.category, stacklevel=2)
        return self


def convergence_warning_count(estimator, X, y):
    """A diagnostic, not a quality metric: lower is better; never used for refit."""
    return estimator.n_convergence_warnings_


def fold_diagnostics(cv_results, n_folds):
    """One row per candidate/fold, including NaN scores from failed fits."""
    rows = []
    for candidate, params in enumerate(cv_results["params"]):
        for fold in range(n_folds):
            rows.append({
                "candidate": candidate, "fold": fold, "params": params,
                "roc_auc": cv_results[f"split{fold}_test_roc_auc"][candidate],
                "convergence_warnings": cv_results[f"split{fold}_test_convergence_warnings"][candidate],
            })
    return rows
