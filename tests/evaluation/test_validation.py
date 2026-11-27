import unittest
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation.validation import AuditedPipeline, convergence_warning_count, fold_diagnostics


class ValidationTests(unittest.TestCase):
    def test_parallel_cv_records_nonconvergence_without_changing_selection(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = LogisticRegression(solver="saga", max_iter=1, tol=1e-12, random_state=42)
        grid = {"clf__C": [0.1, 1.]}
        cv = StratifiedKFold(2, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audited = GridSearchCV(AuditedPipeline([("clf", clf)]), grid, cv=cv, n_jobs=2,
                                   scoring={"roc_auc": "roc_auc", "convergence_warnings": convergence_warning_count},
                                   refit="roc_auc").fit(X, y)
            original = GridSearchCV(Pipeline([("clf", clf)]), grid, cv=cv,
                                    scoring="roc_auc").fit(X, y)
        diagnostics = fold_diagnostics(audited.cv_results_, 2)
        self.assertEqual(len(diagnostics), 4)
        self.assertTrue(all(d["convergence_warnings"] > 0 for d in diagnostics))
        self.assertEqual(audited.best_params_, original.best_params_)
        np.testing.assert_allclose(audited.predict_proba(X), original.predict_proba(X))



if __name__ == "__main__":
    unittest.main()
