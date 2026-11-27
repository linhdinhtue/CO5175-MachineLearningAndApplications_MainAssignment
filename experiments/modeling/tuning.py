"""Grid search (ROC-AUC, stratified k-fold) per seed, refit on full train, evaluate on test."""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.loader import load_adult
from src.evaluation.metrics import classification_metrics
from src.models.factory import build_pipeline, complexity, expand_grid, feature_table
from src.data.preprocessing import reference_categories
from src.common.config import parse_config
from src.common.artifacts import env_info, output_dir, save_json
from src.evaluation.validation import AuditedPipeline, convergence_warning_count, fold_diagnostics

METRICS = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy", "log_loss"]


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])
    grid = expand_grid(cfg["model"]["grid"])

    rows, best_params = [], {}
    for seed in cfg["seeds"]:
        cv = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        search = GridSearchCV(
            AuditedPipeline(build_pipeline(cfg["model"], X_train, seed).steps),
            grid,
            scoring={"roc_auc": "roc_auc", "convergence_warnings": convergence_warning_count},
            cv=cv,
            refit="roc_auc",
            n_jobs=cfg.get("n_jobs", -1),
            error_score=np.nan,
        )
        # Fold warning counts are returned by the worker-side diagnostic scorer.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            total_s = time.perf_counter() - t0

        cv_res = pd.DataFrame(search.cv_results_)
        # Preserve the original single-metric CSV columns for downstream readers.
        for col in list(cv_res.columns):
            if col.endswith("_test_roc_auc"):
                cv_res[col.replace("_test_roc_auc", "_test_score")] = cv_res[col]
        cv_res.to_csv(out / f"cv_results_seed{seed}.csv", index=False)
        diagnostics = pd.DataFrame(fold_diagnostics(search.cv_results_, cfg["n_folds"]))
        diagnostics.to_csv(out / f"cv_diagnostics_seed{seed}.csv", index=False)
        save_json(search.best_estimator_.fit_warnings_, out / f"refit_warnings_seed{seed}.json")

        model = search.best_estimator_
        t0 = time.perf_counter()
        proba = model.predict_proba(X_test)[:, 1]
        infer_s = time.perf_counter() - t0

        pd.DataFrame({"row": X_test.index, "y_true": y_test.values, "proba": proba}).to_csv(
            out / f"predictions_seed{seed}.csv", index=False
        )
        feature_table(model).to_csv(out / f"features_seed{seed}.csv", index=False)

        params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        best_params[str(seed)] = params
        rows.append({
            "seed": seed,
            **classification_metrics(y_test, proba),
            "cv_roc_auc_mean": search.best_score_,
            "cv_roc_auc_std": cv_res.loc[search.best_index_, "std_test_score"],
            "best_params": json.dumps(params),
            "n_configs": len(cv_res),
            "n_failed_fits": int(cv_res.filter(regex=r"^split\d+_test_score$").isna().sum().sum()),
            "n_convergence_warnings_refit": model.n_convergence_warnings_,
            "n_convergence_warnings_cv": int(diagnostics.convergence_warnings.sum()),
            "tuning_seconds": total_s - search.refit_time_,
            "refit_seconds": search.refit_time_,
            "inference_seconds": infer_s,
            **complexity(model),
        })

        if seed == cfg["seeds"][0] and "pre" in model.named_steps:
            save_json(reference_categories(model.named_steps["pre"]), out / "reference_categories.json")

    test_df = pd.DataFrame(rows)
    test_df.to_csv(out / "test_metrics.csv", index=False)
    # sample std (ddof=1) across seeds
    summary = test_df[METRICS + ["tuning_seconds", "refit_seconds"]].agg(["mean", "std"]).transpose()
    summary.to_csv(out / "test_summary.csv")

    save_json(best_params, out / "best_params.json")
    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print(test_df[["seed", "roc_auc", "cv_roc_auc_mean", "best_params"]].to_string(index=False))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
