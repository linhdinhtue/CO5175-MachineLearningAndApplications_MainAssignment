"""Grid search (ROC-AUC, stratified k-fold) per seed, refit on full train, evaluate on test."""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import load_adult
from src.metrics import classification_metrics
from src.modeling import build_pipeline, complexity, expand_grid, feature_table
from src.preprocess import reference_categories
from src.utils import env_info, output_dir, parse_config, save_json

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
            build_pipeline(cfg["model"], X_train, seed),
            grid,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=cfg.get("n_jobs", -1),
            error_score=np.nan,
        )
        # only warnings raised in the main process (i.e. the refit) are captured
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            total_s = time.perf_counter() - t0

        cv_res = pd.DataFrame(search.cv_results_)
        cv_res.to_csv(out / f"cv_results_seed{seed}.csv", index=False)

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
            "n_convergence_warnings_refit": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
            "tuning_seconds": total_s - search.refit_time_,
            "refit_seconds": search.refit_time_,
            "inference_seconds": infer_s,
            **complexity(model),
        })

        if seed == cfg["seeds"][0]:
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
