"""Baseline: Logistic Regression L2, C=1 — 5-fold CV + test, over several seeds."""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.loader import load_adult
from src.evaluation.metrics import classification_metrics
from src.data.preprocessing import build_preprocessor, reference_categories
from src.common.config import parse_config
from src.common.artifacts import env_info, output_dir, save_json


def make_model(X, cfg: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(X, scale=True, drop_first=True)),
        ("clf", LogisticRegression(random_state=seed, **cfg["model"])),
    ])


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])

    cv_rows, test_rows = [], []
    for seed in cfg["seeds"]:
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
            model = make_model(X_train, cfg, seed)
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            proba = model.predict_proba(X_train.iloc[va])[:, 1]
            cv_rows.append({"seed": seed, "fold": fold, **classification_metrics(y_train.iloc[va], proba)})

        model = make_model(X_train, cfg, seed)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = model.predict_proba(X_test)[:, 1]
        infer_s = time.perf_counter() - t0

        clf = model.named_steps["clf"]
        test_rows.append({
            "seed": seed,
            **classification_metrics(y_test, proba),
            "refit_seconds": fit_s,
            "inference_seconds": infer_s,
            "n_iter": int(clf.n_iter_[0]),
            "n_active_coef": int((np.abs(clf.coef_) > 1e-6).sum()),
            "n_params": int(clf.coef_.size + clf.intercept_.size),
        })

        if seed == cfg["seeds"][0]:
            pre = model.named_steps["pre"]
            pd.DataFrame({
                "feature": pre.get_feature_names_out(),
                "coef": clf.coef_[0],
                "odds_ratio": np.exp(clf.coef_[0]),
            }).to_csv(out / f"coefficients_seed{seed}.csv", index=False)
            save_json(reference_categories(pre), out / "reference_categories.json")

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    cv_df.to_csv(out / "cv_metrics.csv", index=False)
    test_df.to_csv(out / "test_metrics.csv", index=False)

    metric_cols = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy", "log_loss"]
    # sample std (ddof=1) across seeds, as stated in the proposal
    summary = test_df[metric_cols].agg(["mean", "std"]).transpose()
    summary.to_csv(out / "test_summary.csv")

    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
