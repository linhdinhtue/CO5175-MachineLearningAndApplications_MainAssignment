"""Ablation: swap the penalty (L1 / L2 / Elastic Net) at a fixed C, same pipeline/solver."""
import time

import pandas as pd

from src.data import load_adult
from src.metrics import classification_metrics
from src.modeling import build_pipeline, complexity, feature_table
from src.utils import env_info, output_dir, parse_config, save_json

SUMMARY_COLS = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy",
                "log_loss", "n_active_coef", "refit_seconds"]


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])

    rows = []
    for variant, params in cfg["variants"].items():
        for seed in cfg["seeds"]:
            model = build_pipeline({"type": "logistic", "fixed": params}, X_train, seed)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0
            proba = model.predict_proba(X_test)[:, 1]

            feature_table(model).to_csv(out / f"features_{variant}_seed{seed}.csv", index=False)
            rows.append({
                "variant": variant,
                "seed": seed,
                **classification_metrics(y_test, proba),
                "refit_seconds": fit_s,
                **complexity(model),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_metrics.csv", index=False)
    # sample std (ddof=1) across seeds
    summary = df.groupby("variant")[SUMMARY_COLS].agg(["mean", "std"])
    summary.to_csv(out / "ablation_summary.csv")

    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
