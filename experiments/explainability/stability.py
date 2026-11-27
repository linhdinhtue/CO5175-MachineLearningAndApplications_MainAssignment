"""Bonus: bootstrap stability of explanations. Refit on bootstrap resamples of train with
fixed hyperparameters; compare SHAP importance rankings and Logistic coefficients with the
full-data fit, on a fixed background and a fixed set of explained test rows."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.loader import load_adult
from src.evaluation.metrics import classification_metrics
from src.models.factory import ACTIVE_COEF_THRESHOLD, build_pipeline
from src.explainability.shap_utils import group_by_original, shap_frame
from src.common.config import parse_config
from src.common.artifacts import env_info, output_dir, save_json

METRICS = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy", "log_loss"]


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])
    seed = cfg["seed"]
    # same draw order as experiments/explainability/explain.py, so background and explained rows match
    rng = np.random.default_rng(seed)
    bg_idx = np.sort(rng.choice(X_train.index, size=cfg["n_background"], replace=False))
    ex_idx = np.sort(rng.choice(X_test.index, size=cfg["n_explain"], replace=False))

    boot_rng = np.random.default_rng(cfg["bootstrap_seed"])
    runs = [("full", X_train.index.to_numpy())] + [
        (str(b), boot_rng.choice(X_train.index, size=len(X_train), replace=True))
        for b in range(cfg["n_bootstrap"])
    ]

    imp_rows, coef_rows, metric_rows = [], [], []
    for run, idx in runs:
        for name, model_cfg in cfg["models"].items():
            pipe = build_pipeline(model_cfg, X_train, seed)
            pipe.fit(X_train.loc[idx], y_train.loc[idx])
            proba = pipe.predict_proba(X_test)[:, 1]
            metric_rows.append({"run": run, "model": name, **classification_metrics(y_test, proba)})

            values = shap_frame(pipe, model_cfg["type"], X_train.loc[bg_idx], X_test.loc[ex_idx])
            imp = group_by_original(values, pipe).abs().mean()
            imp_rows += [{"run": run, "model": name, "feature": f, "mean_abs_shap": v} for f, v in imp.items()]

            if model_cfg["type"] == "logistic":
                names = pipe.named_steps["pre"].get_feature_names_out()
                coef = pipe.named_steps["clf"].coef_[0]
                coef_rows += [{"run": run, "model": name, "feature": f, "coef": c} for f, c in zip(names, coef)]
        print("done", run, flush=True)

    imp = pd.DataFrame(imp_rows)
    imp["rank"] = imp.groupby(["run", "model"])["mean_abs_shap"].rank(ascending=False, method="min")
    imp.to_csv(out / "importance_runs.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics_runs.csv", index=False)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(out / "coef_runs.csv", index=False)

    full = imp[imp.run == "full"].set_index(["model", "feature"])["mean_abs_shap"]
    boots = imp[imp.run != "full"]
    k = cfg["top_k"]
    stab_rows = []
    for (model, run), g in boots.groupby(["model", "run"]):
        s = g.set_index("feature")["mean_abs_shap"]
        f = full.loc[model].reindex(s.index)
        top_b, top_f = set(s.nlargest(k).index), set(f.nlargest(k).index)
        stab_rows.append({
            "model": model, "run": run,
            "spearman_vs_full": spearmanr(s, f).statistic,
            f"jaccard_top{k}_vs_full": len(top_b & top_f) / len(top_b | top_f),
            "top1_feature": s.idxmax(),
            "top1_same_as_full": s.idxmax() == f.idxmax(),
        })
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(out / "stability_runs.csv", index=False)
    stab_summary = stab.groupby("model").agg(
        spearman_mean=("spearman_vs_full", "mean"),
        spearman_min=("spearman_vs_full", "min"),
        jaccard_mean=(f"jaccard_top{k}_vs_full", "mean"),
        jaccard_min=(f"jaccard_top{k}_vs_full", "min"),
        top1_agreement=("top1_same_as_full", "mean"),
    )
    stab_summary.to_csv(out / "stability_summary.csv")

    rank_summary = boots.groupby(["model", "feature"]).agg(
        rank_median=("rank", "median"), rank_min=("rank", "min"), rank_max=("rank", "max"),
        mean_abs_shap_mean=("mean_abs_shap", "mean"), mean_abs_shap_std=("mean_abs_shap", "std"),
    ).reset_index()
    rank_summary["rank_full"] = [imp[(imp.run == "full") & (imp.model == m) & (imp.feature == f)]["rank"].iloc[0]
                                 for m, f in zip(rank_summary.model, rank_summary.feature)]
    rank_summary.sort_values(["model", "rank_median"]).to_csv(out / "rank_summary.csv", index=False)

    n_boot = cfg["n_bootstrap"]
    coef_summary = []
    if len(coefs):
        full_coef = coefs[coefs.run == "full"].set_index(["model", "feature"])["coef"]
        bc = coefs[coefs.run != "full"]
        for (model, feat), g in bc.groupby(["model", "feature"]):
            active = g[g.coef.abs() > ACTIVE_COEF_THRESHOLD]
            ref = full_coef.get((model, feat), np.nan)
            coef_summary.append({
                "model": model, "feature": feat,
                "coef_full": ref,
                "selection_freq": len(active) / n_boot,
                "sign_agreement_with_full": float((np.sign(active.coef) == np.sign(ref)).mean()) if len(active) else np.nan,
                "coef_p05": g.coef.quantile(0.05), "coef_p95": g.coef.quantile(0.95),
            })
    coef_summary = pd.DataFrame(coef_summary)
    coef_summary.to_csv(out / "coef_summary.csv", index=False)

    boot_metrics = metrics[metrics.run != "full"].groupby("model")[METRICS].agg(
        ["mean", "std", lambda v: v.quantile(0.025), lambda v: v.quantile(0.975)])
    boot_metrics.columns = [f"{m}_{s}" for m, s in zip(boot_metrics.columns.get_level_values(0),
                                                       ["mean", "std", "p2.5", "p97.5"] * len(METRICS))]
    boot_metrics.to_csv(out / "metrics_summary.csv")

    models = list(cfg["models"])
    cross = {}
    if len(models) >= 2:
        a, b = full.loc[models[0]], full.loc[models[1]].reindex(full.loc[models[0]].index)
        cross = {f"spearman_full_{models[0]}_vs_{models[1]}": float(spearmanr(a, b).statistic)}
    save_json(cross, out / "cross_model.json")
    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print(stab_summary.round(3).to_string())
    print(cross)


if __name__ == "__main__":
    main()
