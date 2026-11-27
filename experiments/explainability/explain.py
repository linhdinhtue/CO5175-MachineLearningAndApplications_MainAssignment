"""SHAP for GBDT (TreeExplainer) and the best Logistic (LinearExplainer): additivity check,
global/local plots, attribution grouped by original feature, and fixed TP/TN/FP/FN cases."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from src.data.loader import load_adult  # noqa: E402
from src.evaluation.metrics import THRESHOLD, classification_metrics  # noqa: E402
from src.models.factory import build_pipeline  # noqa: E402
from src.data.preprocessing import original_feature_of_columns  # noqa: E402
from src.explainability.shap_utils import build_explainer, transform  # noqa: E402
from src.common.config import parse_config  # noqa: E402
from src.common.artifacts import env_info, output_dir, save_json  # noqa: E402


def select_cases(y: pd.Series, preds: dict, n_disagree: int) -> pd.DataFrame:
    """Smallest test row id per TP/TN/FP/FN for each model, plus first disagreements."""
    rows = []
    for model, p in preds.items():
        groups = {"TP": (y == 1) & (p == 1), "TN": (y == 0) & (p == 0),
                  "FP": (y == 0) & (p == 1), "FN": (y == 1) & (p == 0)}
        for kind, mask in groups.items():
            ids = y.index[mask]
            rows.append({"case": f"{model}_{kind}", "row": int(ids.min()) if len(ids) else None})
    g, lg = preds["gbdt"], preds["logistic"]
    for name, mask in {"gbdt_pos_logistic_neg": (g == 1) & (lg == 0),
                       "gbdt_neg_logistic_pos": (g == 0) & (lg == 1)}.items():
        for i, rid in enumerate(sorted(y.index[mask])[:n_disagree]):
            rows.append({"case": f"{name}_{i}", "row": int(rid)})
    return pd.DataFrame(rows)


def save_plot(path) -> None:
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    (out / "plots").mkdir(exist_ok=True)
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])

    pipes, probas, timing = {}, {}, {}
    for name, model_cfg in cfg["models"].items():
        pipe = build_pipeline(model_cfg, X_train, seed)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        timing[f"{name}_fit_seconds"] = time.perf_counter() - t0
        pipes[name] = pipe
        probas[name] = pipe.predict_proba(X_test)[:, 1]

    preds = {k: pd.Series((v >= THRESHOLD).astype(int), index=X_test.index) for k, v in probas.items()}
    pd.DataFrame({"row": X_test.index, "y_true": y_test.values,
                  **{f"proba_{k}": v for k, v in probas.items()}}).to_csv(out / "predictions.csv", index=False)

    cases = select_cases(y_test, preds, cfg["n_disagreements_per_direction"])
    cases["y_true"] = cases["row"].map(y_test)
    for k in probas:
        cases[f"proba_{k}"] = cases["row"].map(pd.Series(probas[k], index=X_test.index))
    cases.to_csv(out / "cases.csv", index=False)

    bg_idx = np.sort(rng.choice(X_train.index, size=cfg["n_background"], replace=False))
    sample_idx = np.sort(rng.choice(X_test.index, size=cfg["n_explain"], replace=False))
    case_idx = cases["row"].dropna().astype(int).to_numpy()
    explain_idx = np.union1d(sample_idx, case_idx)
    in_sample = np.isin(explain_idx, sample_idx)
    sample_pos = np.flatnonzero(in_sample)

    meta = {
        "seed": seed,
        "background_rows": bg_idx.tolist(),
        "sample_rows": sample_idx.tolist(),
        "sample_label_counts": {int(k): int(v) for k, v in y_test.loc[sample_idx].value_counts().items()},
        "explained_output": "raw log-odds (decision_function) of class >50K",
        "explainers": {
            "gbdt": "TreeExplainer(feature_perturbation='interventional', model_output='raw')",
            "logistic": "LinearExplainer(masker=Independent(background))",
        },
        "metrics_test": {k: classification_metrics(y_test, v) for k, v in probas.items()},
        "classification_report": {
            k: classification_report(y_test, preds[k], output_dict=True) for k in preds
        },
    }

    grouped_importance = {}
    for name, pipe in pipes.items():
        clf = pipe.named_steps["clf"]
        X_bg = transform(pipe, X_train.loc[bg_idx])
        X_ex = transform(pipe, X_test.loc[explain_idx])
        explainer = build_explainer(clf, cfg["models"][name]["type"], X_bg)
        t0 = time.perf_counter()
        expl = explainer(X_ex)
        timing[f"{name}_shap_seconds"] = time.perf_counter() - t0

        raw = clf.decision_function(X_ex.to_numpy())
        base = np.broadcast_to(np.asarray(expl.base_values, dtype=float).reshape(-1), (len(explain_idx),))
        err = np.abs(base + expl.values.sum(axis=1) - raw)
        meta[f"{name}_additivity"] = {"max_abs_error": float(err.max()), "mean_abs_error": float(err.mean())}

        values = pd.DataFrame(expl.values, columns=X_ex.columns, index=explain_idx)
        values.insert(0, "base_value", base)
        values.insert(1, "in_sample", in_sample)
        values.to_csv(out / f"shap_values_{name}.csv", index_label="row")

        groups = original_feature_of_columns(pipe.named_steps["pre"])
        grouped = pd.DataFrame(expl.values, columns=groups, index=explain_idx).T.groupby(level=0, sort=False).sum().T
        grouped.to_csv(out / f"shap_grouped_{name}.csv", index_label="row")
        grouped_importance[name] = grouped.iloc[sample_pos].abs().mean()

        pd.DataFrame({
            "feature": X_ex.columns,
            "original_feature": groups,
            "mean_abs_shap": np.abs(expl.values[sample_pos]).mean(axis=0),
            "mean_shap": expl.values[sample_pos].mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(out / f"importance_{name}.csv", index=False)

        sample_expl = expl[sample_pos]
        shap.plots.beeswarm(sample_expl, max_display=cfg["max_display"], show=False)
        save_plot(out / "plots" / f"beeswarm_{name}.png")
        shap.plots.bar(sample_expl, max_display=cfg["max_display"], show=False)
        save_plot(out / "plots" / f"bar_{name}.png")
        grouped_expl = shap.Explanation(values=grouped.to_numpy()[sample_pos], feature_names=list(grouped.columns))
        shap.plots.bar(grouped_expl, max_display=cfg["max_display"], show=False)
        save_plot(out / "plots" / f"bar_grouped_{name}.png")

        pos_of = {rid: i for i, rid in enumerate(explain_idx)}
        # several case labels can point to the same row; draw each row once
        labels = cases.dropna(subset=["row"]).groupby("row", sort=True)["case"].agg("+".join)
        for rid, label in labels.items():
            shap.plots.waterfall(expl[pos_of[int(rid)]], max_display=cfg["max_display"], show=False)
            save_plot(out / "plots" / f"waterfall_{name}_row{int(rid)}_{label}.png")

    imp = pd.DataFrame(grouped_importance)
    imp.columns = [f"mean_abs_shap_{c}" for c in imp.columns]
    for c in list(imp.columns):
        imp[c.replace("mean_abs_shap", "rank")] = imp[c].rank(ascending=False).astype(int)
    imp.sort_values(imp.columns[0], ascending=False).to_csv(out / "importance_grouped.csv", index_label="original_feature")

    meta["timing"] = timing
    save_json(meta, out / "explain_meta.json")
    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print({k: v for k, v in meta.items() if k.endswith("_additivity")})
    print(timing)
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
