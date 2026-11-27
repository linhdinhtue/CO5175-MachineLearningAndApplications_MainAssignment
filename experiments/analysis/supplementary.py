"""Supplementary analyses on the seed-42 tuned models: paired bootstrap CIs on the test set,
group metrics by sex/race, and dependence plots (GBDT SHAP vs EBM shape vs Logistic SHAP)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score  # noqa: E402

from src.data.loader import load_adult, load_raw  # noqa: E402
from src.evaluation.metrics import THRESHOLD, classification_metrics  # noqa: E402
from src.models.factory import build_pipeline  # noqa: E402
from src.explainability.shap_utils import shap_frame  # noqa: E402
from src.common.config import parse_config  # noqa: E402
from src.common.artifacts import env_info, output_dir, save_json  # noqa: E402

BOOT_METRICS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "log_loss": lambda t, p: log_loss(t, p, labels=[0, 1]),
}


def paired_bootstrap(y: np.ndarray, probas: dict, pairs: list, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = {(a, b, k): [] for a, b in pairs for k in BOOT_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = {(m, k): fn(y[idx], p[idx]) for m, p in probas.items() for k, fn in BOOT_METRICS.items()}
        for a, b in pairs:
            for k in BOOT_METRICS:
                deltas[(a, b, k)].append(vals[(a, k)] - vals[(b, k)])
    rows = []
    for (a, b, k), d in deltas.items():
        d = np.asarray(d)
        rows.append({
            "pair": f"{a} - {b}", "metric": k,
            "delta_point": BOOT_METRICS[k](y, probas[a]) - BOOT_METRICS[k](y, probas[b]),
            "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975),
            "share_boot_delta_le_0": float((d <= 0).mean()),
        })
    return pd.DataFrame(rows)


def group_metrics(y: pd.Series, pred: pd.Series, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g in sorted(groups.dropna().unique()):
        m = groups == g
        yt, yp = y[m], pred[m]
        tp, fp = int(((yt == 1) & (yp == 1)).sum()), int(((yt == 0) & (yp == 1)).sum())
        fn, tn = int(((yt == 1) & (yp == 0)).sum()), int(((yt == 0) & (yp == 0)).sum())
        rows.append({
            "group": g, "n": int(m.sum()), "n_pos": int(yt.sum()), "base_rate": yt.mean(),
            "selection_rate": yp.mean(),
            "tpr": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
        })
    return pd.DataFrame(rows)


def ebm_shape(ebm, feature: str):
    data = ebm.explain_global().data(ebm.term_names_.index(feature))
    edges, scores = np.asarray(data["names"], dtype=float), np.asarray(data["scores"], dtype=float)
    return edges, scores


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    (out / "plots").mkdir(exist_ok=True)
    seed = cfg["seed"]
    X_train, y_train, X_test, y_test = load_adult(cfg["drop_features"])
    _, raw_test = load_raw()

    pipes, probas, metrics = {}, {}, {}
    for name, model_cfg in cfg["models"].items():
        pipe = build_pipeline(model_cfg, X_train, seed).fit(X_train, y_train)
        pipes[name] = pipe
        probas[name] = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = classification_metrics(y_test, probas[name])
        print("fitted", name, round(metrics[name]["roc_auc"], 4), flush=True)
    pd.DataFrame(metrics).T.to_csv(out / "test_metrics.csv")

    boot = paired_bootstrap(y_test.to_numpy(), probas, [tuple(p) for p in cfg["pairs"]], cfg["n_boot"], seed)
    boot.to_csv(out / "paired_bootstrap.csv", index=False)

    fair_rows, gap_rows = [], []
    for attr in cfg["group_attributes"]:
        groups = raw_test[attr].reindex(X_test.index)
        for name, p in probas.items():
            pred = pd.Series((p >= THRESHOLD).astype(int), index=X_test.index)
            gm = group_metrics(y_test, pred, groups)
            gm.insert(0, "model", name)
            gm.insert(1, "attribute", attr)
            fair_rows.append(gm)
            big = gm[gm.n >= cfg["min_group_size"]]
            gap_rows.append({
                "model": name, "attribute": attr, "groups_used": ", ".join(map(str, big.group)),
                "selection_rate_gap": big.selection_rate.max() - big.selection_rate.min(),
                "tpr_gap": big.tpr.max() - big.tpr.min(),
                "fpr_gap": big.fpr.max() - big.fpr.min(),
                "ppv_gap": big.ppv.max() - big.ppv.min(),
            })
    pd.concat(fair_rows).to_csv(out / "group_metrics.csv", index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(out / "group_gaps.csv", index=False)

    rng = np.random.default_rng(seed)
    bg_idx = np.sort(rng.choice(X_train.index, size=cfg["n_background"], replace=False))
    ex_idx = np.sort(rng.choice(X_test.index, size=cfg["n_explain"], replace=False))
    dep = {}
    for name in ["gbdt", cfg["logistic_for_dependence"]]:
        dep[name] = shap_frame(pipes[name], cfg["models"][name]["type"], X_train.loc[bg_idx], X_test.loc[ex_idx])
    ebm = pipes["ebm"].named_steps["clf"] if "ebm" in pipes else None

    feats = cfg["dependence_features"]
    fig, axes = plt.subplots(1, len(feats), figsize=(4.5 * len(feats), 3.8))
    dep_rows = []
    for ax, feat in zip(np.atleast_1d(axes), feats):
        x = X_test.loc[ex_idx, feat]
        for name, color in [("gbdt", "tab:red"), (cfg["logistic_for_dependence"], "tab:blue")]:
            ax.scatter(x, dep[name][f"num__{feat}"], s=6, alpha=0.5, color=color, label=f"SHAP {name}")
        dep_rows.append(pd.DataFrame({"row": ex_idx, "feature": feat, "x": x.values,
                                      **{f"shap_{n}": dep[n][f"num__{feat}"].values for n in dep}}))
        if ebm is not None:
            edges, scores = ebm_shape(ebm, feat)
            if len(edges) == len(scores) + 1:
                ax.step(edges[:-1], scores, where="post", color="tab:green", lw=2, label="EBM shape")
            else:
                ax.plot(edges[: len(scores)], scores, color="tab:green", lw=2, label="EBM shape")
        if feat in cfg.get("symlog_features", []):
            ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlim(x.min(), x.max())
        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("contribution (log-odds)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.savefig(out / "plots" / "dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.concat(dep_rows).to_csv(out / "dependence_values.csv", index=False)

    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")
    print(boot.round(4).to_string(index=False))
    print(gaps.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
