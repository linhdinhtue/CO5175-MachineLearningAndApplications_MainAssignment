"""Reproduce Lundberg & Lee (2017): Figure 3 (sample efficiency of Kernel SHAP vs Shapley
sampling vs LIME on a dense and a sparse decision tree) and Figure 4 (two toy games with
exact Shapley values). All methods work in the binary simplified-input space z' of the paper:
z'_j = 1 keeps x_j, z'_j = 0 replaces it with a single reference value."""
import itertools
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from src.utils import env_info, output_dir, parse_config, save_json  # noqa: E402


def exact_shapley(f, n_features: int, players: list[int]) -> tuple[np.ndarray, float]:
    """Exact Shapley values (Eq. 8) by enumerating coalitions of `players`; others are null."""
    r = len(players)
    coalitions = list(itertools.product([0, 1], repeat=r))
    Z = np.zeros((len(coalitions), n_features))
    for k, c in enumerate(coalitions):
        Z[k, players] = c
    v = dict(zip(coalitions, f(Z)))
    phi = np.zeros(n_features)
    for a, i in enumerate(players):
        for c in coalitions:
            if c[a]:
                continue
            s = sum(c)
            w = math.factorial(s) * math.factorial(r - s - 1) / math.factorial(r)
            phi[i] += w * (v[c[:a] + (1,) + c[a + 1:]] - v[c])
    return phi, float(v[(0,) * r])


def kernel_shap(f, n_features: int, n_evals: int, seed: int, l1_reg) -> np.ndarray:
    np.random.seed(seed)
    explainer = shap.KernelExplainer(f, np.zeros((1, n_features)))
    values = explainer.shap_values(np.ones((1, n_features)), nsamples=n_evals, l1_reg=l1_reg, silent=True)
    return np.asarray(values, dtype=float).reshape(-1)


def shapley_sampling(f, n_features: int, feature: int, n_evals: int, rng) -> float:
    """Permutation sampling (Strumbelj & Kononenko 2014); each sample costs 2 evaluations."""
    k = max(n_evals // 2, 1)
    Z = np.zeros((2 * k, n_features))
    for s in range(k):
        perm = rng.permutation(n_features)
        before = perm[: np.flatnonzero(perm == feature)[0]]
        Z[2 * s, before] = 1
        Z[2 * s + 1, before] = 1
        Z[2 * s + 1, feature] = 1
    out = f(Z)
    return float((out[1::2] - out[0::2]).mean())


def lime_coefs(f, n_features: int, n_samples: int, seed: int, train: np.ndarray) -> np.ndarray:
    explainer = LimeTabularExplainer(
        train, mode="regression", categorical_features=list(range(n_features)),
        discretize_continuous=False, random_state=seed,
    )
    exp = explainer.explain_instance(np.ones(n_features), f, num_features=n_features, num_samples=n_samples)
    coefs = np.zeros(n_features)
    # in regression mode LIME stores the fit under key 1 and its negation under key 0
    for j, w in exp.as_map()[1]:
        coefs[j] = w
    return coefs


def build_setup(spec: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    M, rel = spec["n_features"], spec["relevant_features"]
    X = rng.normal(size=(spec["n_train"], M))
    if spec["target"] == "and":
        y = np.all(X[:, :rel] > 0, axis=1).astype(float)
    else:
        y = X[:, :rel] @ rng.normal(size=rel)
    tree = DecisionTreeRegressor(max_depth=spec["max_depth"], random_state=seed).fit(X, y)

    x = rng.normal(size=M)
    if spec["target"] == "and":
        x[:rel] = np.abs(x[:rel]) + 0.5
    ref = np.full(M, float(spec["ref_value"]))

    def f(Z):
        Z = np.asarray(Z, dtype=float)
        return tree.predict(Z * x + (1 - Z) * ref)

    used = sorted({int(j) for j in tree.tree_.feature if j >= 0})
    phi, base = exact_shapley(f, M, used)
    feature = int(np.argmax(np.abs(phi))) if spec["feature"] == "max_abs" else int(spec["feature"])

    tree_phi = shap.TreeExplainer(tree, data=ref[None, :], feature_perturbation="interventional").shap_values(x[None, :])
    return {
        "f": f, "M": M, "feature": feature, "phi": phi, "base": base,
        "info": {
            "n_features": M,
            "used_features": used,
            "tree_depth": int(tree.get_depth()),
            "tree_leaves": int(tree.get_n_leaves()),
            "explained_feature": feature,
            "true_shapley": float(phi[feature]),
            "f_x": float(f(np.ones((1, M)))[0]),
            "f_ref": base,
            "local_accuracy_error": float(abs(base + phi.sum() - f(np.ones((1, M)))[0])),
            "tree_explainer_max_abs_diff": float(np.abs(np.asarray(tree_phi).reshape(-1) - phi).max()),
        },
    }


def run_figure3(cfg: dict, out) -> pd.DataFrame:
    rows, infos = [], {}
    for name, spec in cfg["setups"].items():
        st = build_setup(spec, cfg["seed"])
        infos[name] = st["info"]
        f, M, feat = st["f"], st["M"], st["feature"]
        train = np.random.default_rng(cfg["seed"]).integers(0, 2, size=(cfg["lime_train_rows"], M)).astype(float)
        timing = {"kernel_shap": 0.0, "shapley_sampling": 0.0, "lime": 0.0}
        for n in cfg["n_evals"]:
            for rep in range(cfg["n_reps"]):
                seed = cfg["seed"] * 100_003 + n * 1_000 + rep
                t0 = time.perf_counter()
                k = kernel_shap(f, M, n, seed, cfg["kernel_l1_reg"])[feat]
                t1 = time.perf_counter()
                s = shapley_sampling(f, M, feat, n, np.random.default_rng(seed))
                t2 = time.perf_counter()
                lm = lime_coefs(f, M, n, seed, train)[feat]
                t3 = time.perf_counter()
                timing["kernel_shap"] += t1 - t0
                timing["shapley_sampling"] += t2 - t1
                timing["lime"] += t3 - t2
                rows += [
                    {"setup": name, "method": "kernel_shap", "n_evals": n, "rep": rep, "estimate": k},
                    {"setup": name, "method": "shapley_sampling", "n_evals": n, "rep": rep, "estimate": s},
                    {"setup": name, "method": "lime", "n_evals": n, "rep": rep, "estimate": lm},
                ]
        infos[name]["seconds"] = timing
        print(name, infos[name])

    est = pd.DataFrame(rows)
    est.to_csv(out / "fig3_estimates.csv", index=False)
    summary = est.groupby(["setup", "method", "n_evals"])["estimate"].agg(
        median="median", p10=lambda v: v.quantile(0.1), p90=lambda v: v.quantile(0.9), mean="mean", std="std",
    ).reset_index()
    summary["true_shapley"] = summary["setup"].map({k: v["true_shapley"] for k, v in infos.items()})
    summary["abs_error_median"] = (summary["median"] - summary["true_shapley"]).abs()
    summary["width_p10_p90"] = summary["p90"] - summary["p10"]
    summary.to_csv(out / "fig3_summary.csv", index=False)
    save_json(infos, out / "fig3_setups.json")

    colors = {"kernel_shap": "tab:blue", "shapley_sampling": "tab:purple", "lime": "tab:green"}
    labels = {"kernel_shap": "SHAP (Kernel)", "shapley_sampling": "Shapley sampling", "lime": "LIME"}
    fig, axes = plt.subplots(1, len(infos), figsize=(6 * len(infos), 4))
    for ax, (name, info) in zip(np.atleast_1d(axes), infos.items()):
        for method, color in colors.items():
            s = summary[(summary.setup == name) & (summary.method == method)]
            ax.plot(s.n_evals, s["median"], color=color, label=labels[method])
            ax.fill_between(s.n_evals, s.p10, s.p90, color=color, alpha=0.2)
        ax.axhline(info["true_shapley"], color="dimgray", lw=2, label="True Shapley value")
        ax.set_title(f"{name}: {len(info['used_features'])} of {info['n_features']} features used")
        ax.set_xlabel("# of function evaluations")
        ax.set_ylabel(f"Feature importance (feature {info['explained_feature']})")
    np.atleast_1d(axes)[0].legend()
    fig.savefig(out / "plots" / "figure3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def run_figure4(cfg: dict, out) -> pd.DataFrame:
    def sickness(Z):
        Z = np.asarray(Z, dtype=float)
        fever, cough = Z[:, 0] > 0.5, Z[:, 1] > 0.5
        return np.where(fever & cough, 2.0, np.where(fever ^ cough, 5.0, 0.0))

    scores = np.array(cfg["fig4_max_scores"], dtype=float)

    def max_game(Z):
        return (np.asarray(Z, dtype=float) * scores).max(axis=1)

    games = {
        "sickness": (sickness, ["Fever", "Cough", "Congestion"]),
        "max_profit": (max_game, ["Man 1", "Man 2", "Man 3"]),
    }
    paper = cfg["fig4_paper_values_read_from_figure"]
    rows = []
    for game, (f, names) in games.items():
        M = len(names)
        exact, _ = exact_shapley(f, M, list(range(M)))
        kern = kernel_shap(f, M, 2 ** M, cfg["seed"], False)
        train = np.array(list(itertools.product([0, 1], repeat=M)), dtype=float)
        lm = lime_coefs(f, M, cfg["fig4_lime_num_samples"], cfg["seed"], train)
        for j, feat in enumerate(names):
            rows.append({
                "game": game, "feature": feat,
                "exact_shapley": exact[j], "kernel_shap": kern[j], "lime": lm[j],
                **{f"paper_{k}": v[j] for k, v in paper[game].items()},
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "fig4_values.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (game, g) in zip(axes, df.groupby("game", sort=False)):
        cols = ["paper_human", "exact_shapley", "kernel_shap", "lime", "paper_lime"]
        g.set_index("feature")[cols].plot.bar(ax=ax, rot=0)
        ax.set_title(game)
        ax.set_ylabel("feature attribution value")
    fig.savefig(out / "plots" / "figure4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    cfg = parse_config()
    out = output_dir(cfg["_name"])
    (out / "plots").mkdir(exist_ok=True)

    fig4 = run_figure4(cfg, out)
    print(fig4.round(3).to_string(index=False))
    fig3 = run_figure3(cfg, out)
    print(fig3[fig3.n_evals.isin([min(cfg["n_evals"]), max(cfg["n_evals"])])].round(3).to_string(index=False))

    save_json(env_info(), out / "env.json")
    save_json(cfg, out / "config.json")


if __name__ == "__main__":
    main()
