import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor

ACTIVE_COEF_THRESHOLD = 1e-6


def build_pipeline(model_cfg: dict, X: pd.DataFrame, seed: int) -> Pipeline:
    params = {**model_cfg.get("fixed", {}), "random_state": seed}
    kind = model_cfg["type"]
    if kind == "logistic":
        pre = build_preprocessor(X, scale=True, drop_first=True)
        clf = LogisticRegression(**params)
    elif kind == "gbdt":
        pre = build_preprocessor(X, scale=False, drop_first=False)
        clf = GradientBoostingClassifier(**params)
    elif kind == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        # EBM bins numeric and encodes categorical/missing values itself
        return Pipeline([("clf", ExplainableBoostingClassifier(**params))])
    else:
        raise ValueError(f"Unknown model type: {kind}")
    return Pipeline([("pre", pre), ("clf", clf)])


def expand_grid(grid: dict) -> dict:
    """Prefix params with clf__ and expand {"logspace": [start, stop, num]} specs."""
    out = {}
    for name, values in grid.items():
        if isinstance(values, dict) and "logspace" in values:
            start, stop, num = values["logspace"]
            values = np.logspace(start, stop, int(num)).tolist()
        out[f"clf__{name}"] = values
    return out


def complexity(pipe: Pipeline) -> dict:
    clf = pipe.named_steps["clf"]
    info = {"model_bytes": len(pickle.dumps(pipe))}
    if isinstance(clf, LogisticRegression):
        info.update(
            n_coef=int(clf.coef_.size),
            n_intercept=int(clf.intercept_.size),
            n_active_coef=int((np.abs(clf.coef_) > ACTIVE_COEF_THRESHOLD).sum()),
            n_iter=int(clf.n_iter_.max()),
            max_iter_reached=bool(clf.n_iter_.max() >= clf.max_iter),
        )
    elif hasattr(clf, "term_features_"):
        info.update(
            n_terms=len(clf.term_features_),
            n_interaction_terms=int(sum(len(t) > 1 for t in clf.term_features_)),
            n_term_scores=int(sum(np.asarray(s).size for s in clf.term_scores_)),
        )
    else:
        trees = clf.estimators_.ravel()
        info.update(
            n_trees=len(trees),
            n_nodes=int(sum(t.tree_.node_count for t in trees)),
            n_leaves=int(sum(t.tree_.n_leaves for t in trees)),
        )
    return info


def feature_table(pipe: Pipeline) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "term_features_"):
        return pd.DataFrame({"feature": clf.term_names_, "term_importance": clf.term_importances()})
    names = pipe.named_steps["pre"].get_feature_names_out()
    if isinstance(clf, LogisticRegression):
        coef = clf.coef_[0]
        return pd.DataFrame({
            "feature": names,
            "coef": coef,
            "odds_ratio": np.exp(coef),
            "active": np.abs(coef) > ACTIVE_COEF_THRESHOLD,
        })
    return pd.DataFrame({"feature": names, "impurity_importance": clf.feature_importances_})
