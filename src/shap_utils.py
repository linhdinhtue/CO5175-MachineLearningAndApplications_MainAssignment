import pandas as pd
import shap

from src.preprocess import original_feature_of_columns


def transform(pipe, X: pd.DataFrame) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    return pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)


def shap_frame(pipe, model_type: str, X_bg_raw: pd.DataFrame, X_ex_raw: pd.DataFrame) -> pd.DataFrame:
    """Interventional SHAP values (raw log-odds) per encoded column, indexed like X_ex_raw."""
    clf = pipe.named_steps["clf"]
    X_bg, X_ex = transform(pipe, X_bg_raw), transform(pipe, X_ex_raw)
    if model_type == "gbdt":
        explainer = shap.TreeExplainer(clf, data=X_bg, feature_perturbation="interventional", model_output="raw")
    else:
        explainer = shap.LinearExplainer(clf, X_bg)
    return pd.DataFrame(explainer(X_ex).values, columns=X_ex.columns, index=X_ex.index)


def group_by_original(values: pd.DataFrame, pipe) -> pd.DataFrame:
    """Sum one-hot columns back to their original feature, per row."""
    groups = original_feature_of_columns(pipe.named_steps["pre"])
    return values.set_axis(groups, axis=1).T.groupby(level=0, sort=False).sum().T
