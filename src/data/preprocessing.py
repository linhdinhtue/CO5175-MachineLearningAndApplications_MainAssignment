import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin


def build_preprocessor(X: pd.DataFrame, scale: bool, drop_first: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # num_steps = [("impute", SimpleImputer(strategy="median"))]
    num_steps: list[tuple[str, TransformerMixin]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]

    if scale:
        num_steps.append(("scale", StandardScaler()))

    # Use a reference category to avoid redundant dummy variables and aid coefficient interpretation.
    encoder = OneHotEncoder(
        drop="first" if drop_first else None,
        handle_unknown="ignore",
        sparse_output=False,
    )
    # category by mod
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)])

    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols), ("cat", cat_pipe, cat_cols)],
        verbose_feature_names_out=True,
    )


def reference_categories(pre: ColumnTransformer) -> dict:
    """Dropped (reference) category per categorical column of a fitted preprocessor."""
    enc = pre.named_transformers_["cat"].named_steps["onehot"]
    if enc.drop_idx_ is None:
        return {}
    cols = pre.transformers_[1][2]
    return {
        col: str(cats[idx])
        for col, cats, idx in zip(cols, enc.categories_, enc.drop_idx_)
        if idx is not None
    }


def original_feature_of_columns(pre: ColumnTransformer) -> list[str]:
    """Original column name for every output column of a fitted preprocessor, in order."""
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    enc = pre.named_transformers_["cat"].named_steps["onehot"]
    drops = enc.drop_idx_ if enc.drop_idx_ is not None else [None] * len(cat_cols)
    groups = list(num_cols)
    for col, cats, idx in zip(cat_cols, enc.categories_, drops):
        groups += [col] * (len(cats) - (idx is not None))
    return groups
