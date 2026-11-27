import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.common.artifacts import ROOT

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
RAW_DIR = ROOT / "data" / "raw"
TARGET = "income"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", TARGET,
]
FILES = ["adult.data", "adult.test", "adult.names"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_adult(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / name).exists() for name in FILES):
        return
    with urllib.request.urlopen(ADULT_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in FILES:
            (raw_dir / name).write_bytes(zf.read(name))


def _read(path: Path, skiprows: int) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, names=COLUMNS, skiprows=skiprows,
        skipinitialspace=True, na_values="?", comment=None,
    )
    # drop space after delimiter and ? to na for processing
    df = df.dropna(how="all")
    # adult.test labels end with "." (e.g. ">50K.")
    df[TARGET] = df[TARGET].str.rstrip(".").str.strip()
    return df.reset_index(drop=True)


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    download_adult(raw_dir)
    train = _read(raw_dir / "adult.data", skiprows=0)
    # first line of adult.test is "|1x3 Cross validator"
    test = _read(raw_dir / "adult.test", skiprows=1)
    return train, test


def split_xy(df: pd.DataFrame, drop_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    y = (df[TARGET] == ">50K").astype(int)
    X = df.drop(columns=[TARGET, *drop_features])
    return X, y

# main
def load_adult(drop_features: list[str], raw_dir: Path = RAW_DIR):
    train, test = load_raw(raw_dir)
    X_train, y_train = split_xy(train, drop_features)
    X_test, y_test = split_xy(test, drop_features)
    return X_train, y_train, X_test, y_test


def describe(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum()
    return {
        "n_rows": int(len(df)),
        "n_duplicates": int(df.duplicated().sum()),
        "label_counts": df[TARGET].value_counts().to_dict(),
        "missing_per_column": {k: int(v) for k, v in missing_counts.items() if v > 0},
    }
