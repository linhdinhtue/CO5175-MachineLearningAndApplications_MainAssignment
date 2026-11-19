import argparse
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = args.config
    return cfg


def output_dir(name: str) -> Path:
    path = ROOT / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def env_info() -> dict:
    pkgs = {}
    for name in ["numpy", "pandas", "scikit-learn", "shap", "scipy"]:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        ram_gb = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "packages": pkgs,
    }
