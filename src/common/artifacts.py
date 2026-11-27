import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def output_dir(name: str) -> Path:
    # A reproduction runner sets this to an isolated directory for each new run.
    root = Path(os.environ.get("ML_RESULTS_ROOT", str(ROOT / "results")))
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def env_info() -> dict:
    pkgs = {}
    for name in ["numpy", "pandas", "scikit-learn", "shap", "scipy", "lime", "interpret-core"]:
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
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "experiments") for p in sorted((ROOT / folder).rglob("*.py"))
        },
    }
