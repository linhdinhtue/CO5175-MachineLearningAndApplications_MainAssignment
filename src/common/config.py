"""Read experiment configuration without changing its scientific settings."""
import argparse
import json
from pathlib import Path

DEFAULT_CONFIGS = {
    "data.check": "configs/data/data_check.json",
    "modeling.baseline": "configs/modeling/baseline.json",
    "modeling.ablation": "configs/modeling/ablation.json",
    "explainability.explain": "configs/explainability/explain.json",
    "explainability.stability": "configs/explainability/stability.json",
    "reproduction.reproduce_shap": "configs/reproduction/reproduce_shap_v2.json",
    "analysis.supplementary": "configs/analysis/analysis.json",
}


def default_config(task: str) -> str:
    """Resolve CI defaults; tuning must name its model-specific config explicitly."""
    if task not in DEFAULT_CONFIGS:
        raise ValueError(f"Specify a config explicitly for task: {task}")
    return DEFAULT_CONFIGS[task]


def parse_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = args.config
    cfg["_name"] = Path(args.config).stem
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the default config path for a workflow task.")
    parser.add_argument("task")
    args = parser.parse_args()
    try:
        print(default_config(args.task))
    except ValueError as error:
        parser.error(str(error))

