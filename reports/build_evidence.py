"""Verify archived predictions and generate report tables with source hashes.

This does not train models and must not be described as a new experiment.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.common.artifacts import ROOT

ARTIFACTS = {
    "Logistic L1": "tuning_logistic_l1-3",
    "Logistic L2": "tuning_logistic_l2-3",
    "Logistic Elastic Net": "tuning_logistic_elasticnet-3",
    "GBDT": "tuning_gbdt-3",
    "GBDT extended (exploratory)": "tuning_gbdt_extended-4",
    "EBM GAM (1 config)": "tuning_ebm_gam-7",
    "EBM GA2M (1 config)": "tuning_ebm_ga2m-7",
}
METRICS = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy", "log_loss"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evidence.md")
    args = parser.parse_args()
    rows, source_rows = [], []
    count = 0
    for label, folder in ARTIFACTS.items():
        path = args.artifacts_root / folder
        frame = pd.read_csv(path / "test_metrics.csv")
        if sorted(frame.seed.tolist()) != [42, 2026, 5175]:
            raise ValueError(f"Unexpected/missing seeds: {folder}")
        sources = [path / "test_metrics.csv", path / "config.json", path / "env.json"]
        for _, row in frame.iterrows():
            prediction_path = path / f"predictions_seed{int(row.seed)}.csv"
            pred = pd.read_csv(prediction_path)
            actual = classification_metrics(pred.y_true, pred.proba)
            for metric in METRICS:
                if not np.isclose(actual[metric], row[metric], atol=1e-10, rtol=0):
                    raise ValueError(f"Metric mismatch: {folder}, seed={row.seed}, {metric}")
            count += 1
            sources.append(prediction_path)
        rows.append([label] + [f"{frame[m].mean():.4f} ± {frame[m].std(ddof=1):.4f}" for m in METRICS])
        for source in sources:
            source_rows.append([f"{folder}/{source.name}", hashlib.sha256(source.read_bytes()).hexdigest()])
    content = "# Bảng bằng chứng tự sinh từ artifact của nhóm\n\n"
    content += f"Đã tính lại {len(METRICS)} metric từ {count} tệp dự đoán; sai số cho phép 1e-10. "
    content += "Đây là kiểm tra artifact cũ, không phải huấn luyện lại. Std dùng ddof=1.\n\n"
    content += table(["Model", *METRICS], rows)
    content += "\n\n## Ablation tại C=1\n\n"
    ablation_path = args.artifacts_root / "ablation-3" / "ablation_metrics.csv"
    ablation = pd.read_csv(ablation_path)
    content += table(["Penalty", "ROC-AUC", "Active coefficients", "Refit seconds"], [
        [name] + [f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}" for m in ["roc_auc", "n_active_coef", "refit_seconds"]]
        for name, g in ablation.groupby("variant")
    ])
    source_rows.append(["ablation-3/ablation_metrics.csv", hashlib.sha256(ablation_path.read_bytes()).hexdigest()])
    content += "\n\n## Nguồn và SHA-256\n\nCác đường dẫn dưới đây tính từ thư mục artifact đầu vào.\n\n"
    content += table(["Artifact file", "SHA-256"], source_rows) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Verified {count} prediction files; wrote {args.output}")


if __name__ == "__main__":
    main()
