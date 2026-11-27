"""Plan or execute an ordered study suite, without overwriting historical artifacts."""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.artifacts import ROOT, env_info, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["baseline", "core", "full"], default="baseline")
    parser.add_argument("--run-id", required=True, help="New directory name, e.g. linh-baseline-01")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the plan only")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must contain only letters, digits, underscores and hyphens")
    manifest = json.loads((ROOT / "configs" / "suites.json").read_text(encoding="utf-8"))
    jobs = manifest[args.suite]
    out = ROOT / "results" / "reproductions" / args.run_id
    for task, config in jobs:
        print(f"{sys.executable} -m experiments.{task} --config configs/{config}.json", flush=True)
    print(f"Output: {out}", flush=True)
    if not args.execute:
        print("Plan only. Add --execute to run.")
        return
    if args.suite == "full" and importlib.util.find_spec("interpret") is None:
        parser.error("Full suite needs interpret-core. Install requirements.txt before running.")
    # Refuse reusing a run ID, including partially completed runs.
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ML_RESULTS_ROOT=str(out), PYTHONUTF8="1")
    record = {"suite": args.suite, "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": env_info(), "status": "running", "jobs": []}
    save_json(record, out / "run.json")
    for task, config in jobs:
        command = [sys.executable, "-m", f"experiments.{task}", "--config", f"configs/{config}.json"]
        start = time.perf_counter()
        log_path = out / f"{Path(config).name}.log"
        print(f"Running {config}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["jobs"].append({"task": task, "config": config, "exit_code": result.returncode,
                               "seconds": time.perf_counter() - start})
        if result.returncode:
            record["status"] = "failed"
            save_json(record, out / "run.json")
            raise SystemExit(f"{config} failed. See its log; existing artifacts were retained.")
        save_json(record, out / "run.json")
    record["status"] = "complete"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(record, out / "run.json")
    print("Suite complete.")


if __name__ == "__main__":
    main()
