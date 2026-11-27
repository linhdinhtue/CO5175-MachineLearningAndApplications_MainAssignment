"""Integration contracts that can break when modules/configs move into packages."""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.artifacts import ROOT, env_info, output_dir
from src.common.config import DEFAULT_CONFIGS, default_config, parse_config


class LayoutTests(unittest.TestCase):
    def test_suite_entrypoints_and_configs_resolve(self):
        suites = json.loads((ROOT / "configs/suites.json").read_text())
        for jobs in suites.values():
            output_names = []
            for task, config in jobs:
                module = importlib.import_module(f"experiments.{task}")
                self.assertTrue(callable(module.main))
                path = ROOT / "configs" / f"{config}.json"
                self.assertTrue(path.is_file(), path)
                json.loads(path.read_text())
                output_names.append(path.stem)
            self.assertEqual(len(output_names), len(set(output_names)))

    def test_workflow_defaults_match_new_layout(self):
        for task, config in DEFAULT_CONFIGS.items():
            importlib.import_module(f"experiments.{task}")
            self.assertTrue((ROOT / default_config(task)).is_file())
        with self.assertRaises(ValueError):
            default_config("modeling.tuning")

    def test_nested_config_preserves_artifact_name(self):
        path = ROOT / "configs/modeling/baseline.json"
        with patch("sys.argv", ["baseline", "--config", str(path)]):
            cfg = parse_config()
        self.assertEqual(cfg["_name"], "baseline")
        self.assertEqual(cfg["seeds"], [42, 2026, 5175])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ML_RESULTS_ROOT": tmp}):
            self.assertEqual(output_dir(cfg["_name"]), Path(tmp) / "baseline")

    def test_root_and_recursive_source_hashes(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        hashes = {Path(p).as_posix(): digest for p, digest in env_info()["source_sha256"].items()}
        self.assertIn("src/data/loader.py", hashes)
        self.assertIn("experiments/reproduction/reproduce_shap.py", hashes)
        self.assertTrue(all(len(digest) == 64 for digest in hashes.values()))


if __name__ == "__main__":
    unittest.main()
