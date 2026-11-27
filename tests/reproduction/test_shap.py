import json
import unittest

import numpy as np

from src.common.artifacts import ROOT


class ShapTests(unittest.TestCase):
    def test_exact_and_kernel_shap_match_known_game(self):
        from experiments.reproduction.reproduce_shap import exact_shapley, kernel_shap
        f = lambda z: (np.asarray(z) * np.array([5., 4., 0.])).max(axis=1)
        exact, base = exact_shapley(f, 3, [0, 1, 2])
        np.testing.assert_allclose(exact, [3., 2., 0.])
        np.testing.assert_allclose(kernel_shap(f, 3, 8, 0, False), exact, atol=1e-12)
        self.assertEqual(base, 0.)


    def test_reproduction_versions_keep_distinct_budgets(self):
        a = json.loads((ROOT / "configs/reproduction/reproduce_shap.json").read_text())
        b = json.loads((ROOT / "configs/reproduction/reproduce_shap_v2.json").read_text())
        self.assertEqual(a["sampling_budget"], "single_feature")
        self.assertEqual(b["sampling_budget"], "all_features")



if __name__ == "__main__":
    unittest.main()
