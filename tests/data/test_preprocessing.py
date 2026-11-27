import unittest
import warnings

import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_never_fits_validation_values(self):
        train = pd.DataFrame({"number": [1., 2., np.nan], "category": ["a", "b", "a"]})
        validation = pd.DataFrame({"number": [1000.], "category": ["new"]})
        pre = build_preprocessor(train, scale=True, drop_first=True).fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre.transform(validation)
        self.assertEqual(pre.named_transformers_["num"].named_steps["impute"].statistics_[0], 1.5)
        self.assertNotIn("new", pre.named_transformers_["cat"].named_steps["onehot"].categories_[0])



if __name__ == "__main__":
    unittest.main()
