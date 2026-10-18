"""Input validation for predictor thresholds (does not require model weights)."""
import unittest

import torch

from app.inference.predict import ChestXpertPredictor


class PredictionThresholdTests(unittest.TestCase):
    def test_invalid_thresholds_fail_before_inference(self):
        predictor = object.__new__(ChestXpertPredictor)
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(threshold=repr(threshold)):
                with self.assertRaises(ValueError):
                    predictor._predict_tensor(torch.zeros(1, 3, 4, 4), threshold=threshold)


if __name__ == "__main__":
    unittest.main()
