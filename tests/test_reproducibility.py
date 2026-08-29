import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "data/hmda/hmda_Orleans_X_train_1.csv": "4f775a0fc5a732cf98ab98ce413d7d199f2bac381f1c5c76812675cc839ee797",
    "data/hmda/hmda_Orleans_X_test_1.csv": "5862c162990bf7d135f41300fe9c2e2f0bf9caed78461f3bd9212ae7a5f400c7",
    "historical_data/tutorial/train_profile_lr_accuracy_score_hmda.csv": "8075458f608b175c5b8ed4e89477e654d4825fa97b9905e3c2938257fd1c5e96",
    "historical_data/tutorial/test_current_profile_lr_accuracy_score_hmda.csv": "ebdfeedccdc9bd8e1a8f0cc88622a8d01daecabddba7f2f1cf25fc0c812d8d71",
}


class ReproducibilitySmokeTest(unittest.TestCase):
    def test_core_imports(self):
        from glassbox_optimizer import GlassBoxOptimizer  # noqa: F401
        from opaque_optimizer import OpaqueOptimizer  # noqa: F401
        from pipeline_execution import PipelineExecutor  # noqa: F401

    def test_example_configuration(self):
        config = json.loads((ROOT / "config_example.json").read_text())
        self.assertEqual(config["dataset_name"], "hmda")
        self.assertEqual(config["pipeline_type"], "ml")

    def test_tutorial_data_checksums(self):
        for relative_path, expected in EXPECTED_SHA256.items():
            digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative_path)


if __name__ == "__main__":
    unittest.main()
