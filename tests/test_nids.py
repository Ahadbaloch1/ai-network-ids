import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nids import data, features, model  # noqa: E402
from nids.cli import main  # noqa: E402


class DataTests(unittest.TestCase):
    def test_shape_and_labels(self):
        df = data.generate(500, seed=1)
        self.assertEqual(len(df), 500)
        self.assertEqual(list(df.columns), data.RAW_COLUMNS + ["label"])
        self.assertTrue(set(df["label"]).issubset(set(data.CLASSES)))

    def test_reproducible(self):
        a = data.generate(200, seed=5)
        b = data.generate(200, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_unlabeled(self):
        self.assertNotIn("label", data.generate(50, seed=2, labeled=False).columns)

    def test_invalid_sample_count(self):
        with self.assertRaises(ValueError):
            data.generate(0)


class FeatureTests(unittest.TestCase):
    def test_feature_columns(self):
        df = data.generate(100, seed=3, labeled=False)
        X = features.build_features(df)
        self.assertEqual(list(X.columns), features.FEATURE_NAMES)
        self.assertFalse(X.isna().any().any())

    def test_missing_column(self):
        df = data.generate(10, seed=3, labeled=False).drop(columns=["syn_ratio"])
        with self.assertRaises(ValueError):
            features.build_features(df)

    def test_missing_values(self):
        df = data.generate(10, seed=3, labeled=False)
        df.loc[0, "duration"] = None
        with self.assertRaises(ValueError):
            features.build_features(df)

    def test_protocol_one_hot(self):
        df = data.generate(50, seed=4, labeled=False)
        X = features.build_features(df)
        self.assertTrue(((X[["proto_tcp", "proto_udp", "proto_icmp"]].sum(axis=1)) == 1).all())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.generate(3000, seed=11)
        cls.bundle, cls.metrics = model.train(cls.df, seed=11, n_estimators=60)

    def test_accuracy_reasonable(self):
        self.assertGreater(self.metrics["accuracy"], 0.85)

    def test_predict_columns(self):
        new = data.generate(40, seed=12, labeled=False)
        result = model.predict(self.bundle, new)
        for col in ("prediction", "confidence", "anomaly", "alert"):
            self.assertIn(col, result.columns)
        self.assertEqual(len(result), 40)
        self.assertTrue(result["confidence"].between(0, 1).all())

    def test_obvious_port_scan_detected(self):
        scan = pd.DataFrame([{
            "duration": 0.01, "protocol": "tcp", "dst_port": 31337, "src_bytes": 60,
            "dst_bytes": 0, "packets": 1, "syn_ratio": 1.0, "conn_count": 200,
            "unique_ports": 180, "failed_auth_ratio": 0.0}])
        self.assertEqual(model.predict(self.bundle, scan)["prediction"].iloc[0], "port_scan")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.joblib")
            model.save_model(self.bundle, path)
            loaded = model.load_model(path)
            new = data.generate(20, seed=13, labeled=False)
            pd.testing.assert_frame_equal(model.predict(self.bundle, new), model.predict(loaded, new))

    def test_invalid_model_file(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.joblib")
            joblib.dump({"hello": 1}, path)
            with self.assertRaises(ValueError):
                model.load_model(path)

    def test_training_requires_label(self):
        with self.assertRaises(ValueError):
            model.train(data.generate(100, seed=1, labeled=False))


class CliTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            flows = os.path.join(tmp, "flows.csv")
            mdl = os.path.join(tmp, "model.joblib")
            out = os.path.join(tmp, "pred.csv")
            self.assertEqual(main(["generate", "--samples", "1500", "--out", flows]), 0)
            self.assertEqual(main(["train", "--data", flows, "--model", mdl,
                                   "--report", os.path.join(tmp, "rep")]), 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "rep", "confusion_matrix.png")))
            self.assertEqual(main(["predict", "--model", mdl, "--data", flows, "--out", out]), 0)
            self.assertIn("prediction", pd.read_csv(out).columns)

    def test_missing_file_returns_error(self):
        self.assertEqual(main(["train", "--data", "does-not-exist.csv"]), 1)


if __name__ == "__main__":
    unittest.main()
