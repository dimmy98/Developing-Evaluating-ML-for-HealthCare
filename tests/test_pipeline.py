import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import load_data, predict_exported, train

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.path = Path(self.temp.name) / "fixture.csv"
        rng = np.random.default_rng(9)
        n = 90
        age = rng.integers(18, 65, n)
        bmi = rng.uniform(18, 40, n)
        smoker = rng.choice(["yes", "no"], n)
        self.frame = pd.DataFrame({"age": age, "sex": rng.choice(["female","male"], n),
            "bmi": bmi, "smoker": smoker, "charges": np.exp(7 + age*.02 + bmi*.015 + (smoker=="yes")*.7)})
        self.frame.to_csv(self.path, index=False)
    def tearDown(self):
        self.temp.cleanup()
    def test_invalid_values_rejected(self):
        for column, value in [("age", np.inf), ("age", 35.5), ("bmi", -1), ("charges", 0), ("sex", "unsupported"), ("smoker", None)]:
            frame = self.frame.copy()
            frame[column] = frame[column].astype(object)
            frame.loc[0, column] = value
            frame.to_csv(self.path, index=False)
            with self.assertRaises(ValueError):
                load_data(self.path)
    def test_duplicates_removed_before_split(self):
        pd.concat([self.frame, self.frame.iloc[:3]]).to_csv(self.path, index=False)
        frame, removed = load_data(self.path)
        self.assertEqual(removed, 3)
        self.assertEqual(len(frame), 90)
    def test_export_is_reproducible_and_finite(self):
        one, two = train(self.path), train(self.path)
        self.assertEqual(one, two)
        json.dumps(one, allow_nan=False)
        self.assertEqual(one["metadata"]["train_rows"] + one["metadata"]["test_rows"], 90)
        for model in one["models"]:
            self.assertEqual(sum(model["residuals"]["counts"]), one["metadata"]["test_rows"])
            self.assertEqual(sum(map(sum, model["density"]["counts"])), one["metadata"]["test_rows"])
            self.assertTrue(np.isfinite(predict_exported(model, [one["reference"]])).all())
    def test_missing_column_and_too_small_input(self):
        self.frame.drop(columns="bmi").to_csv(self.path,index=False)
        with self.assertRaises(ValueError): load_data(self.path)
        self.frame.iloc[:10].to_csv(self.path,index=False)
        with self.assertRaises(ValueError): load_data(self.path)

if __name__ == "__main__":
    unittest.main()
