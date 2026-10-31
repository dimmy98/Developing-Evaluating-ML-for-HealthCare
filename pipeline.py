"""Train reproducible insurance-charge experiments and export browser-readable models."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["age", "sex", "bmi", "smoker"]


def load_data(path):
    raw = pd.read_csv(path)
    missing = set(FEATURES + ["charges"]) - set(raw.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = raw.copy()
    for col, mapping in [("sex", {"female": 0, "male": 1}), ("smoker", {"no": 0, "yes": 1})]:
        frame[col] = frame[col].astype(str).str.strip().str.lower().map(mapping)
    for col in ["age", "bmi", "charges"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if not np.isfinite(frame[FEATURES + ["charges"]].to_numpy(dtype=float)).all():
        raise ValueError("Missing, non-finite or unsupported values in model columns")
    if not frame.age.between(18, 100).all() or not (frame.age % 1 == 0).all() or not frame.bmi.between(10, 70).all():
        raise ValueError("Age must be a whole number from 18–100 and BMI must be 10–70")
    if not (frame.charges > 0).all():
        raise ValueError("Charges must be positive")
    # Deduplicate before splitting, using the source record's complete set of fields.
    frame = frame.drop_duplicates().reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError("At least 30 unique records are needed")
    return frame, len(raw) - len(frame)


def export_tree(estimator):
    t = estimator.tree_
    return {"left": t.children_left.tolist(), "right": t.children_right.tolist(),
            "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
            "value": t.value[:, 0, 0].tolist()}


def predict_exported(model, rows):
    """Independent portable inference, also used to verify the JSON export."""
    x = np.asarray(rows, dtype=np.float32 if model["kind"] != "linear" else float)
    def tree_prediction(tree):
        result = []
        for row in x:
            i = 0
            while tree["left"][i] != -1:
                i = tree["left"][i] if row[tree["feature"][i]] <= tree["threshold"][i] else tree["right"][i]
            result.append(tree["value"][i])
        return np.asarray(result)
    if model["kind"] == "linear":
        return x @ np.asarray(model["coef"]) + model["intercept"]
    if model["kind"] == "tree":
        return tree_prediction(model["tree"])
    return model["initial"] + model["learning_rate"] * sum(tree_prediction(t) for t in model["trees"])


def train(path, seed=42, test_size=0.2, source="User-supplied CSV"):
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test fraction must be between 0.1 and 0.4")
    frame, duplicates = load_data(path)
    x = frame[FEATURES].to_numpy(dtype=float)
    y = np.log10(frame.charges.to_numpy(dtype=float))
    train_ix, test_ix = train_test_split(np.arange(len(frame)), test_size=test_size, random_state=seed)
    xtrain, xtest, ytrain, ytest = x[train_ix], x[test_ix], y[train_ix], y[test_ix]
    estimators = {
        "linear": LinearRegression(),
        "tree": DecisionTreeRegressor(max_depth=30, min_samples_split=50, min_samples_leaf=10, random_state=seed),
        "boost": GradientBoostingRegressor(random_state=seed),
    }
    models = []
    for key, estimator in estimators.items():
        estimator.fit(xtrain, ytrain)
        yp, ytp = estimator.predict(xtest), estimator.predict(xtrain)
        if key == "linear":
            portable = {"kind": "linear", "coef": estimator.coef_.tolist(), "intercept": float(estimator.intercept_)}
        elif key == "tree":
            portable = {"kind": "tree", "tree": export_tree(estimator)}
        else:
            portable = {"kind": "boost", "initial": float(estimator.init_.constant_[0, 0]),
                        "learning_rate": estimator.learning_rate,
                        "trees": [export_tree(e[0]) for e in estimator.estimators_]}
        # Verify all rows, including branch thresholds cast to float32 like sklearn.
        np.testing.assert_allclose(predict_exported(portable, x), estimator.predict(x), rtol=1e-10, atol=1e-10)
        importance = permutation_importance(estimator, xtest, ytest, n_repeats=10, random_state=seed, scoring="r2")
        models.append({"id": key, "name": {"linear": "Linear regression", "tree": "Decision tree", "boost": "Gradient boosting"}[key],
                       **portable,
                       "metrics": {"test_r2": float(r2_score(ytest, yp)), "train_r2": float(r2_score(ytrain, ytp)),
                                   "test_mse_log10": float(mean_squared_error(ytest, yp)),
                                   "mae_usd": float(mean_absolute_error(10 ** ytest, 10 ** yp))},
                       "importance": importance.importances_mean.tolist(),
                       "importance_std": importance.importances_std.tolist(),
                       "test_predictions_log10": yp.tolist()})
    # Only aggregate diagnostics are published: no original test rows are exported.
    for model in models:
        predictions = np.asarray(model.pop("test_predictions_log10"))
        actual, predicted = 10 ** ytest, 10 ** predictions
        residual = predicted - actual
        # Bin actual/predicted pairs to retain the chart without republishing source rows.
        edges = np.linspace(0, max(float(actual.max()), float(predicted.max())) * 1.01, 19)
        counts, xe, ye = np.histogram2d(actual, predicted, bins=(edges, edges))
        model["density"] = {"edges": edges.tolist(), "counts": counts.astype(int).tolist()}
        hist, bins = np.histogram(residual, bins=18)
        model["residuals"] = {"counts": hist.tolist(), "edges": bins.tolist()}
        groups = []
        for label, mask in [("Non-smoker", xtest[:, 3] == 0), ("Smoker", xtest[:, 3] == 1),
                            ("Age 18–34", xtest[:, 0] < 35), ("Age 35–49", (xtest[:, 0] >= 35) & (xtest[:, 0] < 50)),
                            ("Age 50+", xtest[:, 0] >= 50)]:
            if mask.any():
                groups.append({"label": label, "n": int(mask.sum()), "mae": float(np.mean(np.abs(residual[mask])))})
        model["groups"] = groups
    return {
        "schema_version": 1,
        "metadata": {"source": source, "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "seed": seed, "test_fraction": test_size, "rows": len(frame), "duplicates_removed": duplicates,
                     "train_rows": len(train_ix), "test_rows": len(test_ix), "sklearn_version": sklearn.__version__,
                     "python_version": platform.python_version(), "target": "log10(charges)",
                     "note": "New reproducible experiment; not a reproduction of the dissertation's historical metrics."},
        "features": FEATURES,
        "bounds": {f: [float(xtrain[:, i].min()), float(xtrain[:, i].max())] for i, f in enumerate(FEATURES)},
        "reference": [float(np.median(xtrain[:, 0])), int(pd.Series(xtrain[:, 1]).mode().iloc[0]),
                      float(np.median(xtrain[:, 2])), int(pd.Series(xtrain[:, 3]).mode().iloc[0])],
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("lab/assets/experiment.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--source", default="User-supplied CSV")
    args = parser.parse_args()
    result = train(args.data, args.seed, args.test_size, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for model in result["models"]:
        print(f'{model["name"]}: test R2={model["metrics"]["test_r2"]:.4f}; MAE USD={model["metrics"]["mae_usd"]:.2f}')
    print(f"Verified portable predictions and saved {args.output}")


if __name__ == "__main__":
    main()
