"""Generate deterministic test points for the currently committed model export."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import predict_exported

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "lab/assets/experiment.json").read_text(encoding="utf-8-sig"))
rng = np.random.default_rng(42)
points = []
for _ in range(120):
    points.append([float(rng.uniform(*report["bounds"]["age"])), int(rng.integers(0,2)),
                   float(rng.uniform(*report["bounds"]["bmi"])), int(rng.integers(0,2))])
for model in report["models"]:
    trees = [model["tree"]] if model["kind"] == "tree" else model.get("trees", [])
    for tree in trees[:8]:
        for f, threshold in zip(tree["feature"], tree["threshold"]):
            if f < 0:
                continue
            # Check both sides of the exact float32 boundary.
            value = np.float32(threshold)
            for v in [np.nextafter(value, np.float32(-np.inf)), value, np.nextafter(value, np.float32(np.inf))]:
                x = report["reference"].copy()
                x[f] = float(v)
                points.append(x)
fixtures = [{"x": x, "expected": [float(predict_exported(m,[x])[0]) for m in report["models"]]} for x in points]
(root / "tests/parity.json").write_text(json.dumps(fixtures, separators=(",",":"), allow_nan=False)+"\n",encoding="utf-8")
print(f"Exported {len(points)} inference fixtures")
