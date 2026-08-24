import math
import tempfile
from pathlib import Path

import millwright as mw


frame = mw.Frame.from_rows([[0.0], [1.0], [9.0], [10.0]], ["x"])
pipeline = mw.Pipeline()
pipeline.estimator("rf", mw.RandomForest(n_trees=20, max_depth=3))
pipeline.fit(frame, [0.0, 0.0, 1.0, 1.0])
predictions = pipeline.predict(frame)
assert predictions == [0.0, 0.0, 1.0, 1.0]
metrics = pipeline.evaluate(frame, [0.0, 0.0, 1.0, 1.0])
assert math.isclose(metrics["accuracy"], 1.0)

try:
    mw.Frame.from_rows([[1.0], [2.0, 3.0]])
except ValueError:
    pass
else:
    raise AssertionError("ragged Python input must be rejected")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "model.onnx"
    pipeline.export_onnx(str(path))
    assert path.is_file() and path.stat().st_size > 0

print("millwright Python fit/predict/evaluate/export smoke test passed")
