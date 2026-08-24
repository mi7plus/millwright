import math
import tempfile
from pathlib import Path

import millwright as mw

assert mw.version() == "2.2.0"

frame = mw.Frame.from_rows([[0.0], [1.0], [9.0], [10.0]], ["x"])
assert frame.shape == (4, 1)
assert frame.columns() == ["x"]
assert len(frame) == 4

unfitted = mw.Pipeline()
try:
    unfitted.predict(frame)
except ValueError as error:
    assert "not fitted" in str(error)
else:
    raise AssertionError("predicting before fit must fail")

pipeline = mw.Pipeline()
pipeline.step("impute", mw.SimpleImputer.mean())
pipeline.step("scale", mw.StandardScaler())
pipeline.estimator("rf", mw.RandomForest(n_trees=20, max_depth=3))
assert pipeline.steps() == ["impute", "scale", "rf"]
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

try:
    mw.Pipeline().step("bad", mw.SimpleImputer("unknown"))
except ValueError as error:
    assert "unknown strategy" in str(error)
else:
    raise AssertionError("unknown imputation strategies must be rejected")

with tempfile.TemporaryDirectory() as directory:
    report = Path(directory) / "profile.html"
    table = mw.Table.from_frame(frame)
    assert table.shape == frame.shape
    assert table.to_frame().shape == frame.shape
    mw.Profile.of(table).to_html(str(report))
    assert report.is_file() and report.stat().st_size > 0

search_pipeline = mw.Pipeline().estimator("rf", mw.RandomForest(n_trees=10, max_depth=2))
search = mw.GridSearch(
    search_pipeline,
    {"rf__n_trees": [5, 10]},
    cv=mw.StratifiedKFold(2),
    scoring="accuracy",
)
best = search.fit(frame, [0.0, 0.0, 1.0, 1.0])
assert 0.0 <= best.best_score <= 1.0
assert best.best_params()["rf__n_trees"] in (5, 10)
assert len(best.predict(frame)) == len(frame)

importance = pipeline.explain(frame, mw.Explainer.kernel().nsamples(8).background(2))
assert len(importance) == 1 and importance[0][0] == "x"

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "model.onnx"
    pipeline.export_onnx(str(path))
    assert path.is_file() and path.stat().st_size > 0

    loaded = mw.Pipeline().estimator("onnx", mw.OnnxModel(str(path)))
    loaded.fit(frame, [0.0, 0.0, 1.0, 1.0])
    assert loaded.predict(frame) == predictions

print("millwright Python fit/predict/evaluate/export smoke test passed")
