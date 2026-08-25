"""Dependency-free ABI smoke test, including the supported Python 3.9 floor."""

import millwright as mw


frame = mw.Frame.from_rows([[0.0], [1.0], [8.0], [9.0]], ["x"])
pipeline = mw.Pipeline().estimator(
    "rf", mw.RandomForest(n_trees=8, max_depth=2)
)
pipeline.fit(frame, [0.0, 0.0, 1.0, 1.0])
predictions = pipeline.predict(frame)

assert len(predictions) == 4
assert mw.version() == mw.__version__
