import millwright as mw

frame: mw.Frame = mw.Frame.from_rows([[0.0], [1.0]], ["x"])
pipeline: mw.Pipeline = mw.Pipeline().estimator("rf", mw.RandomForest(n_trees=10))
pipeline.fit(frame, [0.0, 1.0])
predictions: list[float] = pipeline.predict(frame)
shape: tuple[int, int] = frame.shape
