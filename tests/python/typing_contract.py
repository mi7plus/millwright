import millwright as mw

frame: mw.Frame = mw.Frame.from_rows([[0.0], [1.0]], ["x"])
pipeline: mw.Pipeline = mw.Pipeline().estimator("rf", mw.RandomForest(n_trees=10))
pipeline.fit(frame, [0.0, 1.0])
predictions: list[float] = pipeline.predict(frame)
shape: tuple[int, int] = frame.shape

voting: mw.Voting = mw.Voting("hard", "classification").add("rf", pipeline)
automl: mw.AutoML = mw.AutoML.classifier().budget_trials(4).ensemble_kinds(["voting"])
timed_automl: mw.AutoML = mw.AutoML.classifier().budget_minutes(1.0)
probability_pipeline: mw.Pipeline = mw.Pipeline().estimator("lr", mw.LogisticRegression(l2=0.01))
probabilities: mw.Frame = mw.Voting("soft", "classification").add("lr", probability_pipeline).predict_proba(frame)
