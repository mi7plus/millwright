from pathlib import Path

import millwright as mw


def classification_data():
    rows = []
    labels = []
    for i in range(12):
        rows.append([i * 0.1, i * 0.1])
        labels.append(0.0)
        rows.append([9.0 + i * 0.1, 9.0 + i * 0.1])
        labels.append(1.0)
    return mw.Frame.from_rows(rows, ["a", "b"]), labels


def classifier_pipeline(n_trees: int) -> mw.Pipeline:
    return mw.Pipeline().estimator(
        "rf", mw.RandomForest(n_trees=n_trees, max_depth=3)
    )


def test_voting_and_automl_are_first_class_python_apis(tmp_path: Path):
    frame, labels = classification_data()
    voting = mw.Voting("hard", "classification")
    voting.add("rf1", classifier_pipeline(8))
    voting.add("rf2", classifier_pipeline(12))
    voting.fit(frame, labels)
    assert voting.predict(frame) == labels
    voting.export_onnx(str(tmp_path / "voting.onnx"))

    bagging = mw.Bagging(
        classifier_pipeline(8), n_estimators=3, seed=4, task="classification"
    )
    bagging.fit(frame, labels)
    assert bagging.predict(frame) == labels
    bagging.export_onnx(str(tmp_path / "bagging.onnx"))

    stump = mw.Pipeline().estimator(
        "rf", mw.RandomForest(n_trees=1, max_depth=1)
    )
    boosting = mw.Boosting(stump, n_estimators=4, learning_rate=0.8, seed=4)
    boosting.fit(frame, labels)
    assert boosting.predict(frame) == labels
    boosting.export_onnx(str(tmp_path / "boosting.onnx"))

    stacking = mw.Stacking(classifier_pipeline(8), mw.StratifiedKFold(3))
    stacking.base("rf1", classifier_pipeline(8))
    stacking.base("rf2", classifier_pipeline(12))
    stacking.fit(frame, labels)
    assert stacking.predict(frame) == labels
    stacking.export_onnx(str(tmp_path / "stacking.onnx"))

    result = (
        mw.AutoML.classifier()
        .budget_trials(6)
        .scoring("accuracy")
        .cv(mw.StratifiedKFold(3))
        .seed(3)
        .ensemble_size(2)
        .ensemble_kinds(["voting", "bagging", "boosting", "stacking"])
        .prefer_ensemble_on_tie()
        .parallel()
        .fit(frame, labels)
    )
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.predict(frame)) == len(labels)
    assert "rank" in result.leaderboard()
    assert isinstance(result.ensemble_failures(), list)

    artifact = tmp_path / "automl.onnx"
    result.export_onnx(str(artifact))
    assert artifact.stat().st_size > 0


def test_bagging_handles_integer_regression_explicitly():
    frame = mw.Frame.from_rows([[float(i)] for i in range(6)], ["x"])
    labels = [float(i * 2) for i in range(6)]
    base = mw.Pipeline().estimator("lr", mw.LinearRegression())
    model = mw.Bagging(base, n_estimators=5, seed=7, task="regression")
    model.fit(frame, labels)
    assert model.predict(frame)[3] > 4.0

    result = (
        mw.AutoML.regressor()
        .budget_trials(4)
        .cv(mw.KFold(2))
        .no_ensemble()
        .fit(frame, labels)
    )
    assert not result.is_ensemble
    assert result.best_score > 0.9
