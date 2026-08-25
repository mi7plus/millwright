from pathlib import Path

import millwright as mw
import pytest


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


def probability_pipeline(l2: float) -> mw.Pipeline:
    return mw.Pipeline().estimator("logistic", mw.LogisticRegression(l2=l2))


def test_pipeline_probabilities_and_logistic_validation():
    frame, labels = classification_data()
    pipeline = probability_pipeline(0.01)
    pipeline.fit(frame, labels)
    probabilities = pipeline.predict_proba(frame)
    assert probabilities.shape == (len(labels), 2)

    with pytest.raises(ValueError, match="learning_rate"):
        bad = mw.Pipeline().estimator(
            "logistic", mw.LogisticRegression(learning_rate=0.0)
        )
        bad.fit(frame, labels)

    with pytest.raises(ValueError, match="epochs"):
        bad = mw.Pipeline().estimator("logistic", mw.LogisticRegression(epochs=0))
        bad.fit(frame, labels)

    with pytest.raises(ValueError, match="l2"):
        bad = mw.Pipeline().estimator("logistic", mw.LogisticRegression(l2=-0.1))
        bad.fit(frame, labels)

def test_voting_and_automl_are_first_class_python_apis(tmp_path: Path):
    frame, labels = classification_data()
    voting = mw.Voting("hard", "classification")
    voting.add("rf1", classifier_pipeline(8))
    voting.add("rf2", classifier_pipeline(12))
    voting.fit(frame, labels)
    assert voting.predict(frame) == labels
    voting.export_onnx(str(tmp_path / "voting.onnx"))

    soft_voting = mw.Voting("soft", "classification")
    soft_voting.add("lr1", probability_pipeline(0.0))
    soft_voting.add("lr2", probability_pipeline(0.01))
    soft_voting.fit(frame, labels)
    assert soft_voting.predict_proba(frame).shape == (len(labels), 2)

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
        .deployability("onnx")
        .fit(frame, labels)
    )
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.predict(frame)) == len(labels)
    assert "rank" in result.leaderboard()
    assert result.leaderboard_entries()
    assert isinstance(result.candidate_failures(), list)
    assert isinstance(result.ensemble_failures(), list)
    assert isinstance(result.refit_failures(), list)
    fitted = result.best_model()
    assert len(fitted.predict(frame)) == len(labels)
    if "logistic" in result.best_label or "voting-soft" in result.best_label:
        assert fitted.predict_proba(frame).shape == (len(labels), 2)
    if not result.is_ensemble:
        assert result.best_pipeline() is not None

    artifact = tmp_path / "automl.onnx"
    result.export_onnx(str(artifact))
    assert artifact.stat().st_size > 0

    with pytest.raises(ValueError, match="minute budget"):
        mw.AutoML.classifier().budget_minutes(0.0).fit(frame, labels)

    with pytest.raises(ValueError, match="deployability"):
        mw.AutoML.classifier().deployability("portable-ish")


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
    assert result.best_pipeline() is not None
    assert len(result.best_model().predict(frame)) == len(labels)
