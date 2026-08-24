import math

import pytest

import millwright as mw


def test_unfitted_operations_are_rejected(binary_frame, tmp_path):
    pipeline = mw.Pipeline()
    with pytest.raises(ValueError, match="not fitted"):
        pipeline.predict(binary_frame)
    with pytest.raises(ValueError, match="not fitted"):
        pipeline.evaluate(binary_frame, [0.0] * len(binary_frame))
    with pytest.raises(ValueError, match="not fitted"):
        pipeline.export_onnx(str(tmp_path / "model.onnx"))


def test_fit_predict_and_classification_metrics(binary_frame, fitted_classifier):
    predictions = fitted_classifier.predict(binary_frame)
    assert predictions == [0.0, 0.0, 1.0, 1.0]
    metrics = fitted_classifier.evaluate(binary_frame, predictions)
    assert math.isclose(metrics["accuracy"], 1.0)
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()


def test_target_length_mismatch_is_rejected(binary_frame):
    with pytest.raises(ValueError):
        mw.Pipeline().estimator("rf", mw.RandomForest()).fit(binary_frame, [0.0])


def test_unknown_components_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        mw.Pipeline().step("bad", mw.SimpleImputer("unknown"))
    with pytest.raises((TypeError, ValueError)):
        mw.Pipeline().step("bad", object())
    with pytest.raises((TypeError, ValueError)):
        mw.Pipeline().estimator("bad", object())


def test_regression_metrics():
    frame = mw.Frame.from_rows([[0.0], [1.0], [2.0], [3.0]], ["x"])
    pipeline = mw.Pipeline().estimator("lr", mw.LinearRegression())
    pipeline.fit(frame, [1.0, 3.0, 5.0, 7.0])
    metrics = pipeline.evaluate(frame, [1.0, 3.0, 5.0, 7.0])
    assert {"mae", "mse", "rmse", "r2"} <= metrics.keys()
    assert metrics["r2"] > 0.99
