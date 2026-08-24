from pathlib import Path

import pytest

import millwright as mw


def test_grid_search_refits_best_model(binary_frame):
    pipeline = mw.Pipeline().estimator("rf", mw.RandomForest(n_trees=10, max_depth=2))
    search = mw.GridSearch(
        pipeline,
        {"rf__n_trees": [5, 10]},
        cv=mw.StratifiedKFold(2),
        scoring="accuracy",
    )
    best = search.fit(binary_frame, [0.0, 0.0, 1.0, 1.0])
    assert 0.0 <= best.best_score <= 1.0
    assert best.best_params()["rf__n_trees"] in (5, 10)
    assert len(best.predict(binary_frame)) == len(binary_frame)


def test_invalid_search_configuration_is_rejected(binary_frame):
    pipeline = mw.Pipeline().estimator("rf", mw.RandomForest())
    with pytest.raises(ValueError):
        mw.GridSearch(
            pipeline,
            {"missing__parameter": [1]},
            cv=mw.KFold(2),
            scoring="accuracy",
        ).fit(binary_frame, [0.0, 0.0, 1.0, 1.0])


def test_explain_and_onnx_round_trip(binary_frame, fitted_classifier, tmp_path: Path):
    importance = fitted_classifier.explain(
        binary_frame,
        mw.Explainer.kernel().nsamples(8).background(2),
    )
    assert len(importance) == 1
    assert importance[0][0] == "x"

    path = tmp_path / "model.onnx"
    fitted_classifier.export_onnx(str(path))
    assert path.stat().st_size > 0

    loaded = mw.Pipeline().estimator("onnx", mw.OnnxModel(str(path)))
    loaded.fit(binary_frame, [0.0, 0.0, 1.0, 1.0])
    assert loaded.predict(binary_frame) == fitted_classifier.predict(binary_frame)


def test_missing_onnx_model_is_rejected(binary_frame):
    with pytest.raises(ValueError):
        pipeline = mw.Pipeline().estimator("onnx", mw.OnnxModel("missing-model.onnx"))
        pipeline.fit(binary_frame, [0.0, 0.0, 1.0, 1.0])
