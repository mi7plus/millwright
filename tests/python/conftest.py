import pytest

import millwright as mw


@pytest.fixture
def binary_frame():
    return mw.Frame.from_rows(
        [[0.0], [1.0], [9.0], [10.0]],
        ["x"],
    )


@pytest.fixture
def fitted_classifier(binary_frame):
    pipeline = mw.Pipeline()
    pipeline.step("impute", mw.SimpleImputer.mean())
    pipeline.step("scale", mw.StandardScaler())
    pipeline.estimator("rf", mw.RandomForest(n_trees=20, max_depth=3))
    pipeline.fit(binary_frame, [0.0, 0.0, 1.0, 1.0])
    return pipeline
