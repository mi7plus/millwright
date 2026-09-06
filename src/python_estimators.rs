use super::*;

// ---------------------------------------------------------------------------
// Transformer & estimator objects.
//
// Each is a lightweight descriptor the pipeline lowers to the concrete Rust
// type when it is added as a step. `extract` clones the descriptor out of the
// Python object, so every class derives `Clone`.
// ---------------------------------------------------------------------------

/// Standardize each column to zero mean and unit variance.
#[pyclass(name = "StandardScaler", from_py_object)]
#[derive(Clone)]
pub(super) struct PyStandardScaler;
#[pymethods]
impl PyStandardScaler {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Scale each column into `[0, 1]`.
#[pyclass(name = "MinMaxScaler", from_py_object)]
#[derive(Clone)]
pub(super) struct PyMinMaxScaler;
#[pymethods]
impl PyMinMaxScaler {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Fill missing values with a per-column statistic (`"median"` or `"mean"`).
#[pyclass(name = "SimpleImputer", from_py_object)]
#[derive(Clone)]
pub(super) struct PySimpleImputer {
    strategy: String,
}
#[pymethods]
impl PySimpleImputer {
    #[new]
    #[pyo3(signature = (strategy=None))]
    fn new(strategy: Option<String>) -> Self {
        Self {
            strategy: strategy.unwrap_or_else(|| "median".into()),
        }
    }
    #[staticmethod]
    fn median() -> Self {
        Self {
            strategy: "median".into(),
        }
    }
    #[staticmethod]
    fn mean() -> Self {
        Self {
            strategy: "mean".into(),
        }
    }
}

/// One-hot encode inferred low-cardinality integer columns.
#[pyclass(name = "OneHotEncoder", from_py_object)]
#[derive(Clone)]
pub(super) struct PyOneHotEncoder;
#[pymethods]
impl PyOneHotEncoder {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// A random-forest estimator.
#[pyclass(name = "RandomForest", from_py_object)]
#[derive(Clone)]
pub(super) struct PyRandomForest {
    n_trees: u16,
    max_depth: Option<u16>,
}
#[pymethods]
impl PyRandomForest {
    #[new]
    #[pyo3(signature = (n_trees=100, max_depth=None))]
    fn new(n_trees: u16, max_depth: Option<u16>) -> Self {
        Self { n_trees, max_depth }
    }
}

/// Binary logistic regression with probability prediction.
#[pyclass(name = "LogisticRegression", from_py_object)]
#[derive(Clone)]
pub(super) struct PyLogisticRegression {
    learning_rate: f64,
    epochs: usize,
    l2: f64,
}
#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (learning_rate=0.5, epochs=500, l2=0.0))]
    fn new(learning_rate: f64, epochs: usize, l2: f64) -> Self {
        Self {
            learning_rate,
            epochs,
            l2,
        }
    }
}

/// An ordinary-least-squares regressor.
#[pyclass(name = "LinearRegression", from_py_object)]
#[derive(Clone)]
pub(super) struct PyLinearRegression;
#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// A k-nearest-neighbours classifier.
#[pyclass(name = "Knn", from_py_object)]
#[derive(Clone)]
pub(super) struct PyKnn {
    k: usize,
}
#[pymethods]
impl PyKnn {
    #[new]
    #[pyo3(signature = (k=5))]
    fn new(k: usize) -> Self {
        Self { k }
    }
}

/// A support vector classifier (linear by default; pass `gamma` for an RBF
/// kernel, or use `Svc.rbf()`).
#[pyclass(name = "Svc", from_py_object)]
#[derive(Clone)]
pub(super) struct PySvc {
    c: f64,
    gamma: Option<f64>,
}
#[pymethods]
impl PySvc {
    #[new]
    #[pyo3(signature = (c=1.0, gamma=None))]
    fn new(c: f64, gamma: Option<f64>) -> Self {
        Self { c, gamma }
    }
    /// An RBF-kernel SVC.
    #[staticmethod]
    #[pyo3(signature = (gamma=0.5, c=1.0))]
    fn rbf(gamma: f64, c: f64) -> Self {
        Self {
            c,
            gamma: Some(gamma),
        }
    }
}

/// A Gaussian naive-Bayes classifier.
#[pyclass(name = "NaiveBayes", from_py_object)]
#[derive(Clone)]
pub(super) struct PyNaiveBayes;
#[pymethods]
impl PyNaiveBayes {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// A pre-trained ONNX model (e.g. exported from scikit-learn or PyTorch), used
/// as a pipeline's frozen estimator behind Millwright's preprocessing steps.
#[cfg(feature = "onnx")]
#[pyclass(name = "OnnxModel", from_py_object)]
#[derive(Clone)]
pub(super) struct PyOnnxModel {
    path: String,
}
#[cfg(feature = "onnx")]
#[pymethods]
impl PyOnnxModel {
    #[new]
    fn new(path: String) -> Self {
        Self { path }
    }
}

/// Lower a Python transformer object onto the pipeline as a named step.
pub(super) fn add_transformer(
    pipe: CorePipeline,
    name: String,
    obj: &Bound<'_, PyAny>,
) -> PyResult<CorePipeline> {
    if obj.extract::<PyStandardScaler>().is_ok() {
        return Ok(pipe.step(name, StandardScaler::new()));
    }
    if obj.extract::<PyMinMaxScaler>().is_ok() {
        return Ok(pipe.step(name, MinMaxScaler::new()));
    }
    if let Ok(s) = obj.extract::<PySimpleImputer>() {
        let imputer = match s.strategy.as_str() {
            "median" => SimpleImputer::median(),
            "mean" => SimpleImputer::mean(),
            other => return Err(PyValueError::new_err(format!("unknown strategy '{other}'"))),
        };
        return Ok(pipe.step(name, imputer));
    }
    if obj.extract::<PyOneHotEncoder>().is_ok() {
        return Ok(pipe.step(name, OneHotEncoder::infer()));
    }
    Err(PyValueError::new_err(
        "step expects a transformer object \
         (StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder)",
    ))
}

/// Lower a Python estimator object onto the pipeline as the final step.
pub(super) fn set_estimator(
    pipe: CorePipeline,
    name: String,
    obj: &Bound<'_, PyAny>,
) -> PyResult<CorePipeline> {
    if let Ok(rf) = obj.extract::<PyRandomForest>() {
        let mut model = RandomForest::new().n_trees(rf.n_trees);
        if let Some(d) = rf.max_depth {
            model = model.max_depth(d);
        }
        return Ok(pipe.estimator(name, model));
    }
    if let Ok(model) = obj.extract::<PyLogisticRegression>() {
        return Ok(pipe.estimator(
            name,
            LogisticRegression::new()
                .learning_rate(model.learning_rate)
                .epochs(model.epochs)
                .l2(model.l2),
        ));
    }
    if obj.extract::<PyLinearRegression>().is_ok() {
        return Ok(pipe.estimator(name, LinearRegression::new()));
    }
    if let Ok(m) = obj.extract::<PyKnn>() {
        return Ok(pipe.estimator(name, Knn::k(m.k)));
    }
    if let Ok(m) = obj.extract::<PySvc>() {
        let mut model = Svc::new().c(m.c);
        if let Some(g) = m.gamma {
            model = model.gamma(g);
        }
        return Ok(pipe.estimator(name, model));
    }
    if obj.extract::<PyNaiveBayes>().is_ok() {
        return Ok(pipe.estimator(name, NaiveBayes::new()));
    }
    #[cfg(feature = "onnx")]
    if let Ok(m) = obj.extract::<PyOnnxModel>() {
        let model = crate::onnx::InferenceModel::load(&m.path).map_err(to_py_err)?;
        return Ok(pipe.estimator(name, model));
    }
    Err(PyValueError::new_err(
        "estimator expects an estimator object \
         (RandomForest, LogisticRegression, LinearRegression, Knn, Svc, NaiveBayes, OnnxModel)",
    ))
}
