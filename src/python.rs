//! Python bindings (`pip install millwright`) — a pyo3 layer over the stable
//! Rust core.
//!
//! A thin binding, not a fork: everything wraps the same Rust types, so Python
//! runs the Rust engine at Rust speed. The API mirrors the Rust one — a
//! [`Frame`](crate::frame::Frame) of data, composable transformer / estimator
//! objects, and a [`Pipeline`](crate::pipeline::Pipeline) that fits and ships.
//!
//! ```python
//! import millwright as mw
//!
//! train = mw.Frame.from_pandas(df)          # or from_numpy / from_rows
//!
//! pipe = mw.Pipeline()
//! pipe.step("impute", mw.SimpleImputer.median())
//! pipe.step("scale",  mw.StandardScaler())
//! pipe.estimator("rf", mw.RandomForest(n_trees=200, max_depth=8))
//!
//! pipe.fit(train, labels)
//! preds   = pipe.predict(test)
//! metrics = pipe.evaluate(test, labels)     # {"accuracy": ..., "f1": ...}
//!
//! pipe.export_onnx("model.onnx")            # one portable artifact  (feature: onnx)
//! importance = pipe.explain(test)           # SHAP feature ranking   (feature: explain)
//! ```
//!
//! The old convenience builders (`pipe.standard_scaler()`, `pipe.random_forest()`,
//! …) still work; the object API above is the fuller, scikit-learn-shaped one.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::backends::smartcore::{Knn, LinearRegression, NaiveBayes, RandomForest, Svc};
use crate::error::Error;
use crate::evaluate::Report;
use crate::frame::{Dataset, Frame};
use crate::logistic::LogisticRegression;
use crate::pipeline::Pipeline as CorePipeline;
use crate::traits::{Estimator, Model, Predictor, ProbaPredictor};
use crate::transform::{MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler};

use crate::automl::{AutoML, AutoMLResult, Budget, EnsembleKind};
use crate::ensemble::{Bagging, Boosting, EnsembleTask, Stacking, Voting};

#[cfg(feature = "model-selection")]
use crate::selection::{GridSearch, KFold, Metric, ParamGrid, SearchResult, StratifiedKFold};
#[cfg(feature = "model-selection")]
use crate::traits::ParamValue;
#[cfg(feature = "model-selection")]
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

#[cfg(feature = "eda")]
use crate::profile::Profile;
#[cfg(feature = "eda")]
use crate::table::Table;

fn to_py_err(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Column names `f0..f{n-1}` for an unlabelled matrix.
fn default_columns(ncols: usize) -> Vec<String> {
    (0..ncols).map(|i| format!("f{i}")).collect()
}

fn frame_from_rows(rows: Vec<Vec<f64>>) -> PyResult<Frame> {
    let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
    Frame::from_rows(rows, default_columns(ncols)).map_err(to_py_err)
}

/// Coerce a Python argument that is either a [`PyFrame`] or plain rows-of-floats
/// into a Rust [`Frame`].
fn frame_arg(data: &Bound<'_, PyAny>) -> PyResult<Frame> {
    if let Ok(f) = data.extract::<PyFrame>() {
        return Ok(f.inner);
    }
    #[cfg(feature = "eda")]
    if let Ok(t) = data.extract::<PyTable>() {
        return t.inner.to_frame().map_err(to_py_err);
    }
    let rows: Vec<Vec<f64>> = data.extract().map_err(|_| {
        PyValueError::new_err("expected a millwright.Frame, millwright.Table, or list[list[float]]")
    })?;
    frame_from_rows(rows)
}

/// Coerce a Python argument (a `Table` or a `Frame`) into a Rust [`Table`].
#[cfg(feature = "eda")]
fn table_arg(data: &Bound<'_, PyAny>) -> PyResult<Table> {
    if let Ok(t) = data.extract::<PyTable>() {
        return Ok(t.inner);
    }
    if let Ok(f) = data.extract::<PyFrame>() {
        return Table::from_frame(&f.inner).map_err(to_py_err);
    }
    Err(PyValueError::new_err(
        "expected a millwright.Table or millwright.Frame",
    ))
}

#[path = "python_data.rs"]
mod data;
use data::*;

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
struct PyStandardScaler;
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
struct PyMinMaxScaler;
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
struct PySimpleImputer {
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
struct PyOneHotEncoder;
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
struct PyRandomForest {
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
struct PyLogisticRegression {
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
struct PyLinearRegression;
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
struct PyKnn {
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
struct PySvc {
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
struct PyNaiveBayes;
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
struct PyOnnxModel {
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
fn add_transformer(
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
fn set_estimator(
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

// ---------------------------------------------------------------------------
// Explainer (feature: explain).
// ---------------------------------------------------------------------------

/// A SHAP explainer configuration.
#[cfg(feature = "explain")]
#[pyclass(name = "Explainer", from_py_object)]
#[derive(Clone)]
struct PyExplainer {
    nsamples: Option<usize>,
    background: Option<usize>,
}

#[cfg(feature = "explain")]
#[pymethods]
impl PyExplainer {
    /// Kernel SHAP with the library defaults.
    #[staticmethod]
    fn kernel() -> Self {
        Self {
            nsamples: None,
            background: None,
        }
    }
    /// Number of SHAP coalition samples per row.
    fn nsamples(&self, n: usize) -> Self {
        Self {
            nsamples: Some(n),
            background: self.background,
        }
    }
    /// Number of background rows used as the reference.
    fn background(&self, n: usize) -> Self {
        Self {
            nsamples: self.nsamples,
            background: Some(n),
        }
    }
}

#[cfg(feature = "explain")]
impl PyExplainer {
    fn to_inner(&self) -> crate::explain::Explainer {
        let mut e = crate::explain::Explainer::kernel();
        if let Some(n) = self.nsamples {
            e = e.nsamples(n);
        }
        if let Some(b) = self.background {
            e = e.background(b);
        }
        e
    }
}

// ---------------------------------------------------------------------------
// Pipeline.
// ---------------------------------------------------------------------------

/// A preprocessing-plus-model pipeline, driven from Python.
///
/// `unsendable`: the wrapped Rust pipeline holds non-`Sync` trait objects, so
/// the object stays on the thread that created it (fine under the GIL).
#[pyclass(name = "Pipeline", unsendable)]
pub struct PyPipeline {
    inner: CorePipeline,
    fitted: bool,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new() -> Self {
        PyPipeline {
            inner: CorePipeline::new(),
            fitted: false,
        }
    }

    /// Add a named transformer step from a transformer object (chainable).
    fn step<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        transformer: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let pipe = std::mem::take(&mut slf.inner);
        slf.inner = add_transformer(pipe, name, transformer)?;
        Ok(slf)
    }

    /// Set the final estimator from an estimator object (chainable).
    fn estimator<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let pipe = std::mem::take(&mut slf.inner);
        slf.inner = set_estimator(pipe, name, estimator)?;
        Ok(slf)
    }

    // ---- convenience builders (kept for back-compat) --------------------

    /// Add a standard-scaler preprocessing step.
    #[pyo3(signature = (name=None))]
    fn standard_scaler(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(
            name.unwrap_or_else(|| "scale".into()),
            StandardScaler::new(),
        );
    }

    /// Add a min-max scaler preprocessing step.
    #[pyo3(signature = (name=None))]
    fn min_max_scaler(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(name.unwrap_or_else(|| "scale".into()), MinMaxScaler::new());
    }

    /// Add a missing-value imputer. `strategy` is `"median"` (default) or `"mean"`.
    #[pyo3(signature = (name=None, strategy=None))]
    fn simple_imputer(&mut self, name: Option<String>, strategy: Option<String>) -> PyResult<()> {
        let imputer = match strategy.as_deref().unwrap_or("median") {
            "median" => SimpleImputer::median(),
            "mean" => SimpleImputer::mean(),
            other => return Err(PyValueError::new_err(format!("unknown strategy '{other}'"))),
        };
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(name.unwrap_or_else(|| "impute".into()), imputer);
        Ok(())
    }

    /// Add a one-hot encoder that infers low-cardinality integer columns.
    #[pyo3(signature = (name=None))]
    fn one_hot(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(
            name.unwrap_or_else(|| "encode".into()),
            OneHotEncoder::infer(),
        );
    }

    /// Set a random-forest estimator as the final step.
    #[pyo3(signature = (name=None, n_trees=100, max_depth=None))]
    fn random_forest(&mut self, name: Option<String>, n_trees: u16, max_depth: Option<u16>) {
        let mut rf = RandomForest::new().n_trees(n_trees);
        if let Some(d) = max_depth {
            rf = rf.max_depth(d);
        }
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.estimator(name.unwrap_or_else(|| "rf".into()), rf);
    }

    /// Set an ordinary-least-squares regressor as the final step.
    #[pyo3(signature = (name=None))]
    fn linear_regression(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.estimator(name.unwrap_or_else(|| "lr".into()), LinearRegression::new());
    }

    /// Set a k-nearest-neighbours classifier as the final step.
    #[pyo3(signature = (name=None, k=5))]
    fn knn(&mut self, name: Option<String>, k: usize) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.estimator(name.unwrap_or_else(|| "knn".into()), Knn::k(k));
    }

    /// Set a support vector classifier as the final step (RBF when `gamma` set).
    #[pyo3(signature = (name=None, c=1.0, gamma=None))]
    fn svc(&mut self, name: Option<String>, c: f64, gamma: Option<f64>) {
        let mut model = Svc::new().c(c);
        if let Some(g) = gamma {
            model = model.gamma(g);
        }
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.estimator(name.unwrap_or_else(|| "svc".into()), model);
    }

    /// Set a Gaussian naive-Bayes classifier as the final step.
    #[pyo3(signature = (name=None))]
    fn naive_bayes(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.estimator(name.unwrap_or_else(|| "nb".into()), NaiveBayes::new());
    }

    // ---- fit / predict / evaluate ---------------------------------------

    /// Fit the pipeline on a `Frame` (or rows of floats) and a target vector.
    fn fit(&mut self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
        let frame = frame_arg(data)?;
        let dataset = Dataset::new(frame, labels).map_err(to_py_err)?;
        self.inner.fit(&dataset).map_err(to_py_err)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict one value per input row.
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err("pipeline is not fitted"));
        }
        let frame = frame_arg(data)?;
        self.inner.predict(&frame).map_err(to_py_err)
    }

    /// Predict on `data` and score against `labels`, returning a metrics dict
    /// (accuracy/precision/recall/f1 for classification, mae/mse/rmse/r2 for
    /// regression — the task is inferred from the labels).
    fn evaluate(
        &self,
        data: &Bound<'_, PyAny>,
        labels: Vec<f64>,
    ) -> PyResult<HashMap<String, f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err("pipeline is not fitted"));
        }
        let frame = frame_arg(data)?;
        let preds = self.inner.predict(&frame).map_err(to_py_err)?;
        let report = Report::new(&labels, &preds);
        Ok(report.metrics().iter().cloned().collect())
    }

    /// The pipeline's step names, in order.
    fn steps(&self) -> Vec<String> {
        self.inner
            .step_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    // ---- explain / export (feature-gated) -------------------------------

    /// SHAP feature importance for the fitted pipeline: `[(column, mean|shap|)]`,
    /// most important first. Pass an `Explainer` to tune it. (feature: explain)
    #[cfg(feature = "explain")]
    #[pyo3(signature = (data, explainer=None))]
    fn explain(
        &self,
        data: &Bound<'_, PyAny>,
        explainer: Option<PyRef<'_, PyExplainer>>,
    ) -> PyResult<Vec<(String, f64)>> {
        use crate::explain::Explain;
        if !self.fitted {
            return Err(PyValueError::new_err("pipeline is not fitted"));
        }
        let frame = frame_arg(data)?;
        let ex = explainer
            .map(|e| e.to_inner())
            .unwrap_or_else(crate::explain::Explainer::kernel);
        let explanation = self.inner.explain(&ex, &frame).map_err(to_py_err)?;
        Ok(explanation.importance())
    }

    /// Export the whole fitted pipeline to a single ONNX file. Affine steps
    /// (scalers) fold into the estimator's graph; a non-affine step (impute,
    /// one-hot) raises, naming the step. (feature: onnx)
    #[cfg(feature = "onnx")]
    fn export_onnx(&self, path: String) -> PyResult<()> {
        use crate::onnx::ExportOnnx;
        if !self.fitted {
            return Err(PyValueError::new_err("pipeline is not fitted"));
        }
        self.inner.export_onnx(path).map_err(to_py_err)
    }
}

// ---------------------------------------------------------------------------
// Cross-validation & hyperparameter search (feature: model-selection).
// ---------------------------------------------------------------------------

#[cfg(feature = "model-selection")]
#[derive(Clone, Copy)]
enum CvSpec {
    KFold(usize),
    Stratified(usize),
}

/// Plain k-fold cross-validation.
#[cfg(feature = "model-selection")]
#[pyclass(name = "KFold", from_py_object)]
#[derive(Clone)]
struct PyKFold {
    k: usize,
}
#[cfg(feature = "model-selection")]
#[pymethods]
impl PyKFold {
    #[new]
    fn new(k: usize) -> Self {
        Self { k }
    }
}

/// Stratified k-fold — preserves class balance across folds.
#[cfg(feature = "model-selection")]
#[pyclass(name = "StratifiedKFold", from_py_object)]
#[derive(Clone)]
struct PyStratifiedKFold {
    k: usize,
}
#[cfg(feature = "model-selection")]
#[pymethods]
impl PyStratifiedKFold {
    #[new]
    fn new(k: usize) -> Self {
        Self { k }
    }
}

#[cfg(feature = "model-selection")]
fn parse_cv(obj: Option<&Bound<'_, PyAny>>) -> PyResult<CvSpec> {
    let Some(obj) = obj else {
        return Ok(CvSpec::KFold(5));
    };
    if let Ok(s) = obj.extract::<PyStratifiedKFold>() {
        return Ok(CvSpec::Stratified(s.k));
    }
    if let Ok(k) = obj.extract::<PyKFold>() {
        return Ok(CvSpec::KFold(k.k));
    }
    if let Ok(k) = obj.extract::<usize>() {
        return Ok(CvSpec::KFold(k));
    }
    Err(PyValueError::new_err(
        "cv expects a KFold, StratifiedKFold, or an int number of folds",
    ))
}

#[cfg(feature = "model-selection")]
fn parse_metric(s: &str) -> PyResult<Metric> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "accuracy" => Metric::Accuracy,
        "f1" => Metric::F1,
        "mae" => Metric::Mae,
        "mse" => Metric::Mse,
        "rmse" => Metric::Rmse,
        "r2" => Metric::R2,
        other => return Err(PyValueError::new_err(format!("unknown metric '{other}'"))),
    })
}

fn parse_ensemble_task(task: &str) -> PyResult<EnsembleTask> {
    match task.to_ascii_lowercase().as_str() {
        "infer" => Ok(EnsembleTask::Infer),
        "classification" | "classifier" => Ok(EnsembleTask::Classification),
        "regression" | "regressor" => Ok(EnsembleTask::Regression),
        other => Err(PyValueError::new_err(format!(
            "unknown ensemble task '{other}'; expected infer, classification, or regression"
        ))),
    }
}

fn fit_ensemble(model: &mut dyn Model, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
    let dataset = Dataset::new(frame_arg(data)?, labels).map_err(to_py_err)?;
    model.fit(&dataset).map_err(to_py_err)
}

fn predict_ensemble(model: &dyn Model, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    model.predict(&frame_arg(data)?).map_err(to_py_err)
}

fn export_ensemble(model: &dyn Model, path: String) -> PyResult<()> {
    let proto = model.to_onnx_proto().map_err(to_py_err)?;
    onnx_export_rs::graph_builder::save_to_file(&proto, path)
        .map_err(|error| PyValueError::new_err(format!("ONNX save failed: {error}")))
}

/// A hard- or soft-voting ensemble over Python `Pipeline` objects.
#[pyclass(name = "Voting", unsendable)]
struct PyVoting {
    inner: Voting,
}

#[pymethods]
impl PyVoting {
    #[new]
    #[pyo3(signature = (kind="hard", task="infer"))]
    fn new(kind: &str, task: &str) -> PyResult<Self> {
        let inner = match kind.to_ascii_lowercase().as_str() {
            "hard" => Voting::hard(),
            "soft" => Voting::soft(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown voting kind '{other}'"
                )))
            }
        };
        Ok(Self {
            inner: inner.task(parse_ensemble_task(task)?),
        })
    }

    fn add<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        pipeline: PyRef<'_, PyPipeline>,
    ) -> PyRefMut<'a, Self> {
        slf.inner = slf.inner.clone().add(name, pipeline.inner.clone());
        slf
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
        fit_ensemble(&mut self.inner, data, labels)
    }
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        predict_ensemble(&self.inner, data)
    }
    fn predict_proba(&self, data: &Bound<'_, PyAny>) -> PyResult<PyFrame> {
        Ok(PyFrame {
            inner: self
                .inner
                .predict_proba(&frame_arg(data)?)
                .map_err(to_py_err)?,
        })
    }
    fn export_onnx(&self, path: String) -> PyResult<()> {
        export_ensemble(&self.inner, path)
    }
}

/// Bootstrap aggregation over a Python `Pipeline` base estimator.
#[pyclass(name = "Bagging", unsendable)]
struct PyBagging {
    inner: Bagging,
}

#[pymethods]
impl PyBagging {
    #[new]
    #[pyo3(signature = (base, n_estimators=10, seed=0, task="infer"))]
    fn new(
        base: PyRef<'_, PyPipeline>,
        n_estimators: usize,
        seed: u64,
        task: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Bagging::of(base.inner.clone())
                .n_estimators(n_estimators)
                .seed(seed)
                .task(parse_ensemble_task(task)?),
        })
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
        fit_ensemble(&mut self.inner, data, labels)
    }
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        predict_ensemble(&self.inner, data)
    }
    fn export_onnx(&self, path: String) -> PyResult<()> {
        export_ensemble(&self.inner, path)
    }
}

/// SAMME boosting over a Python `Pipeline` classifier.
#[pyclass(name = "Boosting", unsendable)]
struct PyBoosting {
    inner: Boosting,
}

#[pymethods]
impl PyBoosting {
    #[new]
    #[pyo3(signature = (base, n_estimators=50, learning_rate=1.0, seed=0))]
    fn new(
        base: PyRef<'_, PyPipeline>,
        n_estimators: usize,
        learning_rate: f64,
        seed: u64,
    ) -> Self {
        Self {
            inner: Boosting::of(base.inner.clone())
                .n_estimators(n_estimators)
                .learning_rate(learning_rate)
                .seed(seed),
        }
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
        fit_ensemble(&mut self.inner, data, labels)
    }
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        predict_ensemble(&self.inner, data)
    }
    fn export_onnx(&self, path: String) -> PyResult<()> {
        export_ensemble(&self.inner, path)
    }
}

/// Leak-free stacking over Python `Pipeline` objects.
#[pyclass(name = "Stacking", unsendable)]
struct PyStacking {
    inner: Stacking,
}

#[pymethods]
impl PyStacking {
    #[new]
    #[pyo3(signature = (meta, cv=None))]
    fn new(meta: PyRef<'_, PyPipeline>, cv: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let model = Stacking::meta(meta.inner.clone());
        let model = match parse_cv(cv)? {
            CvSpec::KFold(k) => model.cv(KFold::new(k)),
            CvSpec::Stratified(k) => model.cv(StratifiedKFold::new(k)),
        };
        Ok(Self { inner: model })
    }

    fn base<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        pipeline: PyRef<'_, PyPipeline>,
    ) -> PyRefMut<'a, Self> {
        slf.inner = slf.inner.clone().base(name, pipeline.inner.clone());
        slf
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<()> {
        fit_ensemble(&mut self.inner, data, labels)
    }
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        predict_ensemble(&self.inner, data)
    }
    fn export_onnx(&self, path: String) -> PyResult<()> {
        export_ensemble(&self.inner, path)
    }
}

fn parse_ensemble_kind(kind: &str) -> PyResult<EnsembleKind> {
    match kind.to_ascii_lowercase().as_str() {
        "voting" => Ok(EnsembleKind::Voting),
        "bagging" => Ok(EnsembleKind::Bagging),
        "boosting" => Ok(EnsembleKind::Boosting),
        "stacking" => Ok(EnsembleKind::Stacking),
        other => Err(PyValueError::new_err(format!(
            "unknown ensemble kind '{other}'"
        ))),
    }
}

/// Automated preprocessing, model, hyperparameter, and ensemble search.
#[pyclass(name = "AutoML", unsendable)]
struct PyAutoML {
    classifier: bool,
    trials: usize,
    minutes: Option<f64>,
    metric: Metric,
    cv: CvSpec,
    seed: u64,
    ensemble: bool,
    ensemble_size: usize,
    ensemble_kinds: Vec<EnsembleKind>,
    prefer_ensemble: bool,
    parallel: bool,
}

#[pymethods]
impl PyAutoML {
    #[staticmethod]
    fn classifier() -> Self {
        Self {
            classifier: true,
            trials: 40,
            minutes: None,
            metric: Metric::Accuracy,
            cv: CvSpec::Stratified(5),
            seed: 0,
            ensemble: true,
            ensemble_size: 3,
            ensemble_kinds: vec![
                EnsembleKind::Voting,
                EnsembleKind::Bagging,
                EnsembleKind::Boosting,
                EnsembleKind::Stacking,
            ],
            prefer_ensemble: false,
            parallel: false,
        }
    }

    #[staticmethod]
    fn regressor() -> Self {
        Self {
            classifier: false,
            trials: 40,
            minutes: None,
            metric: Metric::R2,
            cv: CvSpec::KFold(5),
            seed: 0,
            ensemble: true,
            ensemble_size: 3,
            ensemble_kinds: vec![
                EnsembleKind::Voting,
                EnsembleKind::Bagging,
                EnsembleKind::Stacking,
            ],
            prefer_ensemble: false,
            parallel: false,
        }
    }

    fn budget_trials<'a>(mut slf: PyRefMut<'a, Self>, trials: usize) -> PyRefMut<'a, Self> {
        slf.trials = trials;
        slf.minutes = None;
        slf
    }

    fn budget_minutes<'a>(mut slf: PyRefMut<'a, Self>, minutes: f64) -> PyRefMut<'a, Self> {
        slf.minutes = Some(minutes);
        slf
    }

    fn scoring<'a>(mut slf: PyRefMut<'a, Self>, metric: &str) -> PyResult<PyRefMut<'a, Self>> {
        slf.metric = parse_metric(metric)?;
        Ok(slf)
    }

    fn cv<'a>(mut slf: PyRefMut<'a, Self>, cv: &Bound<'_, PyAny>) -> PyResult<PyRefMut<'a, Self>> {
        slf.cv = parse_cv(Some(cv))?;
        Ok(slf)
    }

    fn seed<'a>(mut slf: PyRefMut<'a, Self>, seed: u64) -> PyRefMut<'a, Self> {
        slf.seed = seed;
        slf
    }

    fn no_ensemble<'a>(mut slf: PyRefMut<'a, Self>) -> PyRefMut<'a, Self> {
        slf.ensemble = false;
        slf
    }

    fn ensemble_size<'a>(mut slf: PyRefMut<'a, Self>, size: usize) -> PyRefMut<'a, Self> {
        slf.ensemble_size = size.max(2);
        slf
    }

    fn ensemble_kinds<'a>(
        mut slf: PyRefMut<'a, Self>,
        kinds: Vec<String>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.ensemble_kinds = kinds
            .iter()
            .map(|kind| parse_ensemble_kind(kind))
            .collect::<PyResult<_>>()?;
        Ok(slf)
    }

    fn prefer_ensemble_on_tie<'a>(mut slf: PyRefMut<'a, Self>) -> PyRefMut<'a, Self> {
        slf.prefer_ensemble = true;
        slf
    }

    fn parallel<'a>(mut slf: PyRefMut<'a, Self>) -> PyRefMut<'a, Self> {
        slf.parallel = true;
        slf
    }

    fn fit(&self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<PyAutoMLResult> {
        let dataset = Dataset::new(frame_arg(data)?, labels).map_err(to_py_err)?;
        let search = if self.classifier {
            AutoML::classifier()
        } else {
            AutoML::regressor()
        };
        let budget = self
            .minutes
            .map(Budget::minutes)
            .unwrap_or_else(|| Budget::trials(self.trials));
        let search = search
            .budget(budget)
            .metric(self.metric)
            .seed(self.seed)
            .ensemble_size(self.ensemble_size)
            .ensemble_kinds(self.ensemble_kinds.clone());
        let search = match self.cv {
            CvSpec::KFold(k) => search.cv(KFold::new(k)),
            CvSpec::Stratified(k) => search.cv(StratifiedKFold::new(k)),
        };
        let search = if self.ensemble {
            search
        } else {
            search.no_ensemble()
        };
        let search = if self.prefer_ensemble {
            search.prefer_ensemble_on_tie()
        } else {
            search
        };
        let search = if self.parallel {
            search.parallel()
        } else {
            search
        };
        Ok(PyAutoMLResult {
            inner: search.fit(&dataset).map_err(to_py_err)?,
        })
    }
}

/// A fitted AutoML winner and its complete leaderboard.
#[pyclass(name = "AutoMLResult", unsendable)]
struct PyAutoMLResult {
    inner: AutoMLResult,
}

#[pymethods]
impl PyAutoMLResult {
    #[getter]
    fn best_label(&self) -> &str {
        self.inner.best_label()
    }

    #[getter]
    fn best_score(&self) -> f64 {
        self.inner.best_score()
    }

    #[getter]
    fn is_ensemble(&self) -> bool {
        self.inner.is_ensemble()
    }

    fn leaderboard(&self) -> String {
        self.inner.leaderboard()
    }

    fn leaderboard_entries(&self) -> Vec<(String, f64)> {
        self.inner.leaderboard_entries().to_vec()
    }

    fn candidate_failures(&self) -> Vec<(String, String)> {
        self.inner.candidate_failures().to_vec()
    }

    fn ensemble_failures(&self) -> Vec<(String, String)> {
        self.inner.ensemble_failures().to_vec()
    }

    fn best_pipeline(&self) -> Option<PyPipeline> {
        self.inner.best_pipeline().map(|pipeline| PyPipeline {
            inner: pipeline.clone(),
            fitted: true,
        })
    }

    fn best_model(&self) -> PyFittedModel {
        PyFittedModel {
            inner: self.inner.clone_best_model(),
        }
    }

    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        self.inner.predict(&frame_arg(data)?).map_err(to_py_err)
    }

    fn export_onnx(&self, path: String) -> PyResult<()> {
        self.inner.export_onnx(path).map_err(to_py_err)
    }
}

/// A type-erased fitted AutoML winner, whether pipeline or ensemble.
#[pyclass(name = "FittedModel", unsendable)]
struct PyFittedModel {
    inner: Box<dyn Model>,
}

#[pymethods]
impl PyFittedModel {
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        predict_ensemble(self.inner.as_ref(), data)
    }

    fn export_onnx(&self, path: String) -> PyResult<()> {
        export_ensemble(self.inner.as_ref(), path)
    }
}

#[cfg(feature = "model-selection")]
fn param_value(obj: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    // Order matters: a Python `bool` also extracts as `int`.
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(ParamValue::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ParamValue::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(ParamValue::Float(f));
    }
    Err(PyValueError::new_err(
        "grid values must be int, float, or bool",
    ))
}

#[cfg(feature = "model-selection")]
fn param_grid(dict: &Bound<'_, PyDict>) -> PyResult<ParamGrid> {
    let mut grid = ParamGrid::new();
    for (key, val) in dict.iter() {
        let path: String = key.extract()?;
        let values = val
            .try_iter()?
            .map(|v| param_value(&v?))
            .collect::<PyResult<Vec<_>>>()?;
        grid.add(path, values);
    }
    Ok(grid)
}

/// Exhaustive grid search over a `Pipeline`, using the `"step__param"` naming
/// convention for the grid keys.
#[cfg(feature = "model-selection")]
#[pyclass(name = "GridSearch", unsendable)]
struct PyGridSearch {
    pipeline: CorePipeline,
    grid: ParamGrid,
    cv: CvSpec,
    metric: Metric,
}

#[cfg(feature = "model-selection")]
#[pymethods]
impl PyGridSearch {
    #[new]
    #[pyo3(signature = (pipeline, grid, cv=None, scoring=None))]
    fn new(
        pipeline: &PyPipeline,
        grid: &Bound<'_, PyDict>,
        cv: Option<&Bound<'_, PyAny>>,
        scoring: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            pipeline: pipeline.inner.clone(),
            grid: param_grid(grid)?,
            cv: parse_cv(cv)?,
            metric: scoring
                .as_deref()
                .map(parse_metric)
                .transpose()?
                .unwrap_or(Metric::Accuracy),
        })
    }

    /// Set the cross-validation strategy (chainable).
    fn cv<'a>(mut slf: PyRefMut<'a, Self>, cv: &Bound<'_, PyAny>) -> PyResult<PyRefMut<'a, Self>> {
        slf.cv = parse_cv(Some(cv))?;
        Ok(slf)
    }

    /// Set the scoring metric (chainable): `"accuracy"`, `"f1"`, `"mae"`,
    /// `"mse"`, `"rmse"`, or `"r2"`.
    fn scoring<'a>(mut slf: PyRefMut<'a, Self>, metric: &str) -> PyResult<PyRefMut<'a, Self>> {
        slf.metric = parse_metric(metric)?;
        Ok(slf)
    }

    /// Run the search and refit the winner on the full data.
    fn fit(&self, data: &Bound<'_, PyAny>, labels: Vec<f64>) -> PyResult<PySearchResult> {
        let frame = frame_arg(data)?;
        let dataset = Dataset::new(frame, labels).map_err(to_py_err)?;
        let search = GridSearch::new(self.pipeline.clone(), self.grid.clone()).scoring(self.metric);
        let result = match self.cv {
            CvSpec::KFold(k) => search.cv(KFold::new(k)).fit(&dataset),
            CvSpec::Stratified(k) => search.cv(StratifiedKFold::new(k)).fit(&dataset),
        }
        .map_err(to_py_err)?;
        Ok(PySearchResult { inner: result })
    }
}

/// The outcome of a search: the refit best model, its score and parameters.
#[cfg(feature = "model-selection")]
#[pyclass(name = "SearchResult", unsendable)]
struct PySearchResult {
    inner: SearchResult,
}

#[cfg(feature = "model-selection")]
#[pymethods]
impl PySearchResult {
    /// The best cross-validated score.
    #[getter]
    fn best_score(&self) -> f64 {
        self.inner.best_score()
    }

    /// The winning parameter assignment, as a dict.
    fn best_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in self.inner.best_params() {
            match v {
                ParamValue::Int(i) => d.set_item(k, *i)?,
                ParamValue::Float(f) => d.set_item(k, *f)?,
                ParamValue::Bool(b) => d.set_item(k, *b)?,
            }
        }
        Ok(d)
    }

    /// Predict with the refit best model.
    fn predict(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let frame = frame_arg(data)?;
        self.inner.predict(&frame).map_err(to_py_err)
    }
}

/// The installed Millwright version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `millwright` Python module.
#[pymodule]
fn millwright(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrame>()?;
    m.add_class::<PyPipeline>()?;
    #[cfg(feature = "eda")]
    {
        m.add_class::<PyTable>()?;
        m.add_class::<PyProfile>()?;
    }
    m.add_class::<PyStandardScaler>()?;
    m.add_class::<PyMinMaxScaler>()?;
    m.add_class::<PySimpleImputer>()?;
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyRandomForest>()?;
    m.add_class::<PyLogisticRegression>()?;
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyKnn>()?;
    m.add_class::<PySvc>()?;
    m.add_class::<PyNaiveBayes>()?;
    m.add_class::<PyVoting>()?;
    m.add_class::<PyBagging>()?;
    m.add_class::<PyBoosting>()?;
    m.add_class::<PyStacking>()?;
    m.add_class::<PyAutoML>()?;
    m.add_class::<PyAutoMLResult>()?;
    m.add_class::<PyFittedModel>()?;
    #[cfg(feature = "onnx")]
    m.add_class::<PyOnnxModel>()?;
    #[cfg(feature = "explain")]
    m.add_class::<PyExplainer>()?;
    #[cfg(feature = "model-selection")]
    {
        m.add_class::<PyKFold>()?;
        m.add_class::<PyStratifiedKFold>()?;
        m.add_class::<PyGridSearch>()?;
        m.add_class::<PySearchResult>()?;
    }
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
