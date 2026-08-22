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

use crate::backends::smartcore::{LinearRegression, RandomForest};
use crate::error::Error;
use crate::evaluate::Report;
use crate::frame::{Dataset, Frame};
use crate::pipeline::Pipeline as CorePipeline;
use crate::traits::{Estimator, Predictor};
use crate::transform::{MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler};

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

// ---------------------------------------------------------------------------
// Frame — the data matrix, with DataFrame / array ingest.
// ---------------------------------------------------------------------------

/// A dense matrix of features with named columns — the data every step speaks.
#[pyclass(name = "Frame")]
#[derive(Clone)]
pub struct PyFrame {
    inner: Frame,
}

#[pymethods]
impl PyFrame {
    /// Build from rows of floats, optionally naming the columns.
    #[staticmethod]
    #[pyo3(signature = (rows, columns=None))]
    fn from_rows(rows: Vec<Vec<f64>>, columns: Option<Vec<String>>) -> PyResult<Self> {
        let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
        let cols = columns.unwrap_or_else(|| default_columns(ncols));
        Ok(Self {
            inner: Frame::from_rows(rows, cols).map_err(to_py_err)?,
        })
    }

    /// Build from a numpy array (anything with a 2-D `.tolist()`).
    #[staticmethod]
    fn from_numpy(array: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rows: Vec<Vec<f64>> = array.call_method0("tolist")?.extract()?;
        Self::from_rows(rows, None)
    }

    /// Build from a pandas DataFrame — takes its column names and numeric values.
    #[staticmethod]
    fn from_pandas(df: &Bound<'_, PyAny>) -> PyResult<Self> {
        let columns: Vec<String> = df.getattr("columns")?.call_method0("tolist")?.extract()?;
        let rows: Vec<Vec<f64>> = df
            .call_method0("to_numpy")?
            .call_method0("tolist")?
            .extract()?;
        Self::from_rows(rows, Some(columns))
    }

    /// The column names, in order.
    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    /// `(rows, columns)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("Frame({r} rows x {c} cols)")
    }
}

// ---------------------------------------------------------------------------
// Table & Profile — dtype-aware ingest and automated EDA (feature: eda).
// ---------------------------------------------------------------------------

/// A raw, dtype-aware table (strings, categories, dates, nulls) read from CSV
/// or Parquet — the front of the lifecycle, before it lowers to a numeric
/// `Frame`.
#[cfg(feature = "eda")]
#[pyclass(name = "Table")]
#[derive(Clone)]
struct PyTable {
    inner: Table,
}

#[cfg(feature = "eda")]
#[pymethods]
impl PyTable {
    /// Read a CSV file, inferring the schema.
    #[staticmethod]
    fn from_csv(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: Table::from_csv(path).map_err(to_py_err)?,
        })
    }

    /// Read a Parquet file.
    #[staticmethod]
    fn from_parquet(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: Table::from_parquet(path).map_err(to_py_err)?,
        })
    }

    /// Build a numeric table from a `Frame`.
    #[staticmethod]
    fn from_frame(frame: &PyFrame) -> PyResult<Self> {
        Ok(Self {
            inner: Table::from_frame(&frame.inner).map_err(to_py_err)?,
        })
    }

    /// Lower to a numeric `Frame` (categoricals one-hot encoded).
    fn to_frame(&self) -> PyResult<PyFrame> {
        Ok(PyFrame {
            inner: self.inner.to_frame().map_err(to_py_err)?,
        })
    }

    /// `(rows, columns)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("Table({r} rows x {c} cols)")
    }
}

/// Automated EDA over a `Table` (or `Frame`): typed per-column summaries,
/// alerts, and a suggested preprocessing pipeline — rendered to an HTML report.
#[cfg(feature = "eda")]
#[pyclass(name = "Profile")]
struct PyProfile {
    inner: Profile,
}

#[cfg(feature = "eda")]
#[pymethods]
impl PyProfile {
    /// Profile a `Table` or a `Frame`.
    #[staticmethod]
    fn of(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let table = table_arg(data)?;
        Ok(Self {
            inner: Profile::of(&table).map_err(to_py_err)?,
        })
    }

    /// Profile with a known target column (enables target-aware alerts).
    #[staticmethod]
    fn of_with_target(data: &Bound<'_, PyAny>, target: &str) -> PyResult<Self> {
        let table = table_arg(data)?;
        Ok(Self {
            inner: Profile::of_with_target(&table, target).map_err(to_py_err)?,
        })
    }

    /// Write the EDA report to an HTML file.
    fn to_html(&self, path: String) -> PyResult<()> {
        self.inner.to_html(path).map_err(to_py_err)
    }
}

// ---------------------------------------------------------------------------
// Transformer & estimator objects.
//
// Each is a lightweight descriptor the pipeline lowers to the concrete Rust
// type when it is added as a step. `extract` clones the descriptor out of the
// Python object, so every class derives `Clone`.
// ---------------------------------------------------------------------------

/// Standardize each column to zero mean and unit variance.
#[pyclass(name = "StandardScaler")]
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
#[pyclass(name = "MinMaxScaler")]
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
#[pyclass(name = "SimpleImputer")]
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
#[pyclass(name = "OneHotEncoder")]
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
#[pyclass(name = "RandomForest")]
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

/// An ordinary-least-squares regressor.
#[pyclass(name = "LinearRegression")]
#[derive(Clone)]
struct PyLinearRegression;
#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// A pre-trained ONNX model (e.g. exported from scikit-learn or PyTorch), used
/// as a pipeline's frozen estimator behind Millwright's preprocessing steps.
#[cfg(feature = "onnx")]
#[pyclass(name = "OnnxModel")]
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
    if obj.extract::<PyLinearRegression>().is_ok() {
        return Ok(pipe.estimator(name, LinearRegression::new()));
    }
    #[cfg(feature = "onnx")]
    if let Ok(m) = obj.extract::<PyOnnxModel>() {
        let model = crate::onnx::InferenceModel::load(&m.path).map_err(to_py_err)?;
        return Ok(pipe.estimator(name, model));
    }
    Err(PyValueError::new_err(
        "estimator expects an estimator object (RandomForest, LinearRegression, OnnxModel)",
    ))
}

// ---------------------------------------------------------------------------
// Explainer (feature: explain).
// ---------------------------------------------------------------------------

/// A SHAP explainer configuration.
#[cfg(feature = "explain")]
#[pyclass(name = "Explainer")]
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

    /// Add a named transformer step from a transformer object.
    fn step(&mut self, name: String, transformer: &Bound<'_, PyAny>) -> PyResult<()> {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = add_transformer(pipe, name, transformer)?;
        Ok(())
    }

    /// Set the final estimator from an estimator object.
    fn estimator(&mut self, name: String, estimator: &Bound<'_, PyAny>) -> PyResult<()> {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = set_estimator(pipe, name, estimator)?;
        Ok(())
    }

    // ---- convenience builders (kept for back-compat) --------------------

    /// Add a standard-scaler preprocessing step.
    #[pyo3(signature = (name=None))]
    fn standard_scaler(&mut self, name: Option<String>) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(name.unwrap_or_else(|| "scale".into()), StandardScaler::new());
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
        self.inner = pipe.step(name.unwrap_or_else(|| "encode".into()), OneHotEncoder::infer());
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
#[pyclass(name = "KFold")]
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
#[pyclass(name = "StratifiedKFold")]
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
    fn cv<'a>(
        mut slf: PyRefMut<'a, Self>,
        cv: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
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
    m.add_class::<PyLinearRegression>()?;
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
