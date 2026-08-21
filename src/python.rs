//! Python bindings (`pip install millwright`) — the first pyo3 layer over the
//! now-stable Rust core.
//!
//! A thin binding, not a fork: `Pipeline` wraps the same Rust
//! [`Pipeline`](crate::pipeline::Pipeline), so Python runs the Rust engine at
//! Rust speed. This first cut speaks plain Python lists (rows of floats);
//! numpy / pandas / polars zero-copy interop layers on next.
//!
//! ```python
//! import millwright as mw
//! pipe = mw.Pipeline()
//! pipe.standard_scaler()
//! pipe.random_forest(n_trees=100, max_depth=8)
//! pipe.fit(rows, labels)          # rows: list[list[float]], labels: list[float]
//! preds = pipe.predict(rows)      # -> list[float]
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::backends::smartcore::RandomForest;
use crate::error::Error;
use crate::frame::{Dataset, Frame};
use crate::pipeline::Pipeline as CorePipeline;
use crate::traits::{Estimator, Predictor};
use crate::transform::StandardScaler;

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

    /// Add a standard-scaler preprocessing step.
    #[pyo3(signature = (name=None))]
    fn standard_scaler(&mut self, name: Option<String>) {
        let step = name.unwrap_or_else(|| "scale".into());
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.step(step, StandardScaler::new());
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

    /// Fit the pipeline on rows of features and a target vector.
    fn fit(&mut self, rows: Vec<Vec<f64>>, labels: Vec<f64>) -> PyResult<()> {
        let frame = frame_from_rows(rows)?;
        let dataset = Dataset::new(frame, labels).map_err(to_py_err)?;
        self.inner.fit(&dataset).map_err(to_py_err)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict one value per input row.
    fn predict(&self, rows: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err("pipeline is not fitted"));
        }
        let frame = frame_from_rows(rows)?;
        self.inner.predict(&frame).map_err(to_py_err)
    }

    /// The pipeline's step names, in order.
    fn steps(&self) -> Vec<String> {
        self.inner.step_names().into_iter().map(String::from).collect()
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
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
