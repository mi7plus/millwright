use super::*;

// ---------------------------------------------------------------------------
// Frame — the data matrix, with DataFrame / array ingest.
// ---------------------------------------------------------------------------

/// A dense matrix of features with named columns — the data every step speaks.
#[pyclass(name = "Frame", from_py_object)]
#[derive(Clone)]
pub(super) struct PyFrame {
    pub(super) inner: Frame,
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
#[pyclass(name = "Table", from_py_object)]
#[derive(Clone)]
pub(super) struct PyTable {
    pub(super) inner: Table,
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
pub(super) struct PyProfile {
    pub(super) inner: Profile,
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
