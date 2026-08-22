//! The `Frame` data model — the boundary type the whole public API speaks.
//!
//! A [`Frame`] is a contiguous, row-major `f64` buffer plus a schema (column
//! names). It is the one type users pass around; backend adapters convert it
//! to their native array type *at the edge only* (see the smartcore adapter's
//! `as_dense`). This is how Millwright keeps the two-`ndarray`-worlds problem
//! out of user code — exactly how pandas/NumPy sit under scikit-learn.
//!
//! A [`Dataset`] is a `Frame` paired with a target column: what an
//! [`Estimator`](crate::traits::Estimator) fits on.

use std::path::Path;

use crate::error::{Error, Result};

/// The role of a column, so schema-aware steps (encoders, profiling) need not
/// guess. Defaults to [`Numeric`](Dtype::Numeric); [`Table`](crate::table::Table)
/// marks the columns it knows are [`Categorical`](Dtype::Categorical) as it
/// lowers into a `Frame`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// A continuous or already-numeric column.
    Numeric,
    /// A nominal category, carried as an integer code.
    Categorical,
}

/// A contiguous, row-major table of `f64` with named columns.
///
/// Layout: element `(r, c)` lives at `buf[r * ncols + c]`. Each column also
/// carries a [`Dtype`] (defaulting to `Numeric`) so downstream steps know which
/// columns are truly categorical rather than inferring it from the values.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    buf: Vec<f64>,
    nrows: usize,
    ncols: usize,
    columns: Vec<String>,
    dtypes: Vec<Dtype>,
}

impl Frame {
    /// Build a frame from a flat row-major buffer.
    ///
    /// Fails if `buf.len() != nrows * ncols` or if `columns.len() != ncols`.
    pub fn new(buf: Vec<f64>, nrows: usize, ncols: usize, columns: Vec<String>) -> Result<Self> {
        if buf.len() != nrows * ncols {
            return Err(Error::Shape(format!(
                "buffer has {} elements but shape {}x{} needs {}",
                buf.len(),
                nrows,
                ncols,
                nrows * ncols
            )));
        }
        if columns.len() != ncols {
            return Err(Error::Schema(format!(
                "{} column names for {} columns",
                columns.len(),
                ncols
            )));
        }
        Ok(Frame {
            buf,
            nrows,
            ncols,
            columns,
            dtypes: vec![Dtype::Numeric; ncols],
        })
    }

    /// Set the per-column dtypes (builder). Fails if the count disagrees with
    /// the number of columns.
    pub fn with_dtypes(mut self, dtypes: Vec<Dtype>) -> Result<Self> {
        if dtypes.len() != self.ncols {
            return Err(Error::Schema(format!(
                "{} dtypes for {} columns",
                dtypes.len(),
                self.ncols
            )));
        }
        self.dtypes = dtypes;
        Ok(self)
    }

    /// Build a frame from a vector of equal-length rows.
    pub fn from_rows(rows: Vec<Vec<f64>>, columns: Vec<String>) -> Result<Self> {
        let nrows = rows.len();
        let ncols = columns.len();
        let mut buf = Vec::with_capacity(nrows * ncols);
        for (r, row) in rows.into_iter().enumerate() {
            if row.len() != ncols {
                return Err(Error::Shape(format!(
                    "row {r} has {} values, expected {ncols}",
                    row.len()
                )));
            }
            buf.extend(row);
        }
        Frame::new(buf, nrows, ncols, columns)
    }

    /// Read an all-numeric CSV (a header row, then `f64` rows) into a frame.
    ///
    /// Empty cells become `NaN` (ready for a [`SimpleImputer`]). This is the
    /// dependency-free fast path for already-numeric data; for typed data
    /// (strings, categories, dates) use [`Table`](crate::table::Table) behind
    /// the `eda` feature.
    ///
    /// [`SimpleImputer`]: crate::transform::SimpleImputer
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Frame> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Backend(format!("read csv: {e}")))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines
            .next()
            .ok_or_else(|| Error::Schema("empty CSV".into()))?;
        let columns: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
        let ncols = columns.len();
        let mut rows = Vec::new();
        for (i, line) in lines.enumerate() {
            let mut row = Vec::with_capacity(ncols);
            for cell in line.split(',') {
                let t = cell.trim();
                row.push(if t.is_empty() {
                    f64::NAN
                } else {
                    t.parse::<f64>().map_err(|_| {
                        Error::Schema(format!("row {}: '{t}' is not a number", i + 2))
                    })?
                });
            }
            rows.push(row);
        }
        Frame::from_rows(rows, columns)
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// The column names, in order.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// The per-column dtypes (all [`Dtype::Numeric`] unless set).
    pub fn dtypes(&self) -> &[Dtype] {
        &self.dtypes
    }

    /// The dtype of column `c`.
    pub fn dtype(&self, c: usize) -> Dtype {
        self.dtypes[c]
    }

    /// The indices of columns marked [`Dtype::Categorical`].
    pub fn categorical_columns(&self) -> Vec<usize> {
        (0..self.ncols)
            .filter(|&c| self.dtypes[c] == Dtype::Categorical)
            .collect()
    }

    /// The flat row-major backing buffer.
    pub fn buf(&self) -> &[f64] {
        &self.buf
    }

    /// Index of a column by name, if present.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// A single row as a contiguous slice.
    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.ncols;
        &self.buf[start..start + self.ncols]
    }

    /// Element at `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.buf[r * self.ncols + c]
    }

    /// Extract one column as an owned vector.
    pub fn column(&self, c: usize) -> Vec<f64> {
        (0..self.nrows).map(|r| self.get(r, c)).collect()
    }

    /// Materialize the frame as a vector of rows.
    ///
    /// This is the shape most backend adapters want at the conversion edge.
    pub fn as_rows(&self) -> Vec<Vec<f64>> {
        (0..self.nrows).map(|r| self.row(r).to_vec()).collect()
    }

    /// A new frame holding only the given rows, in the given order.
    ///
    /// Used to materialize CV folds and bootstrap samples. Row indices are
    /// assumed in-range.
    pub fn select_rows(&self, idx: &[usize]) -> Frame {
        let mut buf = Vec::with_capacity(idx.len() * self.ncols);
        for &r in idx {
            buf.extend_from_slice(self.row(r));
        }
        Frame {
            buf,
            nrows: idx.len(),
            ncols: self.ncols,
            columns: self.columns.clone(),
            dtypes: self.dtypes.clone(),
        }
    }

    /// Assert that another frame has the same columns (used when a fitted step
    /// is applied to new data).
    pub(crate) fn require_columns(&self, expected: &[String]) -> Result<()> {
        if self.columns != expected {
            return Err(Error::Schema(format!(
                "expected columns {expected:?}, got {:?}",
                self.columns
            )));
        }
        Ok(())
    }
}

/// A training dataset: a feature [`Frame`] plus a target column.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    features: Frame,
    target: Vec<f64>,
}

impl Dataset {
    /// Pair features with a target. Fails if the lengths disagree.
    pub fn new(features: Frame, target: Vec<f64>) -> Result<Self> {
        if target.len() != features.nrows() {
            return Err(Error::Shape(format!(
                "target has {} values but frame has {} rows",
                target.len(),
                features.nrows()
            )));
        }
        Ok(Dataset { features, target })
    }

    /// The feature frame.
    pub fn features(&self) -> &Frame {
        &self.features
    }

    /// The target column.
    pub fn target(&self) -> &[f64] {
        &self.target
    }

    /// Replace the feature frame, keeping the target (used by `Pipeline::fit`
    /// as it threads a frame through its transformers).
    pub(crate) fn with_features(&self, features: Frame) -> Dataset {
        Dataset {
            features,
            target: self.target.clone(),
        }
    }

    /// A new dataset holding only the given rows, in the given order — a CV
    /// fold or a bootstrap sample.
    pub fn select(&self, idx: &[usize]) -> Dataset {
        Dataset {
            features: self.features.select_rows(idx),
            target: idx.iter().map(|&i| self.target[i]).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffer() {
        assert!(Frame::new(vec![1.0, 2.0], 2, 2, vec!["a".into(), "b".into()]).is_err());
    }

    #[test]
    fn round_trips_rows() {
        let f = Frame::from_rows(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(f.shape(), (2, 2));
        assert_eq!(f.get(1, 0), 3.0);
        assert_eq!(f.column(1), vec![2.0, 4.0]);
        assert_eq!(f.as_rows(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn reads_numeric_csv_with_blanks_as_nan() {
        let path = std::env::temp_dir().join("mw_frame_from_csv.csv");
        std::fs::write(&path, "a,b\n1,2\n3,\n").unwrap();
        let f = Frame::from_csv(&path).unwrap();
        assert_eq!(f.shape(), (2, 2));
        assert_eq!(f.get(0, 1), 2.0);
        assert!(f.get(1, 1).is_nan());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dtypes_default_numeric_and_survive_selection() {
        let f = Frame::from_rows(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(f.dtypes(), &[Dtype::Numeric, Dtype::Numeric]);

        let typed = f
            .with_dtypes(vec![Dtype::Categorical, Dtype::Numeric])
            .unwrap();
        assert_eq!(typed.categorical_columns(), vec![0]);
        assert_eq!(typed.dtype(0), Dtype::Categorical);
        // dtypes are preserved through a CV-style row selection
        assert_eq!(
            typed.select_rows(&[1]).dtypes(),
            &[Dtype::Categorical, Dtype::Numeric]
        );
    }

    #[test]
    fn dataset_checks_lengths() {
        let f = Frame::from_rows(vec![vec![1.0], vec![2.0]], vec!["a".into()]).unwrap();
        assert!(Dataset::new(f.clone(), vec![0.0]).is_err());
        assert!(Dataset::new(f, vec![0.0, 1.0]).is_ok());
    }
}
