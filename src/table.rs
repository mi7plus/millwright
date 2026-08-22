//! `Table` — the typed, polars-backed front of the lifecycle.
//!
//! The framework's [`Frame`] is deliberately numeric: a
//! contiguous `f64` matrix that every backend and transformer speaks. But real
//! data arrives as CSV or Parquet with strings, categories, dates, booleans, and
//! nulls. [`Table`] is that raw, dtype-aware world — a thin wrapper over a
//! [`polars`] `DataFrame` — and it *lowers* into a `Frame` at the edge, exactly
//! as the design's lifecycle diagram promises: `polars → Frame`.
//!
//! ```no_run
//! use millwright::prelude::*;
//!
//! # fn main() -> millwright::Result<()> {
//! let table = Table::from_csv("customers.csv")?;   // strings, dates, nulls and all
//! let profile = Profile::of(&table)?;              // look before you leap
//! profile.to_html("eda.html")?;
//!
//! // lower to the numeric world for modelling
//! let train = table.into_dataset("churned")?;
//! # let _ = train;
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use polars::prelude::*;

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};

fn polars_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Backend(format!("polars: {e}"))
}

/// The coarse kind of a column, inferred from its polars dtype. This is the
/// distinction that drives profiling and the suggested preprocessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColKind {
    /// Integer or floating-point.
    Numeric,
    /// True/false.
    Boolean,
    /// String, categorical, or enum — a label.
    Categorical,
    /// Date, datetime, time, or duration.
    Datetime,
}

impl ColKind {
    fn of(dtype: &DataType) -> ColKind {
        match dtype {
            DataType::Boolean => ColKind::Boolean,
            DataType::String | DataType::Categorical(..) | DataType::Enum(..) => {
                ColKind::Categorical
            }
            DataType::Date | DataType::Datetime(..) | DataType::Time | DataType::Duration(..) => {
                ColKind::Datetime
            }
            d if d.is_primitive_numeric() => ColKind::Numeric,
            // Anything exotic (lists, structs, …) is treated as a label.
            _ => ColKind::Categorical,
        }
    }
}

/// A raw, dtype-aware table backed by a polars `DataFrame`.
#[derive(Clone, Debug)]
pub struct Table {
    df: DataFrame,
}

impl Table {
    /// Read a CSV file, inferring the schema from the first rows.
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Table> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000))
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))
            .map_err(polars_err)?
            .finish()
            .map_err(polars_err)?;
        Ok(Table { df })
    }

    /// Read a Parquet file.
    pub fn from_parquet(path: impl AsRef<Path>) -> Result<Table> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| Error::Backend(format!("open parquet: {e}")))?;
        let df = ParquetReader::new(file).finish().map_err(polars_err)?;
        Ok(Table { df })
    }

    /// Wrap an existing polars `DataFrame`.
    pub fn from_polars(df: DataFrame) -> Table {
        Table { df }
    }

    /// Borrow the underlying polars `DataFrame`.
    pub fn as_polars(&self) -> &DataFrame {
        &self.df
    }

    /// `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    /// The number of rows.
    pub fn nrows(&self) -> usize {
        self.df.height()
    }

    /// Column names, in order.
    pub fn column_names(&self) -> Vec<String> {
        self.df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// The `(name, kind)` schema, in column order.
    pub fn schema(&self) -> Vec<(String, ColKind)> {
        self.df
            .get_column_names()
            .into_iter()
            .zip(self.df.dtypes())
            .map(|(name, dt)| (name.to_string(), ColKind::of(&dt)))
            .collect()
    }

    /// The inferred kind of one column.
    pub fn kind(&self, name: &str) -> Result<ColKind> {
        Ok(ColKind::of(self.series(name)?.dtype()))
    }

    /// The number of nulls in one column.
    pub fn null_count(&self, name: &str) -> Result<usize> {
        Ok(self.series(name)?.null_count())
    }

    /// The number of fully duplicated rows.
    pub fn duplicate_rows(&self) -> usize {
        match self.df.is_duplicated() {
            Ok(mask) => mask.iter().filter(|b| *b == Some(true)).count(),
            Err(_) => 0,
        }
    }

    /// The first `n` rows — a cheap preview for profiling a large file.
    pub fn head(&self, n: usize) -> Table {
        Table {
            df: self.df.head(Some(n)),
        }
    }

    pub(crate) fn series(&self, name: &str) -> Result<&Series> {
        Ok(self
            .df
            .column(name)
            .map_err(|_| Error::Schema(format!("Table has no column '{name}'")))?
            .as_materialized_series())
    }

    /// One column as `f64` values, nulls as `None`. Temporal columns lower to
    /// their integer timestamp; categoricals/strings are label-encoded to the
    /// index of their (sorted) distinct value.
    pub(crate) fn column_f64(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let s = self.series(name)?;
        match ColKind::of(s.dtype()) {
            ColKind::Categorical => Ok(label_encode(&series_strings(s)?)),
            _ => series_f64(s),
        }
    }

    /// One column as owned strings, nulls preserved — the raw labels a
    /// categorical profile summarizes.
    pub(crate) fn column_strings(&self, name: &str) -> Result<Vec<Option<String>>> {
        series_strings(self.series(name)?)
    }

    /// Lower the whole table to a numeric [`Frame`], label-encoding categoricals
    /// (nulls become `NaN`, so a downstream
    /// [`SimpleImputer`](crate::transform::SimpleImputer) catches them).
    pub fn to_frame(&self) -> Result<Frame> {
        self.to_frame_with(CategoryEncoding::Label)
    }

    /// Lower to a numeric [`Frame`] with an explicit categorical encoding —
    /// [`CategoryEncoding::OneHot`] expands nominal columns into 0/1 indicator
    /// columns rather than misrepresenting them as ordinal codes.
    pub fn to_frame_with(&self, encoding: CategoryEncoding) -> Result<Frame> {
        let names = self.column_names();
        self.frame_from(&names, encoding)
    }

    /// Lower to a [`Dataset`]: every column except `target` becomes the feature
    /// frame (label-encoded categoricals), and `target` becomes the label vector.
    pub fn into_dataset(&self, target: &str) -> Result<Dataset> {
        self.into_dataset_with(target, CategoryEncoding::Label)
    }

    /// Like [`into_dataset`](Self::into_dataset) but with an explicit feature
    /// encoding. The target is always label-encoded (a single column), whatever
    /// the feature encoding.
    pub fn into_dataset_with(&self, target: &str, encoding: CategoryEncoding) -> Result<Dataset> {
        if self.series(target).is_err() {
            return Err(Error::Schema(format!(
                "into_dataset: no target column '{target}'"
            )));
        }
        let feature_names: Vec<String> = self
            .column_names()
            .into_iter()
            .filter(|n| n != target)
            .collect();
        if feature_names.is_empty() {
            return Err(Error::Schema(
                "into_dataset: table has no feature columns".into(),
            ));
        }
        let features = self.frame_from(&feature_names, encoding)?;
        let target_col = self
            .column_f64(target)?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        Dataset::new(features, target_col)
    }

    /// Build a row-major [`Frame`] from a subset of columns, in the given order,
    /// applying `encoding` to categorical columns.
    fn frame_from(&self, names: &[String], encoding: CategoryEncoding) -> Result<Frame> {
        let nrows = self.nrows();
        let mut out_names: Vec<String> = Vec::new();
        let mut out_cols: Vec<Vec<f64>> = Vec::new();

        for name in names {
            if encoding == CategoryEncoding::OneHot && self.kind(name)? == ColKind::Categorical {
                let values = self.column_strings(name)?;
                let mut distinct: Vec<String> = values.iter().flatten().cloned().collect();
                distinct.sort();
                distinct.dedup();
                for cat in &distinct {
                    out_names.push(format!("{name}={cat}"));
                    out_cols.push(
                        values
                            .iter()
                            .map(|v| {
                                if v.as_deref() == Some(cat.as_str()) {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect(),
                    );
                }
            } else {
                out_names.push(name.clone());
                out_cols.push(
                    self.column_f64(name)?
                        .into_iter()
                        .map(|v| v.unwrap_or(f64::NAN))
                        .collect(),
                );
            }
        }

        let ncols = out_names.len();
        let mut buf = vec![0.0; nrows * ncols];
        for (c, col) in out_cols.iter().enumerate() {
            for (r, &v) in col.iter().enumerate() {
                buf[r * ncols + c] = v;
            }
        }
        Frame::new(buf, nrows, ncols, out_names)
    }
}

/// How categorical columns are lowered into the numeric [`Frame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CategoryEncoding {
    /// Map each category to an integer code — compact, but implies an ordering
    /// the values may not have.
    #[default]
    Label,
    /// Expand each category into one 0/1 indicator column per value, named
    /// `"{col}={value}"` — the nominal-correct choice for linear and tree models.
    OneHot,
}

/// Cast a series to `f64`, routing temporal types through their integer
/// representation first (polars will not cast a datetime straight to float).
fn series_f64(s: &Series) -> Result<Vec<Option<f64>>> {
    let casted = if s.dtype().is_temporal() {
        s.cast(&DataType::Int64)
            .and_then(|i| i.cast(&DataType::Float64))
    } else {
        s.cast(&DataType::Float64)
    }
    .map_err(polars_err)?;
    let ca = casted.f64().map_err(polars_err)?;
    Ok(ca.iter().collect())
}

/// Materialize a column as owned strings, nulls preserved.
fn series_strings(s: &Series) -> Result<Vec<Option<String>>> {
    let casted = s.cast(&DataType::String).map_err(polars_err)?;
    let ca = casted.str().map_err(polars_err)?;
    Ok(ca.iter().map(|o| o.map(|x| x.to_string())).collect())
}

/// Map distinct non-null strings (sorted) to `0.0, 1.0, …`; nulls stay `None`.
fn label_encode(values: &[Option<String>]) -> Vec<Option<f64>> {
    let mut distinct: Vec<&String> = values.iter().flatten().collect();
    distinct.sort();
    distinct.dedup();
    let code = |v: &String| distinct.iter().position(|d| *d == v).unwrap() as f64;
    values.iter().map(|o| o.as_ref().map(&code)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Table {
        // a numeric column with a null, a string category, and a boolean.
        let df = df!(
            "n" => [Some(1.0_f64), None, Some(3.0), Some(3.0)],
            "c" => ["x", "y", "x", "z"],
            "b" => [true, false, true, false],
        )
        .unwrap();
        Table::from_polars(df)
    }

    #[test]
    fn infers_column_kinds() {
        let t = sample();
        let schema = t.schema();
        assert_eq!(t.kind("n").unwrap(), ColKind::Numeric);
        assert_eq!(t.kind("c").unwrap(), ColKind::Categorical);
        assert_eq!(t.kind("b").unwrap(), ColKind::Boolean);
        assert_eq!(schema.len(), 3);
    }

    #[test]
    fn counts_nulls() {
        assert_eq!(sample().null_count("n").unwrap(), 1);
        assert_eq!(sample().null_count("c").unwrap(), 0);
    }

    #[test]
    fn label_encodes_categoricals() {
        // distinct sorted {x,y,z} -> x=0, y=1, z=2
        let codes = sample().column_f64("c").unwrap();
        assert_eq!(codes, vec![Some(0.0), Some(1.0), Some(0.0), Some(2.0)]);
    }

    #[test]
    fn lowers_to_frame_with_nan_for_nulls() {
        let f = sample().to_frame().unwrap();
        assert_eq!(f.shape(), (4, 3));
        assert!(f.get(1, 0).is_nan()); // the missing numeric
        assert_eq!(f.get(0, 2), 1.0); // boolean true -> 1.0
    }

    #[test]
    fn into_dataset_splits_features_and_target() {
        let ds = sample().into_dataset("b").unwrap();
        assert_eq!(ds.features().shape(), (4, 2)); // n, c
        assert_eq!(ds.target(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn head_previews_rows() {
        let df = df!("x" => (0..100).map(|i| i as f64).collect::<Vec<f64>>()).unwrap();
        let t = Table::from_polars(df);
        assert_eq!(t.head(10).nrows(), 10);
    }

    #[test]
    fn one_hot_encoding_expands_categoricals() {
        // c in {x,y,z} -> 3 indicator columns; numeric n and bool b pass through.
        let f = sample().to_frame_with(CategoryEncoding::OneHot).unwrap();
        for col in ["n", "c=x", "c=y", "c=z", "b"] {
            assert!(
                f.columns().contains(&col.to_string()),
                "missing column {col}; got {:?}",
                f.columns()
            );
        }
        // row 0's category is "x"
        let cx = f.column_index("c=x").unwrap();
        assert_eq!(f.get(0, cx), 1.0);
        assert_eq!(f.get(3, cx), 0.0); // row 3 is "z"
    }
}
