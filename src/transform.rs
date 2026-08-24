//! Core transformers — all dependency-free and always available.
//!
//! Scaling ([`StandardScaler`], [`MinMaxScaler`]), imputation
//! ([`SimpleImputer`]), encoding ([`OneHotEncoder`]), outlier clipping
//! ([`Winsorize`]), de-skewing ([`PowerTransform`]), and per-subset composition
//! ([`ColumnTransformer`]) all implement the [`Transformer`] trait. The
//! supervised [`TargetEncoder`] needs the target, so it fits on a
//! [`Dataset`] rather than a bare frame. (SMOTE-style
//! resampling lives in the `balance` module behind the `preprocessing`
//! feature.)

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::frame::{Dataset, Dtype, Frame};
use crate::traits::{ParamValue, Transformer};

/// Standardize columns to zero mean and unit variance: `(x - mean) / std`.
///
/// Fitting learns per-column mean and (population) standard deviation.
/// Columns with zero variance are left unscaled (divided by 1).
#[derive(Clone, Debug)]
pub struct StandardScaler {
    with_mean: bool,
    with_std: bool,
    means: Vec<f64>,
    stds: Vec<f64>,
    columns: Vec<String>,
    fitted: bool,
}

impl StandardScaler {
    /// A scaler that centers and scales.
    pub fn new() -> Self {
        StandardScaler {
            with_mean: true,
            with_std: true,
            means: Vec::new(),
            stds: Vec::new(),
            columns: Vec::new(),
            fitted: false,
        }
    }

    /// Center to zero mean but do not scale.
    pub fn with_mean_only() -> Self {
        StandardScaler {
            with_std: false,
            ..StandardScaler::new()
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        StandardScaler::new()
    }
}

impl Transformer for StandardScaler {
    fn name(&self) -> &'static str {
        "StandardScaler"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (n, p) = frame.shape();
        if n == 0 {
            return Err(Error::Shape("cannot fit StandardScaler on 0 rows".into()));
        }
        let mut means = vec![0.0; p];
        let mut stds = vec![1.0; p];
        for c in 0..p {
            // pass categorical columns through unchanged (defaults are identity)
            if frame.dtype(c) == Dtype::Categorical {
                continue;
            }
            let col = frame.column(c);
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            means[c] = if self.with_mean { mean } else { 0.0 };
            let sd = var.sqrt();
            stds[c] = if self.with_std && sd > f64::EPSILON {
                sd
            } else {
                1.0
            };
        }
        self.means = means;
        self.stds = stds;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("StandardScaler::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                buf.push((frame.get(r, c) - self.means[c]) / self.stds[c]);
            }
        }
        Frame::new(buf, n, p, self.columns.clone())?.with_dtypes(frame.dtypes().to_vec())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "with_mean" => self.with_mean = value.as_bool()?,
            "with_std" => self.with_std = value.as_bool()?,
            other => {
                return Err(Error::Param(format!(
                    "StandardScaler has no parameter '{other}'"
                )))
            }
        }
        Ok(())
    }

    fn as_affine(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        // transform(x) = (x - mean) / std
        self.fitted.then(|| (self.means.clone(), self.stds.clone()))
    }
}

/// Scale each column into `[0, 1]`: `(x - min) / (max - min)`.
///
/// Columns with zero range map to `0.0`.
#[derive(Clone, Debug, Default)]
pub struct MinMaxScaler {
    mins: Vec<f64>,
    ranges: Vec<f64>,
    columns: Vec<String>,
    fitted: bool,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        MinMaxScaler::default()
    }
}

impl Transformer for MinMaxScaler {
    fn name(&self) -> &'static str {
        "MinMaxScaler"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (n, p) = frame.shape();
        if n == 0 {
            return Err(Error::Shape("cannot fit MinMaxScaler on 0 rows".into()));
        }
        let mut mins = vec![0.0; p];
        let mut ranges = vec![1.0; p];
        for c in 0..p {
            // pass categorical columns through unchanged (defaults are identity)
            if frame.dtype(c) == Dtype::Categorical {
                continue;
            }
            let col = frame.column(c);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            mins[c] = min;
            let range = max - min;
            ranges[c] = if range > f64::EPSILON { range } else { 1.0 };
        }
        self.mins = mins;
        self.ranges = ranges;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("MinMaxScaler::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                buf.push((frame.get(r, c) - self.mins[c]) / self.ranges[c]);
            }
        }
        Frame::new(buf, n, p, self.columns.clone())?.with_dtypes(frame.dtypes().to_vec())
    }

    fn as_affine(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        // transform(x) = (x - min) / range
        self.fitted
            .then(|| (self.mins.clone(), self.ranges.clone()))
    }
}

/// The fill strategy for [`SimpleImputer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImputeStrategy {
    /// Replace missing values with the column mean.
    Mean,
    /// Replace missing values with the column median.
    Median,
    /// Replace missing values with a fixed constant.
    Constant(f64),
}

/// Replace missing values (`NaN`) with a per-column statistic.
#[derive(Clone, Debug)]
pub struct SimpleImputer {
    strategy: ImputeStrategy,
    fills: Vec<f64>,
    columns: Vec<String>,
    fitted: bool,
}

impl SimpleImputer {
    /// Impute with the column mean of the non-missing values.
    pub fn mean() -> Self {
        SimpleImputer::with_strategy(ImputeStrategy::Mean)
    }

    /// Impute with the column median of the non-missing values.
    pub fn median() -> Self {
        SimpleImputer::with_strategy(ImputeStrategy::Median)
    }

    /// Impute with a fixed constant.
    pub fn constant(value: f64) -> Self {
        SimpleImputer::with_strategy(ImputeStrategy::Constant(value))
    }

    fn with_strategy(strategy: ImputeStrategy) -> Self {
        SimpleImputer {
            strategy,
            fills: Vec::new(),
            columns: Vec::new(),
            fitted: false,
        }
    }
}

impl Transformer for SimpleImputer {
    fn name(&self) -> &'static str {
        "SimpleImputer"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (_, p) = frame.shape();
        let mut fills = vec![0.0; p];
        for (c, fill) in fills.iter_mut().enumerate() {
            let present: Vec<f64> = frame
                .column(c)
                .into_iter()
                .filter(|v| !v.is_nan())
                .collect();
            *fill = match self.strategy {
                ImputeStrategy::Constant(v) => v,
                ImputeStrategy::Mean => {
                    if present.is_empty() {
                        0.0
                    } else {
                        present.iter().sum::<f64>() / present.len() as f64
                    }
                }
                ImputeStrategy::Median => median(&present),
            };
        }
        self.fills = fills;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("SimpleImputer::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                let v = frame.get(r, c);
                buf.push(if v.is_nan() { self.fills[c] } else { v });
            }
        }
        Frame::new(buf, n, p, self.columns.clone())?.with_dtypes(frame.dtypes().to_vec())
    }

    #[cfg(feature = "onnx")]
    fn onnx_prefix(&self) -> Option<crate::onnx::Prefix> {
        // transform(x) = where(isnan(x), fill, x)
        self.fitted.then(|| crate::onnx::Prefix::Impute {
            fill: self.fills.clone(),
        })
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// One-hot encode integer-coded categorical columns.
///
/// `Frame` is numeric, so categories are the (integral) distinct values of the
/// selected columns. Each selected column expands, in place, to one indicator
/// column per learned category, named `"{col}={value}"`; unseen categories at
/// transform time encode as all-zeros. Non-selected columns pass through.
#[derive(Clone, Debug, Default)]
pub struct OneHotEncoder {
    select: Option<Vec<String>>,
    max_cardinality: usize,
    // Learned at fit: for each input column, the categories to expand (empty =>
    // pass the column through unchanged).
    categories: Vec<(String, Vec<i64>)>,
    fitted: bool,
}

impl OneHotEncoder {
    /// Encode an explicit set of columns by name.
    pub fn columns<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        OneHotEncoder {
            select: Some(names.into_iter().map(Into::into).collect()),
            max_cardinality: usize::MAX,
            categories: Vec::new(),
            fitted: false,
        }
    }

    /// Infer categorical columns: those whose values are all integral with at
    /// most `max_cardinality` (default 10) distinct values.
    pub fn infer() -> Self {
        OneHotEncoder {
            select: None,
            max_cardinality: 10,
            categories: Vec::new(),
            fitted: false,
        }
    }

    /// Override the cardinality threshold used by [`OneHotEncoder::infer`].
    pub fn max_cardinality(mut self, k: usize) -> Self {
        self.max_cardinality = k;
        self
    }
}

impl Transformer for OneHotEncoder {
    fn name(&self) -> &'static str {
        "OneHotEncoder"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        // If the frame carries real dtypes (e.g. it came from a `Table`), trust
        // them; only fall back to the value heuristic when nothing is marked.
        let schema_known = self.select.is_none() && !frame.categorical_columns().is_empty();
        let mut categories = Vec::with_capacity(frame.ncols());
        for (c, name) in frame.columns().iter().enumerate() {
            let col = frame.column(c);
            let selected = match &self.select {
                Some(names) => names.iter().any(|n| n == name),
                None if schema_known => frame.dtype(c) == Dtype::Categorical,
                None => is_integral(&col) && distinct_sorted(&col).len() <= self.max_cardinality,
            };
            let cats = if selected {
                distinct_sorted(&col)
            } else {
                Vec::new()
            };
            categories.push((name.clone(), cats));
        }
        if let Some(names) = &self.select {
            for n in names {
                if !frame.columns().iter().any(|c| c == n) {
                    return Err(Error::Schema(format!("OneHotEncoder: no column '{n}'")));
                }
            }
        }
        self.categories = categories;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("OneHotEncoder::transform".into()));
        }
        let expected: Vec<String> = self.categories.iter().map(|(n, _)| n.clone()).collect();
        frame.require_columns(&expected)?;

        let mut out_cols: Vec<String> = Vec::new();
        for (name, cats) in &self.categories {
            if cats.is_empty() {
                out_cols.push(name.clone());
            } else {
                for v in cats {
                    out_cols.push(format!("{name}={v}"));
                }
            }
        }

        let n = frame.nrows();
        let mut buf = Vec::with_capacity(n * out_cols.len());
        for r in 0..n {
            for (c, (_, cats)) in self.categories.iter().enumerate() {
                let v = frame.get(r, c);
                if cats.is_empty() {
                    buf.push(v);
                } else {
                    let code = v.round() as i64;
                    for cat in cats {
                        buf.push(if *cat == code { 1.0 } else { 0.0 });
                    }
                }
            }
        }
        Frame::new(buf, n, out_cols.len(), out_cols)
    }

    #[cfg(feature = "onnx")]
    fn onnx_prefix(&self) -> Option<crate::onnx::Prefix> {
        self.fitted.then(|| crate::onnx::Prefix::OneHot {
            columns: self.categories.iter().map(|(_, c)| c.clone()).collect(),
        })
    }
}

fn is_integral(col: &[f64]) -> bool {
    col.iter().all(|v| v.is_finite() && v.fract() == 0.0)
}

fn distinct_sorted(col: &[f64]) -> Vec<i64> {
    let mut v: Vec<i64> = col
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.round() as i64)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[path = "transform_advanced.rs"]
mod advanced;
#[cfg(test)]
use advanced::skewness;
pub use advanced::{ColumnTransformer, PowerTransform, TargetEncoder, Winsorize};

#[cfg(test)]
#[path = "transform_tests.rs"]
mod tests;
