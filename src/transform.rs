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
use crate::frame::{Dataset, Frame};
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
        Frame::new(buf, n, p, self.columns.clone())
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
        Frame::new(buf, n, p, self.columns.clone())
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
        Frame::new(buf, n, p, self.columns.clone())
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
        let mut categories = Vec::with_capacity(frame.ncols());
        for (c, name) in frame.columns().iter().enumerate() {
            let col = frame.column(c);
            let selected = match &self.select {
                Some(names) => names.iter().any(|n| n == name),
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

/// Clip each column to a learned `[lower, upper]` quantile band — the robust
/// answer to heavy tails and outliers. Missing values pass through.
#[derive(Clone, Debug)]
pub struct Winsorize {
    lower_q: f64,
    upper_q: f64,
    bounds: Vec<(f64, f64)>,
    columns: Vec<String>,
    fitted: bool,
}

impl Winsorize {
    /// Clip to the 5th/95th percentiles.
    pub fn new() -> Self {
        Winsorize {
            lower_q: 0.05,
            upper_q: 0.95,
            bounds: Vec::new(),
            columns: Vec::new(),
            fitted: false,
        }
    }

    /// Clip to the given lower/upper quantiles (each in `[0, 1]`).
    pub fn quantiles(lower: f64, upper: f64) -> Self {
        Winsorize {
            lower_q: lower,
            upper_q: upper,
            ..Winsorize::new()
        }
    }
}

impl Default for Winsorize {
    fn default() -> Self {
        Winsorize::new()
    }
}

impl Transformer for Winsorize {
    fn name(&self) -> &'static str {
        "Winsorize"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (_, p) = frame.shape();
        let mut bounds = Vec::with_capacity(p);
        for c in 0..p {
            let mut vals: Vec<f64> = frame
                .column(c)
                .into_iter()
                .filter(|v| v.is_finite())
                .collect();
            vals.sort_by(f64::total_cmp);
            let lo = col_quantile(&vals, self.lower_q);
            let hi = col_quantile(&vals, self.upper_q);
            bounds.push((lo, hi));
        }
        self.bounds = bounds;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("Winsorize::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                let (lo, hi) = self.bounds[c];
                let v = frame.get(r, c);
                buf.push(if v.is_nan() { v } else { v.clamp(lo, hi) });
            }
        }
        Frame::new(buf, n, p, self.columns.clone())
    }
}

/// Yeo-Johnson power transform: spread out a skewed, heavy-tailed column toward
/// normality. A per-column `λ` is chosen from a grid to minimize skew; the
/// transform is defined for negative values too (unlike Box-Cox). Missing values
/// pass through.
#[derive(Clone, Debug)]
pub struct PowerTransform {
    lambdas: Vec<f64>,
    columns: Vec<String>,
    fitted: bool,
}

impl PowerTransform {
    /// A Yeo-Johnson transform with per-column `λ` chosen at fit time.
    pub fn yeo_johnson() -> Self {
        PowerTransform {
            lambdas: Vec::new(),
            columns: Vec::new(),
            fitted: false,
        }
    }
}

impl Default for PowerTransform {
    fn default() -> Self {
        PowerTransform::yeo_johnson()
    }
}

impl Transformer for PowerTransform {
    fn name(&self) -> &'static str {
        "PowerTransform"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (_, p) = frame.shape();
        let mut lambdas = Vec::with_capacity(p);
        for c in 0..p {
            let vals: Vec<f64> = frame
                .column(c)
                .into_iter()
                .filter(|v| v.is_finite())
                .collect();
            lambdas.push(best_lambda(&vals));
        }
        self.lambdas = lambdas;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("PowerTransform::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                let v = frame.get(r, c);
                buf.push(if v.is_nan() {
                    v
                } else {
                    yeo_johnson(v, self.lambdas[c])
                });
            }
        }
        Frame::new(buf, n, p, self.columns.clone())
    }
}

/// Apply different transformers to different column subsets, in the
/// scikit-learn `ColumnTransformer` style. Columns not named by any group pass
/// through unchanged (unless [`ColumnTransformer::drop_remainder`] is set).
///
/// Transformed groups appear first in the output, in the order they were added,
/// then the passthrough columns.
#[derive(Clone, Default)]
pub struct ColumnTransformer {
    groups: Vec<(Vec<String>, Box<dyn Transformer>)>,
    passthrough: bool,
}

impl ColumnTransformer {
    /// A column transformer that passes non-selected columns through.
    pub fn new() -> Self {
        ColumnTransformer {
            groups: Vec::new(),
            passthrough: true,
        }
    }

    /// Apply `transformer` to the named `columns`.
    pub fn add<I, S>(mut self, transformer: impl Transformer + 'static, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.groups.push((
            columns.into_iter().map(Into::into).collect(),
            Box::new(transformer),
        ));
        self
    }

    /// Drop, rather than pass through, columns not named by any group.
    pub fn drop_remainder(mut self) -> Self {
        self.passthrough = false;
        self
    }
}

impl Transformer for ColumnTransformer {
    fn name(&self) -> &'static str {
        "ColumnTransformer"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        for (cols, t) in &mut self.groups {
            let sub = sub_frame(frame, cols)?;
            t.fit(&sub)?;
        }
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        let n = frame.nrows();
        let mut out_names: Vec<String> = Vec::new();
        let mut out_cols: Vec<Vec<f64>> = Vec::new();
        let mut used: Vec<String> = Vec::new();

        for (cols, t) in &self.groups {
            let tf = t.transform(&sub_frame(frame, cols)?)?;
            for c in 0..tf.ncols() {
                out_names.push(tf.columns()[c].clone());
                out_cols.push(tf.column(c));
            }
            used.extend(cols.iter().cloned());
        }
        if self.passthrough {
            for name in frame.columns() {
                if !used.contains(name) {
                    let idx = frame.column_index(name).unwrap();
                    out_names.push(name.clone());
                    out_cols.push(frame.column(idx));
                }
            }
        }
        frame_from_columns(out_names, out_cols, n)
    }
}

/// Supervised target (mean) encoding for categorical columns: replace each
/// category with the smoothed mean of the target over its rows. Because it needs
/// the target, it fits on a [`Dataset`] and is applied before (or outside) the
/// unsupervised pipeline — not a plain [`Transformer`]. Categories are the
/// integral values of the selected columns (as produced by
/// [`Table`](crate::table::Table) label-encoding).
#[derive(Clone, Debug)]
pub struct TargetEncoder {
    columns: Vec<String>,
    smoothing: f64,
    global_mean: f64,
    maps: Vec<HashMap<i64, f64>>,
    fitted: bool,
}

impl TargetEncoder {
    /// Target-encode the named columns.
    pub fn columns<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        TargetEncoder {
            columns: names.into_iter().map(Into::into).collect(),
            smoothing: 1.0,
            global_mean: 0.0,
            maps: Vec::new(),
            fitted: false,
        }
    }

    /// Set the smoothing weight `m` (pulls small categories toward the global
    /// mean): `(sum + m·global) / (count + m)`. Default 1.0.
    pub fn smoothing(mut self, m: f64) -> Self {
        self.smoothing = m;
        self
    }

    /// Learn per-category encodings from a labelled dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let frame = data.features();
        let y = data.target();
        let n = y.len().max(1) as f64;
        self.global_mean = y.iter().sum::<f64>() / n;

        let mut maps = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            let idx = frame
                .column_index(name)
                .ok_or_else(|| Error::Schema(format!("TargetEncoder: no column '{name}'")))?;
            let mut agg: HashMap<i64, (f64, f64)> = HashMap::new();
            #[allow(clippy::needless_range_loop)] // r indexes both the frame and y
            for r in 0..frame.nrows() {
                let v = frame.get(r, idx);
                if v.is_nan() {
                    continue;
                }
                let e = agg.entry(v.round() as i64).or_insert((0.0, 0.0));
                e.0 += y[r];
                e.1 += 1.0;
            }
            let map = agg
                .into_iter()
                .map(|(k, (sum, count))| {
                    let enc = (sum + self.smoothing * self.global_mean) / (count + self.smoothing);
                    (k, enc)
                })
                .collect();
            maps.push(map);
        }
        self.maps = maps;
        self.fitted = true;
        Ok(())
    }

    /// Replace each selected column's categories with their learned encoding;
    /// unseen categories map to the global mean. Other columns pass through.
    pub fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("TargetEncoder::transform".into()));
        }
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                let name = &frame.columns()[c];
                let v = frame.get(r, c);
                let encoded = match self.columns.iter().position(|x| x == name) {
                    Some(gi) if !v.is_nan() => *self.maps[gi]
                        .get(&(v.round() as i64))
                        .unwrap_or(&self.global_mean),
                    Some(_) => self.global_mean,
                    None => v,
                };
                buf.push(encoded);
            }
        }
        Frame::new(buf, n, p, frame.columns().to_vec())
    }

    /// Fit on the dataset, then transform its features.
    pub fn fit_transform(&mut self, data: &Dataset) -> Result<Frame> {
        self.fit(data)?;
        self.transform(data.features())
    }
}

// --- helpers shared by the transformers above ---

fn col_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Yeo-Johnson transform of a single value at parameter `λ`.
fn yeo_johnson(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if (lambda).abs() < 1e-9 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-9 {
        -(-x + 1.0).ln()
    } else {
        -(((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

/// Skewness of a slice (population), or 0 for degenerate input.
fn skewness(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd < f64::EPSILON {
        return 0.0;
    }
    vals.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>() / n
}

/// Pick the Yeo-Johnson `λ` from a grid that minimizes the transformed skew.
fn best_lambda(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 1.0;
    }
    let mut best = (1.0, f64::INFINITY);
    let mut lambda = -2.0;
    while lambda <= 2.0 + 1e-9 {
        let transformed: Vec<f64> = vals.iter().map(|&x| yeo_johnson(x, lambda)).collect();
        let s = skewness(&transformed).abs();
        if s < best.1 {
            best = (lambda, s);
        }
        lambda += 0.25;
    }
    best.0
}

fn sub_frame(frame: &Frame, names: &[String]) -> Result<Frame> {
    let mut cols = Vec::with_capacity(names.len());
    for n in names {
        let idx = frame
            .column_index(n)
            .ok_or_else(|| Error::Schema(format!("ColumnTransformer: no column '{n}'")))?;
        cols.push(frame.column(idx));
    }
    frame_from_columns(names.to_vec(), cols, frame.nrows())
}

fn frame_from_columns(names: Vec<String>, cols: Vec<Vec<f64>>, nrows: usize) -> Result<Frame> {
    let ncols = names.len();
    let mut buf = vec![0.0; nrows * ncols];
    for (c, col) in cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            buf[r * ncols + c] = v;
        }
    }
    Frame::new(buf, nrows, ncols, names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winsorize_clips_to_band() {
        // 0..=10; 5th/95th percentiles clip the extremes inward.
        let rows: Vec<Vec<f64>> = (0..=10).map(|i| vec![i as f64]).collect();
        let f = Frame::from_rows(rows, vec!["x".into()]).unwrap();
        let out = Winsorize::quantiles(0.1, 0.9).fit_transform(&f).unwrap();
        let col = out.column(0);
        // bounds are the 10th/90th pct = 1 and 9; 0 -> 1, 10 -> 9.
        assert_eq!(col[0], 1.0);
        assert_eq!(col[10], 9.0);
        assert_eq!(col[5], 5.0);
    }

    #[test]
    fn power_transform_reduces_skew() {
        // a right-skewed column
        let rows: Vec<Vec<f64>> = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 20.0]
            .iter()
            .map(|&x| vec![x])
            .collect();
        let f = Frame::from_rows(rows, vec!["x".into()]).unwrap();
        let before = skewness(&f.column(0)).abs();
        let out = PowerTransform::yeo_johnson().fit_transform(&f).unwrap();
        let after = skewness(&out.column(0)).abs();
        assert!(after < before, "skew not reduced: {before} -> {after}");
    }

    #[test]
    fn column_transformer_scales_one_group_passes_rest() {
        let f = Frame::from_rows(
            vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        // scale only "a"; "b" passes through
        let out = ColumnTransformer::new()
            .add(StandardScaler::new(), ["a"])
            .fit_transform(&f)
            .unwrap();
        assert_eq!(out.columns(), &["a".to_string(), "b".into()]);
        // a standardized (mean 0), b untouched
        assert!((out.column(0).iter().sum::<f64>()).abs() < 1e-9);
        assert_eq!(out.column(1), vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn target_encoder_maps_category_to_mean_target() {
        // cat in {0,1}; target mean is 1.0 for cat 0, 0.0 for cat 1
        let x = Frame::from_rows(
            vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]],
            vec!["cat".into()],
        )
        .unwrap();
        let ds = Dataset::new(x.clone(), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        // no smoothing for an exact check
        let out = TargetEncoder::columns(["cat"])
            .smoothing(0.0)
            .fit_transform(&ds)
            .unwrap();
        assert_eq!(out.column(0), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn standardizes_to_zero_mean_unit_std() {
        let f = Frame::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]], vec!["x".into()]).unwrap();
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&f).unwrap();
        let col = out.column(0);
        let mean: f64 = col.iter().sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-9);
        // population std of the standardized column is 1
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0;
        assert!((var - 1.0).abs() < 1e-9);
    }

    #[test]
    fn transform_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(StandardScaler::new().transform(&f).is_err());
    }

    #[test]
    fn minmax_maps_to_unit_interval() {
        let f =
            Frame::from_rows(vec![vec![10.0], vec![20.0], vec![30.0]], vec!["x".into()]).unwrap();
        let mut s = MinMaxScaler::new();
        let out = s.fit_transform(&f).unwrap();
        assert_eq!(out.column(0), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn imputer_fills_mean_and_median() {
        let nan = f64::NAN;
        let f = Frame::from_rows(
            vec![
                vec![1.0, 1.0],
                vec![nan, 2.0],
                vec![3.0, nan],
                vec![5.0, 4.0],
            ],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        // mean of a's present {1,3,5} = 3; median of b's present {1,2,4} = 2
        let mut mean = SimpleImputer::mean();
        let om = mean.fit_transform(&f).unwrap();
        assert_eq!(om.get(1, 0), 3.0);
        let mut med = SimpleImputer::median();
        let od = med.fit_transform(&f).unwrap();
        assert_eq!(od.get(2, 1), 2.0);
    }

    #[test]
    fn onehot_expands_selected_column() {
        let f = Frame::from_rows(
            vec![
                vec![0.0, 5.0],
                vec![2.0, 6.0],
                vec![1.0, 7.0],
                vec![0.0, 8.0],
            ],
            vec!["cat".into(), "num".into()],
        )
        .unwrap();
        let mut enc = OneHotEncoder::columns(["cat"]);
        let out = enc.fit_transform(&f).unwrap();
        // cat has categories {0,1,2} -> 3 indicator cols + passthrough num = 4
        assert_eq!(
            out.columns(),
            &[
                "cat=0".to_string(),
                "cat=1".into(),
                "cat=2".into(),
                "num".into()
            ]
        );
        assert_eq!(out.row(0), &[1.0, 0.0, 0.0, 5.0]); // cat=0
        assert_eq!(out.row(1), &[0.0, 0.0, 1.0, 6.0]); // cat=2
    }

    #[test]
    fn onehot_infers_low_cardinality_integer_columns() {
        let f = Frame::from_rows(
            vec![vec![0.0, 1.5], vec![1.0, 2.5], vec![0.0, 3.5]],
            vec!["flag".into(), "cont".into()],
        )
        .unwrap();
        let mut enc = OneHotEncoder::infer();
        let out = enc.fit_transform(&f).unwrap();
        // flag is integral low-cardinality -> expanded; cont is continuous -> passthrough
        assert_eq!(
            out.columns(),
            &["flag=0".to_string(), "flag=1".into(), "cont".into()]
        );
    }
}
