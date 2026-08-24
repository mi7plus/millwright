use super::*;

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
            // never clip a categorical column
            if frame.dtype(c) == Dtype::Categorical {
                bounds.push((f64::NEG_INFINITY, f64::INFINITY));
                continue;
            }
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
        Frame::new(buf, n, p, self.columns.clone())?.with_dtypes(frame.dtypes().to_vec())
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
            // lambda 1.0 is the Yeo-Johnson identity — leave categoricals alone
            if frame.dtype(c) == Dtype::Categorical {
                lambdas.push(1.0);
                continue;
            }
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
        Frame::new(buf, n, p, self.columns.clone())?.with_dtypes(frame.dtypes().to_vec())
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
            for (idx, name) in frame.columns().iter().enumerate() {
                if !used.contains(name) {
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
