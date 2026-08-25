//! `Profile` — automated EDA over a [`Table`], the Rust answer to
//! ydata-profiling, with a twist only a framework that owns the whole lifecycle
//! can pull off: it returns a *typed* analysis (not just an HTML blob), and
//! hands back a **suggested preprocessing pipeline** to start from.
//!
//! ```no_run
//! use millwright::prelude::*;
//!
//! # fn main() -> millwright::Result<()> {
//! # #[cfg(feature = "smartcore-backend")]
//! # {
//! let table = Table::from_csv("train.csv")?;
//! let profile = Profile::of_with_target(&table, "target")?;
//!
//! profile.to_html("eda_report.html")?;              // a shareable report
//! for alert in profile.alerts() {
//!     println!("{alert}");                            // typed, actionable
//! }
//!
//! // EDA drafts the starting pipeline — you just add the model.
//! let pipe = profile.suggest_pipeline().estimator("rf", RandomForest::new());
//! # let _ = pipe;
//! # }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use crate::error::{Error, Result};
use crate::pipeline::Pipeline;
use crate::table::{ColKind, Table};
use crate::transform::{OneHotEncoder, PowerTransform, SimpleImputer, StandardScaler, Winsorize};

// -------------------------------------------------------------------------
// Typed result
// -------------------------------------------------------------------------

/// A full, typed profile of a [`Table`]. The HTML report is just one renderer
/// over these fields.
#[derive(Clone, Debug)]
pub struct Profile {
    overview: Overview,
    columns: Vec<ColumnProfile>,
    missingness: Missingness,
    correlations: CorrMatrix,
    target: Option<TargetProfile>,
    alerts: Vec<Alert>,
}

/// Dataset-level summary.
#[derive(Clone, Debug)]
pub struct Overview {
    pub nrows: usize,
    pub ncols: usize,
    pub n_numeric: usize,
    pub n_categorical: usize,
    pub n_datetime: usize,
    pub n_boolean: usize,
    pub missing_cells: usize,
    pub total_cells: usize,
    pub duplicate_rows: usize,
}

/// A per-column profile, split by kind.
#[derive(Clone, Debug)]
pub enum ColumnProfile {
    Numeric(NumericProfile),
    Categorical(CategoricalProfile),
}

impl ColumnProfile {
    /// The column name, whatever the kind.
    pub fn name(&self) -> &str {
        match self {
            ColumnProfile::Numeric(n) => &n.name,
            ColumnProfile::Categorical(c) => &c.name,
        }
    }
    /// The number of missing values.
    pub fn missing(&self) -> usize {
        match self {
            ColumnProfile::Numeric(n) => n.missing,
            ColumnProfile::Categorical(c) => c.missing,
        }
    }
}

/// Summary statistics for a numeric column.
#[derive(Clone, Debug)]
pub struct NumericProfile {
    pub name: String,
    pub count: usize,
    pub missing: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
    pub skew: f64,
    /// Excess kurtosis (0.0 for a normal distribution).
    pub kurtosis: f64,
    pub zeros: usize,
    pub distinct: usize,
    /// Equal-width histogram bins over `[min, max]`.
    pub histogram: Vec<HistBin>,
    /// IQR-rule outlier count (values outside `[q25 - 1.5·IQR, q75 + 1.5·IQR]`).
    pub outliers: usize,
    /// Z-score outlier count (`|z| > 3`).
    pub outliers_z: usize,
}

/// One histogram bar.
#[derive(Clone, Copy, Debug)]
pub struct HistBin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
}

/// Frequency summary for a categorical / boolean / datetime column.
#[derive(Clone, Debug)]
pub struct CategoricalProfile {
    pub name: String,
    pub count: usize,
    pub missing: usize,
    pub distinct: usize,
    /// The most frequent values, `(value, count)`, descending.
    pub top: Vec<(String, usize)>,
}

/// Per-column null counts, plus columns that tend to go missing together.
#[derive(Clone, Debug)]
pub struct Missingness {
    pub per_column: Vec<(String, usize)>,
    pub total: usize,
    /// `(a, b, phi)` for column pairs whose null patterns correlate
    /// (`|phi| > 0.5`) — a hint that missingness is structural, not random.
    pub co_missing: Vec<(String, String, f64)>,
}

/// Pearson **and** Spearman correlation over the numeric columns, with high-|r|
/// (Pearson) pairs flagged.
#[derive(Clone, Debug)]
pub struct CorrMatrix {
    pub columns: Vec<String>,
    /// Pearson correlation matrix.
    pub matrix: Vec<Vec<f64>>,
    /// Spearman rank-correlation matrix (same column order).
    pub spearman: Vec<Vec<f64>>,
    /// `(a, b, r)` for Pearson `|r| > 0.95`, `a` before `b` in column order.
    pub high_pairs: Vec<(String, String, f64)>,
}

/// The target's relationship to the features.
#[derive(Clone, Debug)]
pub struct TargetProfile {
    pub name: String,
    pub kind: TargetKind,
}

/// Classification (class balance) vs. regression (feature correlations).
#[derive(Clone, Debug)]
pub enum TargetKind {
    Classification { classes: Vec<(String, usize)> },
    Regression { correlations: Vec<(String, f64)> },
}

/// An actionable data-quality finding, naming the preprocessing that answers it.
#[derive(Clone, Debug)]
pub struct Alert {
    pub column: Option<String>,
    pub message: String,
    /// The suggested step, e.g. `"SimpleImputer"`, `"OneHotEncoder"`, `"Drop"`.
    pub suggested: &'static str,
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.column {
            Some(c) => write!(f, "[{}] {} → {}", c, self.message, self.suggested),
            None => write!(f, "{} → {}", self.message, self.suggested),
        }
    }
}

// -------------------------------------------------------------------------
// Construction
// -------------------------------------------------------------------------

type KindCounts = (usize, usize, usize, usize);
type NullMask = (String, Vec<Option<f64>>);

struct ColumnAnalysis {
    columns: Vec<ColumnProfile>,
    per_column_missing: Vec<(String, usize)>,
    null_masks: Vec<NullMask>,
    kind_counts: KindCounts,
}

fn analyze_columns(table: &Table, schema: &[(String, ColKind)]) -> Result<ColumnAnalysis> {
    let mut analysis = ColumnAnalysis {
        columns: Vec::with_capacity(schema.len()),
        per_column_missing: Vec::with_capacity(schema.len()),
        null_masks: Vec::new(),
        kind_counts: (0, 0, 0, 0),
    };
    for (name, kind) in schema {
        increment_kind_count(&mut analysis.kind_counts, *kind);
        let missing = table.null_count(name)?;
        analysis.per_column_missing.push((name.clone(), missing));
        if missing > 0 {
            analysis
                .null_masks
                .push((name.clone(), null_mask(table, name, *kind)?));
        }
        analysis.columns.push(column_profile(table, name, *kind)?);
    }
    Ok(analysis)
}

fn increment_kind_count(counts: &mut KindCounts, kind: ColKind) {
    match kind {
        ColKind::Numeric => counts.0 += 1,
        ColKind::Categorical => counts.1 += 1,
        ColKind::Datetime => counts.2 += 1,
        ColKind::Boolean => counts.3 += 1,
    }
}

fn null_mask(table: &Table, name: &str, kind: ColKind) -> Result<Vec<Option<f64>>> {
    let missing = if kind == ColKind::Numeric {
        table
            .column_f64(name)?
            .iter()
            .map(Option::is_none)
            .collect::<Vec<_>>()
    } else {
        table
            .column_strings(name)?
            .iter()
            .map(Option::is_none)
            .collect::<Vec<_>>()
    };
    Ok(missing
        .into_iter()
        .map(|is_missing| Some(if is_missing { 1.0 } else { 0.0 }))
        .collect())
}

fn column_profile(table: &Table, name: &str, kind: ColKind) -> Result<ColumnProfile> {
    if kind == ColKind::Numeric {
        Ok(ColumnProfile::Numeric(numeric_profile(name, table)?))
    } else {
        Ok(ColumnProfile::Categorical(categorical_profile(
            name, table,
        )?))
    }
}

fn co_missing_pairs(null_masks: &[NullMask]) -> Vec<(String, String, f64)> {
    let mut pairs = Vec::new();
    for (index, (left_name, left_mask)) in null_masks.iter().enumerate() {
        for (right_name, right_mask) in &null_masks[index + 1..] {
            let phi = pearson(left_mask, right_mask);
            if phi.is_finite() && phi.abs() > 0.5 {
                pairs.push((left_name.clone(), right_name.clone(), phi));
            }
        }
    }
    pairs
}

fn build_overview(
    table: &Table,
    schema: &[(String, ColKind)],
    nrows: usize,
    missing_cells: usize,
    counts: KindCounts,
) -> Overview {
    Overview {
        nrows,
        ncols: schema.len(),
        n_numeric: counts.0,
        n_categorical: counts.1,
        n_datetime: counts.2,
        n_boolean: counts.3,
        missing_cells,
        total_cells: nrows * schema.len(),
        duplicate_rows: table.duplicate_rows(),
    }
}

impl Profile {
    /// Profile a table with no designated target.
    pub fn of(table: &Table) -> Result<Profile> {
        Profile::build(table, None)
    }

    /// Profile a table and analyse the relationship to `target`.
    pub fn of_with_target(table: &Table, target: &str) -> Result<Profile> {
        if table.series(target).is_err() {
            return Err(Error::Schema(format!(
                "Profile: no target column '{target}'"
            )));
        }
        Profile::build(table, Some(target))
    }

    fn build(table: &Table, target: Option<&str>) -> Result<Profile> {
        let schema = table.schema();
        let nrows = table.nrows();
        let analysis = analyze_columns(table, &schema)?;
        let missing_total = analysis.per_column_missing.iter().map(|(_, m)| m).sum();
        let overview = build_overview(table, &schema, nrows, missing_total, analysis.kind_counts);
        let missingness = Missingness {
            per_column: analysis.per_column_missing,
            total: missing_total,
            co_missing: co_missing_pairs(&analysis.null_masks),
        };

        let correlations = correlations(table, &schema)?;
        let target_profile = target
            .map(|name| target_profile(table, &schema, name))
            .transpose()?;
        let alerts = alerts(&overview, &analysis.columns, &correlations, &target_profile);

        Ok(Profile {
            overview,
            columns: analysis.columns,
            missingness,
            correlations,
            target: target_profile,
            alerts,
        })
    }

    /// The dataset-level overview.
    pub fn overview(&self) -> &Overview {
        &self.overview
    }
    /// The per-column profiles, in column order.
    pub fn columns(&self) -> &[ColumnProfile] {
        &self.columns
    }
    /// Per-column null counts.
    pub fn missingness(&self) -> &Missingness {
        &self.missingness
    }
    /// The numeric-column correlation matrix.
    pub fn correlations(&self) -> &CorrMatrix {
        &self.correlations
    }
    /// The target relationship, if a target was given.
    pub fn target(&self) -> Option<&TargetProfile> {
        self.target.as_ref()
    }
    /// The actionable data-quality alerts.
    pub fn alerts(&self) -> &[Alert] {
        &self.alerts
    }

    /// A short text summary.
    pub fn summary(&self) -> String {
        let o = &self.overview;
        let mut s = format!(
            "{} rows × {} cols  ({} numeric, {} categorical, {} datetime, {} bool)\n\
             missing: {}/{} cells   duplicate rows: {}\n",
            o.nrows,
            o.ncols,
            o.n_numeric,
            o.n_categorical,
            o.n_datetime,
            o.n_boolean,
            o.missing_cells,
            o.total_cells,
            o.duplicate_rows,
        );
        if !self.alerts.is_empty() {
            s.push_str(&format!("{} alerts:\n", self.alerts.len()));
            for a in &self.alerts {
                s.push_str(&format!("  {a}\n"));
            }
        }
        s
    }

    /// Draft a starting preprocessing [`Pipeline`] from the findings: a median
    /// imputer when anything is missing, a power transform / winsorizer for
    /// skewed or outlier-heavy columns, one-hot encoding for low-cardinality
    /// categoricals, a standard scaler over the numerics, and (with the
    /// `preprocessing` feature) a train-time SMOTE balancer when the target
    /// classes are imbalanced. The result has no estimator — add yours with
    /// [`Pipeline::estimator`](crate::pipeline::Pipeline::estimator).
    ///
    /// This assumes label-encoded input (the default
    /// [`Table::into_dataset`](crate::table::Table::into_dataset)), so don't also
    /// lower with [`CategoryEncoding::OneHot`](crate::table::CategoryEncoding),
    /// or the categoricals would be encoded twice.
    pub fn suggest_pipeline(&self) -> Pipeline {
        let mut pipe = Pipeline::new();
        if self.missingness.total > 0 {
            pipe = pipe.step("impute", SimpleImputer::median());
        }
        if self.alerts.iter().any(|a| a.suggested == "Winsorize") {
            pipe = pipe.step("winsorize", Winsorize::new());
        }
        if self.alerts.iter().any(|a| a.suggested == "PowerTransform") {
            pipe = pipe.step("power", PowerTransform::yeo_johnson());
        }
        let has_low_card_cat = self.columns.iter().any(|c| match c {
            ColumnProfile::Categorical(c) => c.distinct >= 2 && c.distinct <= 15,
            _ => false,
        });
        if has_low_card_cat {
            pipe = pipe.step("encode", OneHotEncoder::infer());
        }
        if self.overview.n_numeric > 0 {
            pipe = pipe.step("scale", StandardScaler::new());
        }
        // Class imbalance -> a train-time SMOTE balancer (needs `preprocessing`).
        #[cfg(feature = "preprocessing")]
        if self.alerts.iter().any(|a| a.suggested == "Smote") {
            pipe = pipe.balance(crate::balance::Smote::new());
        }
        pipe
    }

    /// Render a self-contained HTML report to `path`.
    pub fn to_html(&self, path: impl AsRef<Path>) -> Result<()> {
        let html = self.render_html();
        std::fs::write(path.as_ref(), html)
            .map_err(|e| Error::Backend(format!("write report: {e}")))
    }
}

// -------------------------------------------------------------------------
// Per-column computation
// -------------------------------------------------------------------------

fn numeric_profile(name: &str, table: &Table) -> Result<NumericProfile> {
    let raw = table.column_f64(name)?;
    let missing = raw.iter().filter(|v| v.is_none()).count();
    let mut present: Vec<f64> = raw
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite())
        .collect();
    let count = present.len();
    if count == 0 {
        return Ok(NumericProfile {
            name: name.into(),
            count: 0,
            missing,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            p25: f64::NAN,
            median: f64::NAN,
            p75: f64::NAN,
            max: f64::NAN,
            skew: f64::NAN,
            kurtosis: f64::NAN,
            zeros: 0,
            distinct: 0,
            histogram: Vec::new(),
            outliers: 0,
            outliers_z: 0,
        });
    }
    present.sort_by(f64::total_cmp);
    let n = count as f64;
    let mean = present.iter().sum::<f64>() / n;
    let var = present.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let (skew, kurtosis) = if std > f64::EPSILON {
        let m3 = present
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n;
        // excess kurtosis: 0.0 for a normal distribution
        let m4 = present
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n
            - 3.0;
        (m3, m4)
    } else {
        (0.0, 0.0)
    };
    let (min, max) = (present[0], present[count - 1]);
    let p25 = quantile(&present, 0.25);
    let median = quantile(&present, 0.50);
    let p75 = quantile(&present, 0.75);
    let iqr = p75 - p25;
    let (lo, hi) = (p25 - 1.5 * iqr, p75 + 1.5 * iqr);
    let outliers = present.iter().filter(|&&x| x < lo || x > hi).count();
    let outliers_z = if std > f64::EPSILON {
        present
            .iter()
            .filter(|&&x| ((x - mean) / std).abs() > 3.0)
            .count()
    } else {
        0
    };
    let zeros = present.iter().filter(|&&x| x == 0.0).count();
    let distinct = {
        let mut d = present.clone();
        d.dedup();
        d.len()
    };
    let histogram = histogram(&present, min, max, 10);

    Ok(NumericProfile {
        name: name.into(),
        count,
        missing,
        mean,
        std,
        min,
        p25,
        median,
        p75,
        max,
        skew,
        kurtosis,
        zeros,
        distinct,
        histogram,
        outliers,
        outliers_z,
    })
}

fn categorical_profile(name: &str, table: &Table) -> Result<CategoricalProfile> {
    let raw = table.column_strings(name)?;
    let missing = raw.iter().filter(|v| v.is_none()).count();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut count = 0;
    for v in raw.into_iter().flatten() {
        *counts.entry(v).or_insert(0) += 1;
        count += 1;
    }
    let distinct = counts.len();
    let mut top: Vec<(String, usize)> = counts.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top.truncate(10);
    Ok(CategoricalProfile {
        name: name.into(),
        count,
        missing,
        distinct,
        top,
    })
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn histogram(sorted: &[f64], min: f64, max: f64, bins: usize) -> Vec<HistBin> {
    if min == max {
        return vec![HistBin {
            lo: min,
            hi: max,
            count: sorted.len(),
        }];
    }
    let width = (max - min) / bins as f64;
    let mut out: Vec<HistBin> = (0..bins)
        .map(|i| HistBin {
            lo: min + i as f64 * width,
            hi: min + (i + 1) as f64 * width,
            count: 0,
        })
        .collect();
    for &x in sorted {
        let mut idx = ((x - min) / width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        out[idx].count += 1;
    }
    out
}

// -------------------------------------------------------------------------
// Correlations & target
// -------------------------------------------------------------------------

fn numeric_columns(schema: &[(String, ColKind)]) -> Vec<String> {
    schema
        .iter()
        .filter(|(_, k)| *k == ColKind::Numeric)
        .map(|(n, _)| n.clone())
        .collect()
}

fn correlations(table: &Table, schema: &[(String, ColKind)]) -> Result<CorrMatrix> {
    let columns = numeric_columns(schema);
    let cols: Vec<Vec<Option<f64>>> = columns
        .iter()
        .map(|n| table.column_f64(n))
        .collect::<Result<_>>()?;

    let k = columns.len();
    let mut matrix = vec![vec![f64::NAN; k]; k];
    let mut spearman_mat = vec![vec![f64::NAN; k]; k];
    let mut high_pairs = Vec::new();
    for i in 0..k {
        matrix[i][i] = 1.0;
        spearman_mat[i][i] = 1.0;
        for j in (i + 1)..k {
            let r = pearson(&cols[i], &cols[j]);
            matrix[i][j] = r;
            matrix[j][i] = r;
            let rs = spearman(&cols[i], &cols[j]);
            spearman_mat[i][j] = rs;
            spearman_mat[j][i] = rs;
            if r.is_finite() && r.abs() > 0.95 {
                high_pairs.push((columns[i].clone(), columns[j].clone(), r));
            }
        }
    }
    Ok(CorrMatrix {
        columns,
        matrix,
        spearman: spearman_mat,
        high_pairs,
    })
}

/// Spearman rank correlation: Pearson over the ranks of the paired, present,
/// finite values (ties share their average rank).
fn spearman(xs: &[Option<f64>], ys: &[Option<f64>]) -> f64 {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter_map(|(a, b)| match (a, b) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some((*a, *b)),
            _ => None,
        })
        .collect();
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let rx = ranks(&pairs.iter().map(|(a, _)| *a).collect::<Vec<_>>());
    let ry = ranks(&pairs.iter().map(|(_, b)| *b).collect::<Vec<_>>());
    let rx: Vec<Option<f64>> = rx.into_iter().map(Some).collect();
    let ry: Vec<Option<f64>> = ry.into_iter().map(Some).collect();
    pearson(&rx, &ry)
}

/// Fractional ranks (1-based), averaging ties.
fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // ranks i..j (0-based) share the average 1-based rank
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            out[k] = avg;
        }
        i = j;
    }
    out
}

/// Pearson correlation over the rows where both values are present and finite.
fn pearson(xs: &[Option<f64>], ys: &[Option<f64>]) -> f64 {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter_map(|(a, b)| match (a, b) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some((*a, *b)),
            _ => None,
        })
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = pairs.iter().map(|(a, _)| a).sum::<f64>() / n;
    let my = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (a, b) in &pairs {
        cov += (a - mx) * (b - my);
        vx += (a - mx).powi(2);
        vy += (b - my).powi(2);
    }
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

fn target_profile(
    table: &Table,
    schema: &[(String, ColKind)],
    target: &str,
) -> Result<TargetProfile> {
    let kind = table.kind(target)?;
    let is_classification = match kind {
        ColKind::Categorical | ColKind::Boolean => true,
        ColKind::Numeric => {
            // integral, low-cardinality numeric reads as class labels
            let vals: Vec<f64> = table.column_f64(target)?.into_iter().flatten().collect();
            let integral = vals.iter().all(|v| v.fract() == 0.0);
            let distinct = {
                let mut d: Vec<i64> = vals.iter().map(|v| *v as i64).collect();
                d.sort_unstable();
                d.dedup();
                d.len()
            };
            integral && distinct <= 20
        }
        ColKind::Datetime => false,
    };

    if is_classification {
        let raw = if kind == ColKind::Numeric {
            table
                .column_f64(target)?
                .into_iter()
                .map(|o| o.map(|v| format!("{}", v as i64)))
                .collect::<Vec<_>>()
        } else {
            table.column_strings(target)?
        };
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for v in raw.into_iter().flatten() {
            *counts.entry(v).or_insert(0) += 1;
        }
        let mut classes: Vec<(String, usize)> = counts.into_iter().collect();
        classes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(TargetProfile {
            name: target.into(),
            kind: TargetKind::Classification { classes },
        })
    } else {
        let y = table.column_f64(target)?;
        let mut correlations: Vec<(String, f64)> = numeric_columns(schema)
            .into_iter()
            .filter(|n| n != target)
            .map(|n| {
                let x = table.column_f64(&n).unwrap_or_default();
                (n, pearson(&x, &y))
            })
            .collect();
        correlations.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(TargetProfile {
            name: target.into(),
            kind: TargetKind::Regression { correlations },
        })
    }
}

#[path = "profile_alerts.rs"]
mod profile_alerts;
use profile_alerts::alerts;

#[path = "profile_render.rs"]
mod render;

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
