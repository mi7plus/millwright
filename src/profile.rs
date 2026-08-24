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

        let mut columns = Vec::new();
        let mut per_column_missing = Vec::new();
        let mut null_masks: Vec<(String, Vec<Option<f64>>)> = Vec::new();
        let (mut n_numeric, mut n_categorical, mut n_datetime, mut n_boolean) = (0, 0, 0, 0);

        for (name, kind) in &schema {
            match kind {
                ColKind::Numeric => n_numeric += 1,
                ColKind::Categorical => n_categorical += 1,
                ColKind::Datetime => n_datetime += 1,
                ColKind::Boolean => n_boolean += 1,
            }
            let missing = table.null_count(name)?;
            per_column_missing.push((name.clone(), missing));

            // A null-indicator mask (1.0 = missing) for columns that have nulls,
            // so we can later find columns that go missing together.
            if missing > 0 {
                let mask: Vec<Option<f64>> = if *kind == ColKind::Numeric {
                    table
                        .column_f64(name)?
                        .iter()
                        .map(|v| Some(if v.is_none() { 1.0 } else { 0.0 }))
                        .collect()
                } else {
                    table
                        .column_strings(name)?
                        .iter()
                        .map(|v| Some(if v.is_none() { 1.0 } else { 0.0 }))
                        .collect()
                };
                null_masks.push((name.clone(), mask));
            }

            let profile = if *kind == ColKind::Numeric {
                ColumnProfile::Numeric(numeric_profile(name, table)?)
            } else {
                ColumnProfile::Categorical(categorical_profile(name, table)?)
            };
            columns.push(profile);
        }

        // Column pairs whose null patterns correlate (phi = Pearson on the masks).
        let mut co_missing = Vec::new();
        for i in 0..null_masks.len() {
            for j in (i + 1)..null_masks.len() {
                let phi = pearson(&null_masks[i].1, &null_masks[j].1);
                if phi.is_finite() && phi.abs() > 0.5 {
                    co_missing.push((null_masks[i].0.clone(), null_masks[j].0.clone(), phi));
                }
            }
        }

        let missing_total: usize = per_column_missing.iter().map(|(_, m)| m).sum();
        let overview = Overview {
            nrows,
            ncols: schema.len(),
            n_numeric,
            n_categorical,
            n_datetime,
            n_boolean,
            missing_cells: missing_total,
            total_cells: nrows * schema.len(),
            duplicate_rows: table.duplicate_rows(),
        };
        let missingness = Missingness {
            per_column: per_column_missing,
            total: missing_total,
            co_missing,
        };

        let correlations = correlations(table, &schema)?;
        let target_profile = match target {
            Some(t) => Some(target_profile(table, &schema, t)?),
            None => None,
        };
        let alerts = alerts(&overview, &columns, &correlations, &target_profile);

        Ok(Profile {
            overview,
            columns,
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

// -------------------------------------------------------------------------
// Alerts
// -------------------------------------------------------------------------

fn alerts(
    overview: &Overview,
    columns: &[ColumnProfile],
    corr: &CorrMatrix,
    target: &Option<TargetProfile>,
) -> Vec<Alert> {
    let mut out = Vec::new();
    let n = overview.nrows.max(1) as f64;

    for col in columns {
        match col {
            ColumnProfile::Numeric(np) => {
                if np.missing as f64 / n > 0.2 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("{:.0}% missing", 100.0 * np.missing as f64 / n),
                        suggested: "SimpleImputer",
                    });
                }
                if np.distinct <= 1 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: "constant / zero-variance".into(),
                        suggested: "Drop",
                    });
                }
                if np.skew.is_finite() && np.skew.abs() > 2.0 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("skewed (skew {:.1})", np.skew),
                        suggested: "PowerTransform",
                    });
                }
                if np.outliers as f64 / n > 0.01 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("{} IQR outliers", np.outliers),
                        suggested: "Winsorize",
                    });
                }
            }
            ColumnProfile::Categorical(cp) => {
                if cp.missing as f64 / n > 0.2 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("{:.0}% missing", 100.0 * cp.missing as f64 / n),
                        suggested: "SimpleImputer",
                    });
                }
                if cp.distinct > 20 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("high cardinality ({} levels)", cp.distinct),
                        suggested: "TargetEncoder",
                    });
                } else if cp.distinct >= 2 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("categorical ({} levels)", cp.distinct),
                        suggested: "OneHotEncoder",
                    });
                }
            }
        }
    }

    for (a, b, r) in &corr.high_pairs {
        out.push(Alert {
            column: Some(format!("{a} ~ {b}")),
            message: format!("correlated |r| = {:.2}", r.abs()),
            suggested: "drop one",
        });
    }

    if let Some(TargetProfile {
        kind: TargetKind::Classification { classes },
        ..
    }) = target
    {
        if let (Some((_, max)), Some((_, min))) = (classes.first(), classes.last()) {
            if *min > 0 && *max as f64 / *min as f64 >= 3.0 {
                out.push(Alert {
                    column: None,
                    message: format!("class imbalance {}:{}", max, min),
                    suggested: "Smote",
                });
            }
        }
    }

    out
}

// -------------------------------------------------------------------------
// HTML report
// -------------------------------------------------------------------------

impl Profile {
    fn render_html(&self) -> String {
        let o = &self.overview;
        let mut body = String::new();

        body.push_str(&format!(
            "<h1>Data profile</h1><div class=cards>\
             <div class=card><b>{}</b><span>rows</span></div>\
             <div class=card><b>{}</b><span>columns</span></div>\
             <div class=card><b>{:.1}%</b><span>missing</span></div>\
             <div class=card><b>{}</b><span>duplicate rows</span></div>\
             <div class=card><b>{}·{}·{}·{}</b><span>num·cat·date·bool</span></div>\
             </div>",
            o.nrows,
            o.ncols,
            if o.total_cells > 0 {
                100.0 * o.missing_cells as f64 / o.total_cells as f64
            } else {
                0.0
            },
            o.duplicate_rows,
            o.n_numeric,
            o.n_categorical,
            o.n_datetime,
            o.n_boolean,
        ));

        if !self.alerts.is_empty() {
            body.push_str("<h2>Alerts</h2><table><tr><th>Column</th><th>Finding</th><th>Suggested step</th></tr>");
            for a in &self.alerts {
                body.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td class=step>{}</td></tr>",
                    esc(a.column.as_deref().unwrap_or("—")),
                    esc(&a.message),
                    esc(a.suggested),
                ));
            }
            body.push_str("</table>");
        }

        body.push_str("<h2>Columns</h2>");
        for col in &self.columns {
            match col {
                ColumnProfile::Numeric(np) => {
                    body.push_str(&format!(
                        "<div class=col><h3>{} <span class=tag>numeric</span></h3>\
                         <div class=stats>mean {:.3} · std {:.3} · min {:.3} · median {:.3} · max {:.3} · \
                         skew {:.2} · kurtosis {:.2} · missing {} · distinct {} · \
                         outliers {} IQR / {} z</div>{}</div>",
                        esc(&np.name),
                        np.mean, np.std, np.min, np.median, np.max,
                        np.skew, np.kurtosis,
                        np.missing, np.distinct, np.outliers, np.outliers_z,
                        histogram_svg(&np.histogram),
                    ));
                }
                ColumnProfile::Categorical(cp) => {
                    let mut bars = String::new();
                    let top_n = cp.top.first().map(|(_, c)| *c).unwrap_or(1).max(1);
                    for (v, c) in &cp.top {
                        bars.push_str(&format!(
                            "<div class=bar><span class=lab>{}</span>\
                             <span class=track><span class=fill style='width:{:.1}%'></span></span>\
                             <span class=n>{}</span></div>",
                            esc(v),
                            100.0 * *c as f64 / top_n as f64,
                            c,
                        ));
                    }
                    body.push_str(&format!(
                        "<div class=col><h3>{} <span class=tag>categorical</span></h3>\
                         <div class=stats>distinct {} · missing {}</div>{}</div>",
                        esc(&cp.name),
                        cp.distinct,
                        cp.missing,
                        bars,
                    ));
                }
            }
        }

        if !self.correlations.high_pairs.is_empty() {
            body.push_str(
                "<h2>Highly correlated pairs</h2><table><tr><th>Pair</th><th>Pearson r</th><th>Spearman ρ</th></tr>",
            );
            let cols = &self.correlations.columns;
            for (a, b, r) in &self.correlations.high_pairs {
                let rho = match (
                    cols.iter().position(|c| c == a),
                    cols.iter().position(|c| c == b),
                ) {
                    (Some(i), Some(j)) => self.correlations.spearman[i][j],
                    _ => f64::NAN,
                };
                body.push_str(&format!(
                    "<tr><td>{} ~ {}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                    esc(a),
                    esc(b),
                    r,
                    rho,
                ));
            }
            body.push_str("</table>");
        }

        if !self.missingness.co_missing.is_empty() {
            body.push_str(
                "<h2>Columns missing together</h2><table><tr><th>Pair</th><th>phi</th></tr>",
            );
            for (a, b, phi) in &self.missingness.co_missing {
                body.push_str(&format!(
                    "<tr><td>{} ~ {}</td><td>{:.3}</td></tr>",
                    esc(a),
                    esc(b),
                    phi,
                ));
            }
            body.push_str("</table>");
        }

        if let Some(t) = &self.target {
            body.push_str(&format!("<h2>Target · {}</h2>", esc(&t.name)));
            match &t.kind {
                TargetKind::Classification { classes } => {
                    body.push_str("<table><tr><th>Class</th><th>Count</th></tr>");
                    for (c, n) in classes {
                        body.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>", esc(c), n));
                    }
                    body.push_str("</table>");
                }
                TargetKind::Regression { correlations } => {
                    body.push_str("<table><tr><th>Feature</th><th>corr with target</th></tr>");
                    for (c, r) in correlations {
                        body.push_str(&format!("<tr><td>{}</td><td>{:.3}</td></tr>", esc(c), r));
                    }
                    body.push_str("</table>");
                }
            }
        }

        format!("<!doctype html><html><head><meta charset=utf-8><title>Data profile</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")
    }
}

const CSS: &str = "\
:root{--ink:#16191c;--soft:#556069;--line:#dce3e8;--oxide:#bc4a1e;--patina:#2c8272;--bg:#f5f8f9}\
body{margin:0;background:#eef1f3;color:var(--ink);font:16px/1.6 system-ui,sans-serif}\
main{max-width:900px;margin:0 auto;padding:32px 24px}\
h1{margin:0 0 16px}h2{margin:28px 0 10px;font-size:1.3rem}h3{margin:0 0 6px;font-size:1rem}\
.cards{display:flex;flex-wrap:wrap;gap:10px}\
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:90px}\
.card b{display:block;font-size:1.3rem;color:var(--oxide)}.card span{font-size:.75rem;color:var(--soft)}\
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:.9rem}\
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}\
th{background:var(--bg);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--soft)}\
.step{font-family:ui-monospace,monospace;color:var(--oxide)}\
.col{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:10px}\
.tag{font-size:.68rem;color:var(--patina);border:1px solid var(--patina);border-radius:5px;padding:1px 6px;vertical-align:middle}\
.stats{font-size:.82rem;color:var(--soft);margin-bottom:8px}\
.hist{display:flex;align-items:flex-end;gap:2px;height:60px}\
.hist .b{flex:1;background:var(--oxide);opacity:.85;border-radius:2px 2px 0 0;min-height:1px}\
.bar{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:2px 0}\
.bar .lab{width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\
.bar .track{flex:1;background:var(--bg);border-radius:4px;overflow:hidden}\
.bar .fill{display:block;height:10px;background:var(--patina)}\
.bar .n{width:48px;text-align:right;color:var(--soft)}";

fn histogram_svg(bins: &[HistBin]) -> String {
    if bins.is_empty() {
        return String::new();
    }
    let max = bins.iter().map(|b| b.count).max().unwrap_or(1).max(1);
    let mut s = String::from("<div class=hist>");
    for b in bins {
        s.push_str(&format!(
            "<div class=b style='height:{:.1}%' title='[{:.2}, {:.2}) — {}'></div>",
            100.0 * b.count as f64 / max as f64,
            b.lo,
            b.hi,
            b.count,
        ));
    }
    s.push_str("</div>");
    s
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn sample() -> Table {
        let df = df!(
            "age"   => [Some(20.0_f64), Some(30.0), None, Some(40.0)],
            "city"  => ["ny", "sf", "ny", "la"],
            "const" => [1.0_f64, 1.0, 1.0, 1.0],
            "label" => [0_i64, 1, 0, 1],
        )
        .unwrap();
        Table::from_polars(df)
    }

    fn numeric<'a>(p: &'a Profile, name: &str) -> &'a NumericProfile {
        p.columns()
            .iter()
            .find_map(|c| match c {
                ColumnProfile::Numeric(n) if n.name == name => Some(n),
                _ => None,
            })
            .expect("numeric column")
    }

    #[test]
    fn overview_counts_kinds_and_missing() {
        let p = Profile::of(&sample()).unwrap();
        let o = p.overview();
        assert_eq!((o.nrows, o.ncols), (4, 4));
        assert_eq!(o.n_numeric, 3); // age, const, label
        assert_eq!(o.n_categorical, 1); // city
        assert_eq!(o.missing_cells, 1);
        assert_eq!(p.missingness().total, 1);
    }

    #[test]
    fn numeric_stats_ignore_nulls() {
        let p = Profile::of(&sample()).unwrap();
        let age = numeric(&p, "age");
        assert_eq!(age.missing, 1);
        assert!((age.mean - 30.0).abs() < 1e-9); // mean of {20,30,40}
        assert_eq!(age.count, 3);
    }

    #[test]
    fn flags_constant_and_categorical() {
        let p = Profile::of(&sample()).unwrap();
        assert!(p
            .alerts()
            .iter()
            .any(|a| a.suggested == "Drop" && a.column.as_deref() == Some("const")));
        assert!(p
            .alerts()
            .iter()
            .any(|a| a.suggested == "OneHotEncoder" && a.column.as_deref() == Some("city")));
    }

    #[test]
    fn suggests_impute_encode_scale() {
        let p = Profile::of(&sample()).unwrap();
        let pipe = p.suggest_pipeline();
        assert_eq!(pipe.step_names(), vec!["impute", "encode", "scale"]);
    }

    #[test]
    fn target_classification_class_balance() {
        let p = Profile::of_with_target(&sample(), "label").unwrap();
        match &p.target().unwrap().kind {
            TargetKind::Classification { classes } => {
                assert_eq!(classes.len(), 2);
                assert!(classes.iter().all(|(_, n)| *n == 2));
            }
            _ => panic!("expected classification target"),
        }
    }

    #[test]
    fn renders_html_report() {
        let p = Profile::of_with_target(&sample(), "label").unwrap();
        let html = p.render_html();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("Data profile"));
        assert!(html.contains("Alerts"));
    }

    #[test]
    fn computes_kurtosis_and_z_outliers() {
        let mut xs = vec![1.0_f64; 19];
        xs.push(100.0); // one extreme value
        let p = Profile::of(&Table::from_polars(df!("x" => xs).unwrap())).unwrap();
        let x = numeric(&p, "x");
        assert!(
            x.kurtosis.is_finite() && x.kurtosis > 3.0,
            "kurtosis {}",
            x.kurtosis
        );
        assert_eq!(x.outliers_z, 1, "the extreme value is a z-score outlier");
    }

    #[test]
    fn computes_spearman_correlation() {
        // y = x^2: perfectly monotonic (Spearman = 1) but nonlinear (Pearson < 1).
        let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let p = Profile::of(&Table::from_polars(df!("x" => x, "y" => y).unwrap())).unwrap();
        let c = p.correlations();
        let (i, j) = (
            c.columns.iter().position(|n| n == "x").unwrap(),
            c.columns.iter().position(|n| n == "y").unwrap(),
        );
        assert!(
            (c.spearman[i][j] - 1.0).abs() < 1e-9,
            "spearman {}",
            c.spearman[i][j]
        );
        assert!(c.matrix[i][j] < 0.99, "pearson {}", c.matrix[i][j]);
    }

    #[test]
    fn flags_co_missing_columns() {
        // `a` and `b` are null on exactly the same rows.
        let a = [Some(1.0_f64), None, Some(3.0), None, Some(5.0)];
        let b = [Some(1.0_f64), None, Some(3.0), None, Some(5.0)];
        let c = [Some(1.0_f64), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let p = Profile::of(&Table::from_polars(
            df!("a" => a, "b" => b, "c" => c).unwrap(),
        ))
        .unwrap();
        let cm = &p.missingness().co_missing;
        assert!(
            cm.iter()
                .any(|(x, y, phi)| (x == "a" && y == "b") && *phi > 0.9),
            "co_missing = {cm:?}"
        );
    }

    #[test]
    fn high_cardinality_suggests_target_encoder() {
        let ids: Vec<String> = (0..30).map(|i| format!("id{i}")).collect();
        let p = Profile::of(&Table::from_polars(df!("uid" => ids).unwrap())).unwrap();
        assert!(p
            .alerts()
            .iter()
            .any(|a| a.suggested == "TargetEncoder" && a.column.as_deref() == Some("uid")));
    }

    #[test]
    fn imbalance_raises_a_smote_alert() {
        let mut y = vec![0_i64; 9];
        y.extend([1, 1, 1]); // 9:3 = 3:1 imbalance
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let p = Profile::of_with_target(&Table::from_polars(df!("x" => x, "y" => y).unwrap()), "y")
            .unwrap();
        assert!(p.alerts().iter().any(|a| a.suggested == "Smote"));
        let _ = p.suggest_pipeline(); // builds without panicking
    }
}
