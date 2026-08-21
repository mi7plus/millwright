//! Explainability — SHAP values and permutation importance.
//!
//! SHAP uses [`shap-rs`](https://docs.rs/shap-rs)'s model-agnostic KernelSHAP:
//! any [`Predictor`] is wrapped as the prediction function, so a fitted
//! pipeline, ensemble, or bare model all explain the same way. Permutation
//! importance is computed directly over the same contract.
//!
//! ```no_run
//! use millwright::prelude::*;
//! # fn main() -> millwright::Result<()> {
//! # let model: RandomForest = todo!();
//! # let test: Frame = todo!();
//! let explanation = model.explain(&Explainer::kernel(), &test)?;
//! for (feature, importance) in explanation.importance() {
//!     println!("{feature}: {importance:.3}");
//! }
//! # Ok(())
//! # }
//! ```

use shap_rs::explain_sample;

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::rng::Rng;
use crate::traits::Predictor;

/// Configuration for a SHAP explainer.
#[derive(Clone, Copy, Debug)]
pub struct Explainer {
    nsamples: usize,
    background: usize,
}

impl Explainer {
    /// A KernelSHAP explainer (100 coalition samples, up to 50 background rows).
    pub fn kernel() -> Self {
        Explainer {
            nsamples: 100,
            background: 50,
        }
    }

    /// Number of coalition samples per explained row (higher = more accurate,
    /// slower).
    pub fn nsamples(mut self, n: usize) -> Self {
        self.nsamples = n;
        self
    }

    /// Maximum number of rows used as the reference background distribution.
    pub fn background(mut self, n: usize) -> Self {
        self.background = n;
        self
    }
}

/// Per-row SHAP values: a [`Frame`] shaped like the explained data, where each
/// cell is a feature's contribution to that row's prediction.
pub struct Explanation {
    values: Frame,
}

impl Explanation {
    /// The raw per-row, per-feature SHAP values.
    pub fn values(&self) -> &Frame {
        &self.values
    }

    /// Global importance: mean absolute SHAP value per feature, descending.
    pub fn importance(&self) -> Vec<(String, f64)> {
        let f = &self.values;
        let n = f.nrows().max(1) as f64;
        let mut out: Vec<(String, f64)> = f
            .columns()
            .iter()
            .enumerate()
            .map(|(c, name)| {
                let mean_abs = f.column(c).iter().map(|v| v.abs()).sum::<f64>() / n;
                (name.clone(), mean_abs)
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// Any [`Predictor`] can be explained.
pub trait Explain: Predictor {
    /// Compute SHAP values for every row of `frame`.
    fn explain(&self, explainer: &Explainer, frame: &Frame) -> Result<Explanation> {
        let cols = frame.columns().to_vec();
        let rows = frame.as_rows();
        let background: Vec<Vec<f64>> = rows.iter().take(explainer.background).cloned().collect();
        if background.is_empty() {
            return Err(Error::Shape("explain: empty background".into()));
        }

        // Wrap this model as SHAP's prediction function.
        let predict = |batch: &[Vec<f64>]| -> Vec<f64> {
            match Frame::from_rows(batch.to_vec(), cols.clone()) {
                Ok(fr) => self.predict(&fr).unwrap_or_else(|_| vec![0.0; batch.len()]),
                Err(_) => vec![0.0; batch.len()],
            }
        };

        let p = frame.ncols();
        let mut buf = Vec::with_capacity(frame.nrows() * p);
        let predict_ref = &predict;
        for row in &rows {
            let sv = explain_sample(predict_ref, row, &background, explainer.nsamples)
                .map_err(|e| Error::Backend(format!("SHAP failed: {e}")))?;
            buf.extend(sv);
        }
        Ok(Explanation {
            values: Frame::new(buf, frame.nrows(), p, cols)?,
        })
    }
}

impl<T: Predictor + ?Sized> Explain for T {}

/// Permutation importance: how much a model's error grows when each feature is
/// randomly shuffled, averaged over `n_repeats`. Larger means more important.
///
/// Error is misclassification rate for a classification target (integral) and
/// mean squared error otherwise.
pub fn permutation_importance(
    model: &dyn Predictor,
    dataset: &Dataset,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<(String, f64)>> {
    let base = model.predict(dataset.features())?;
    let base_err = error(dataset.target(), &base);
    let mut rng = Rng::new(seed);
    let cols = dataset.features().columns().to_vec();

    let mut out = Vec::with_capacity(cols.len());
    for (c, name) in cols.iter().enumerate() {
        let mut delta = 0.0;
        for _ in 0..n_repeats.max(1) {
            let permuted = permute_column(dataset.features(), c, &mut rng)?;
            let preds = model.predict(&permuted)?;
            delta += error(dataset.target(), &preds) - base_err;
        }
        out.push((name.clone(), delta / n_repeats.max(1) as f64));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

fn permute_column(frame: &Frame, c: usize, rng: &mut Rng) -> Result<Frame> {
    let (n, p) = frame.shape();
    let mut col = frame.column(c);
    rng.shuffle(&mut col);
    let mut buf = frame.buf().to_vec();
    for (r, v) in col.into_iter().enumerate() {
        buf[r * p + c] = v;
    }
    Frame::new(buf, n, p, frame.columns().to_vec())
}

fn error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().max(1) as f64;
    if !y_true.is_empty() && y_true.iter().all(|v| v.is_finite() && v.fract() == 0.0) {
        // classification: misclassification rate
        let wrong = y_true
            .iter()
            .zip(y_pred)
            .filter(|(t, p)| (**t - **p).abs() >= f64::EPSILON)
            .count() as f64;
        wrong / n
    } else {
        y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / n
    }
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;
    use crate::traits::Estimator;

    fn trained() -> (RandomForest, Dataset) {
        // Class depends only on the first feature; the second is noise.
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            rows.push(vec![0.0 + i as f64 * 0.05, (i % 5) as f64]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.05, (i % 5) as f64]);
            y.push(1.0);
        }
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["signal".into(), "noise".into()]).unwrap(),
            y,
        )
        .unwrap();
        let mut rf = RandomForest::new().n_trees(30);
        rf.fit(&ds).unwrap();
        (rf, ds)
    }

    #[test]
    fn shap_ranks_the_signal_feature_first() {
        let (rf, ds) = trained();
        let exp = rf
            .explain(&Explainer::kernel().nsamples(64), ds.features())
            .unwrap();
        assert_eq!(exp.values().shape(), ds.features().shape());
        let importance = exp.importance();
        assert_eq!(importance[0].0, "signal");
    }

    #[test]
    fn permutation_importance_ranks_the_signal_feature_first() {
        let (rf, ds) = trained();
        let imp = permutation_importance(&rf, &ds, 5, 0).unwrap();
        assert_eq!(imp[0].0, "signal");
        assert!(imp[0].1 >= imp[1].1);
    }
}
