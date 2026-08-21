//! Evaluation reports — the metrics half of "trust the model, not just run it".
//!
//! [`Evaluate::evaluate`] runs a model over a labelled [`Dataset`] and bundles
//! the appropriate metrics into a [`Report`]. The task (classification vs.
//! regression) is inferred from the target: an all-integral target is treated
//! as class labels; anything else as regression.
//!
//! This is core (no extra dependencies). The richer, model-specific diagnostics
//! — OLS residual tests, SHAP, report figures — live in the `diagnostics`,
//! `explain`, and `viz` modules.

use std::collections::BTreeSet;
use std::fmt;

use crate::error::Result;
use crate::frame::Dataset;
use crate::traits::Predictor;

/// Whether a [`Report`] is over a classification or regression target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Classification,
    Regression,
}

/// A bundle of evaluation metrics for one model on one test set.
#[derive(Clone, Debug)]
pub struct Report {
    task: Task,
    n: usize,
    metrics: Vec<(String, f64)>,
}

impl Report {
    /// Build a report from aligned truth / prediction vectors, inferring the
    /// task from the target.
    pub fn new(y_true: &[f64], y_pred: &[f64]) -> Report {
        let task = if is_classification(y_true) {
            Task::Classification
        } else {
            Task::Regression
        };
        let metrics = match task {
            Task::Classification => classification_metrics(y_true, y_pred),
            Task::Regression => regression_metrics(y_true, y_pred),
        };
        Report {
            task,
            n: y_true.len(),
            metrics,
        }
    }

    /// The inferred task.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Every `(metric, value)` pair, in report order.
    pub fn metrics(&self) -> &[(String, f64)] {
        &self.metrics
    }

    /// Look up one metric by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?} report ({} rows)", self.task, self.n)?;
        for (name, value) in &self.metrics {
            writeln!(f, "  {name:<10} {value:.4}")?;
        }
        Ok(())
    }
}

fn is_classification(y: &[f64]) -> bool {
    !y.is_empty() && y.iter().all(|v| v.is_finite() && v.fract() == 0.0)
}

fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> Vec<(String, f64)> {
    let n = y_true.len() as f64;
    let mae = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n;
    let mse = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    };
    vec![
        ("mae".into(), mae),
        ("mse".into(), mse),
        ("rmse".into(), mse.sqrt()),
        ("r2".into(), r2),
    ]
}

fn classification_metrics(y_true: &[f64], y_pred: &[f64]) -> Vec<(String, f64)> {
    let n = y_true.len() as f64;
    let correct = y_true
        .iter()
        .zip(y_pred)
        .filter(|(t, p)| (**t - **p).abs() < f64::EPSILON)
        .count() as f64;
    let accuracy = correct / n;

    // Macro-averaged precision / recall / F1 over the observed classes.
    let classes: BTreeSet<i64> = y_true
        .iter()
        .chain(y_pred)
        .map(|v| v.round() as i64)
        .collect();
    let (mut prec_sum, mut rec_sum, mut f1_sum) = (0.0, 0.0, 0.0);
    for &c in &classes {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;
        for (t, p) in y_true.iter().zip(y_pred) {
            let (t, p) = (t.round() as i64, p.round() as i64);
            match (t == c, p == c) {
                (true, true) => tp += 1.0,
                (false, true) => fp += 1.0,
                (true, false) => fn_ += 1.0,
                (false, false) => {}
            }
        }
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        prec_sum += precision;
        rec_sum += recall;
        f1_sum += f1;
    }
    let k = classes.len().max(1) as f64;
    vec![
        ("accuracy".into(), accuracy),
        ("precision".into(), prec_sum / k),
        ("recall".into(), rec_sum / k),
        ("f1".into(), f1_sum / k),
    ]
}

/// Convenience: any [`Predictor`] can `evaluate` itself on a test set.
pub trait Evaluate: Predictor {
    /// Predict on `dataset` and score against its target.
    fn evaluate(&self, dataset: &Dataset) -> Result<Report> {
        let preds = self.predict(dataset.features())?;
        Ok(Report::new(dataset.target(), &preds))
    }
}

impl<T: Predictor + ?Sized> Evaluate for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_report_is_perfect_for_perfect_preds() {
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let r = Report::new(&y, &y);
        assert_eq!(r.task(), Task::Classification);
        assert_eq!(r.get("accuracy"), Some(1.0));
        assert_eq!(r.get("f1"), Some(1.0));
    }

    #[test]
    fn regression_report_computes_r2() {
        let t = vec![1.0, 2.0, 3.5, 4.0];
        let p = vec![1.1, 1.9, 3.4, 4.2];
        let r = Report::new(&t, &p);
        assert_eq!(r.task(), Task::Regression);
        assert!(r.get("r2").unwrap() > 0.98);
        assert!(r.get("rmse").unwrap() > 0.0);
    }

    #[test]
    fn half_right_classification_scores_half() {
        let t = vec![0.0, 0.0, 1.0, 1.0];
        let p = vec![0.0, 1.0, 0.0, 1.0];
        let r = Report::new(&t, &p);
        assert_eq!(r.get("accuracy"), Some(0.5));
    }
}
