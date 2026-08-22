//! Scoring metrics, backed by `model-selection-rs`.

use ndarray::Array1;

use model_selection_rs::scoring::smartcore_adapter::SmartcoreF1;
use model_selection_rs::scoring::{
    Accuracy as MsAccuracy, MeanAbsoluteError, MeanSquaredError, R2Score, RootMeanSquaredError,
    Scorer,
};

/// A scoring metric, backed by `model-selection-rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    /// Classification accuracy (higher is better).
    Accuracy,
    /// Binary F1 (higher is better); labels encoded `0.0` / `1.0`.
    F1,
    /// Mean absolute error (lower is better).
    Mae,
    /// Mean squared error (lower is better).
    Mse,
    /// Root mean squared error (lower is better).
    Rmse,
    /// Coefficient of determination R² (higher is better).
    R2,
}

impl Metric {
    fn scorer(&self) -> Box<dyn Scorer> {
        match self {
            Metric::Accuracy => Box::new(MsAccuracy),
            Metric::F1 => Box::new(SmartcoreF1::default()),
            Metric::Mae => Box::new(MeanAbsoluteError),
            Metric::Mse => Box::new(MeanSquaredError),
            Metric::Rmse => Box::new(RootMeanSquaredError),
            Metric::R2 => Box::new(R2Score),
        }
    }

    /// Whether a larger score is an improvement.
    pub fn greater_is_better(&self) -> bool {
        self.scorer().greater_is_better()
    }

    /// Score aligned truth / prediction vectors.
    pub fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let t = Array1::from(y_true.to_vec());
        let p = Array1::from(y_pred.to_vec());
        let s = self.scorer().score(&t, &p);
        // A degenerate fold (e.g. an all-negative prediction) leaves smartcore's
        // F1 evaluating precision·recall / (precision + recall) = 0/0 = NaN. A
        // single NaN fold poisons the cross-validated mean and the search's
        // best-score comparison. Follow scikit-learn's convention (`zero_division=0`):
        // an undefined F1 is 0.0, keeping CV scores finite and comparable.
        if matches!(self, Metric::F1) && !s.is_finite() {
            return 0.0;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1_is_well_defined_on_normal_predictions() {
        assert!((Metric::F1.score(&[0., 1., 0., 1.], &[0., 1., 0., 1.]) - 1.0).abs() < 1e-9);
        let realistic = Metric::F1.score(&[0., 0., 1., 1., 0., 1.], &[0., 1., 1., 0., 0., 1.]);
        assert!(realistic.is_finite() && realistic > 0.0);
    }

    #[test]
    fn f1_of_a_degenerate_fold_is_zero_not_nan() {
        // No true positives (all-negative prediction) -> precision + recall = 0.
        // smartcore returns 0/0 = NaN; we clamp to sklearn's 0.0.
        let all_negative = Metric::F1.score(&[0., 1., 0., 1.], &[0., 0., 0., 0.]);
        assert_eq!(
            all_negative, 0.0,
            "degenerate F1 should be 0.0, got {all_negative}"
        );
        let no_true_positive = Metric::F1.score(&[0., 1., 0., 1.], &[1., 0., 1., 0.]);
        assert_eq!(no_true_positive, 0.0);
    }
}
