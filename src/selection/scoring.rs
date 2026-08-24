//! Scoring metrics, backed by `model-selection-rs`.

use ndarray::Array1;

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
            Metric::F1 => unreachable!("F1 is implemented locally"),
            Metric::Mae => Box::new(MeanAbsoluteError),
            Metric::Mse => Box::new(MeanSquaredError),
            Metric::Rmse => Box::new(RootMeanSquaredError),
            Metric::R2 => Box::new(R2Score),
        }
    }

    /// Whether a larger score is an improvement.
    pub fn greater_is_better(&self) -> bool {
        matches!(self, Metric::Accuracy | Metric::F1 | Metric::R2)
    }

    /// Score aligned truth / prediction vectors.
    pub fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        if matches!(self, Metric::F1) {
            let (mut tp, mut fp, mut false_negative) = (0.0, 0.0, 0.0);
            for (&truth, &prediction) in y_true.iter().zip(y_pred) {
                match (truth == 1.0, prediction == 1.0) {
                    (true, true) => tp += 1.0,
                    (false, true) => fp += 1.0,
                    (true, false) => false_negative += 1.0,
                    (false, false) => {}
                }
            }
            let denominator = 2.0 * tp + fp + false_negative;
            return if denominator == 0.0 {
                0.0
            } else {
                2.0 * tp / denominator
            };
        }
        let t = Array1::from(y_true.to_vec());
        let p = Array1::from(y_pred.to_vec());
        self.scorer().score(&t, &p)
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
        // Undefined F1 follows sklearn's `zero_division=0` convention.
        let all_negative = Metric::F1.score(&[0., 1., 0., 1.], &[0., 0., 0., 0.]);
        assert_eq!(
            all_negative, 0.0,
            "degenerate F1 should be 0.0, got {all_negative}"
        );
        let no_true_positive = Metric::F1.score(&[0., 1., 0., 1.], &[1., 0., 1., 0.]);
        assert_eq!(no_true_positive, 0.0);
    }
}
