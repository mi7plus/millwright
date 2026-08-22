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
        self.scorer().score(&t, &p)
    }
}
