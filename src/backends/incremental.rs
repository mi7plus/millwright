//! Out-of-core learning — `partial_fit` via incremental-rs.
//!
//! [`IncrementalLinear`] adapts
//! [`incremental-rs`](https://docs.rs/incremental-rs)'s SGD linear regression
//! behind the framework's [`PartialFit`] + [`Predictor`] contracts, so a model
//! can be trained batch-by-batch over data that never fully loads into memory.
//! incremental-rs speaks `ndarray 0.15`; the conversion happens here, at the
//! edge.

use incremental_rs::{IncrementalLinearRegression, IncrementalSupervisedEstimator, LearningRateSchedule};
use ndarray015::{Array1, Array2};

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::traits::{PartialFit, Predictor};

fn to_array2(frame: &Frame) -> Result<Array2<f64>> {
    let (n, p) = frame.shape();
    Array2::from_shape_vec((n, p), frame.buf().to_vec())
        .map_err(|e| Error::Backend(format!("ndarray conversion failed: {e}")))
}

/// An incremental (SGD) linear regressor, updated one batch at a time.
pub struct IncrementalLinear {
    inner: IncrementalLinearRegression,
    fitted: bool,
}

impl IncrementalLinear {
    /// A regressor with a constant learning rate and no L2 penalty.
    pub fn new() -> Self {
        IncrementalLinear::with_rate(0.01, 0.0)
    }

    /// A regressor with an explicit constant learning rate and L2 penalty.
    pub fn with_rate(learning_rate: f64, l2_penalty: f64) -> Self {
        IncrementalLinear {
            inner: IncrementalLinearRegression::new(
                LearningRateSchedule::Constant { initial_rate: learning_rate },
                l2_penalty,
            ),
            fitted: false,
        }
    }
}

impl Default for IncrementalLinear {
    fn default() -> Self {
        IncrementalLinear::new()
    }
}

impl PartialFit for IncrementalLinear {
    fn name(&self) -> &'static str {
        "IncrementalLinear"
    }

    fn partial_fit(&mut self, batch: &Dataset) -> Result<()> {
        let x = to_array2(batch.features())?;
        let y = Array1::from(batch.target().to_vec());
        self.inner
            .partial_fit(&x, &y)
            .map_err(|e| Error::Backend(format!("partial_fit: {e}")))?;
        self.fitted = true;
        Ok(())
    }
}

impl Predictor for IncrementalLinear {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("IncrementalLinear::predict".into()));
        }
        let x = to_array2(frame)?;
        let y = self
            .inner
            .predict(&x)
            .map_err(|e| Error::Backend(format!("incremental predict: {e}")))?;
        Ok(y.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_a_line_over_streamed_batches() {
        // y = 3*x + 2, streamed in small batches.
        let mut model = IncrementalLinear::with_rate(0.05, 0.0);
        for epoch in 0..200 {
            let start = (epoch % 5) as f64;
            let rows: Vec<Vec<f64>> = (0..4).map(|i| vec![start + i as f64 * 0.25]).collect();
            let y: Vec<f64> = rows.iter().map(|r| 3.0 * r[0] + 2.0).collect();
            let batch = Dataset::new(Frame::from_rows(rows, vec!["x".into()]).unwrap(), y).unwrap();
            model.partial_fit(&batch).unwrap();
        }
        let probe = Frame::from_rows(vec![vec![2.0]], vec!["x".into()]).unwrap();
        let pred = model.predict(&probe).unwrap()[0];
        assert!((pred - 8.0).abs() < 1.0, "expected ~8.0, got {pred}");
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(IncrementalLinear::new().predict(&f).is_err());
    }
}
