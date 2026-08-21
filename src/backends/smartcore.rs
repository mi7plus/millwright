//! The smartcore backend — Millwright's first engine.
//!
//! Wraps a couple of smartcore models behind the framework's traits. The
//! `Frame -> DenseMatrix` conversion happens here, at the edge, via
//! [`as_dense`]; nothing above this module ever names a smartcore type.
//!
//! Phase 0 adapts two models to prove the contract across a real backend:
//! - [`RandomForest`] — the classifier from the design brief's API example.
//! - [`LinearRegression`] — the regression path.

use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression as ScLinearRegression, LinearRegressionParameters,
};

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::traits::{Estimator, ParamValue, Predictor};

/// Convert a [`Frame`] to smartcore's native `DenseMatrix<f64>`.
///
/// This is the one place the boundary type crosses into an engine's world.
pub fn as_dense(frame: &Frame) -> Result<DenseMatrix<f64>> {
    DenseMatrix::from_2d_vec(&frame.as_rows())
        .map_err(|e| Error::Backend(format!("DenseMatrix conversion failed: {e}")))
}

/// A random-forest classifier backed by smartcore.
///
/// Class labels are the (integral) values of the [`Dataset`] target, cast to
/// integers for smartcore and back to `f64` on predict.
pub struct RandomForest {
    n_trees: u16,
    max_depth: Option<u16>,
    model: Option<RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>>>,
}

impl RandomForest {
    /// A forest with smartcore's defaults (100 trees, unbounded depth).
    pub fn new() -> Self {
        RandomForest {
            n_trees: 100,
            max_depth: None,
            model: None,
        }
    }

    /// Set the number of trees.
    pub fn n_trees(mut self, n: u16) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the maximum tree depth.
    pub fn max_depth(mut self, d: u16) -> Self {
        self.max_depth = Some(d);
        self
    }
}

impl Default for RandomForest {
    fn default() -> Self {
        RandomForest::new()
    }
}

impl Estimator for RandomForest {
    fn name(&self) -> &'static str {
        "RandomForest"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y: Vec<i64> = dataset.target().iter().map(|v| v.round() as i64).collect();

        let mut params = RandomForestClassifierParameters::default().with_n_trees(self.n_trees);
        if let Some(d) = self.max_depth {
            params = params.with_max_depth(d);
        }

        let model = RandomForestClassifier::fit(&x, &y, params)
            .map_err(|e| Error::Backend(format!("RandomForest fit failed: {e}")))?;
        self.model = Some(model);
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_trees" => self.n_trees = value.as_i64()? as u16,
            "max_depth" => self.max_depth = Some(value.as_i64()? as u16),
            other => {
                return Err(Error::Param(format!(
                    "RandomForest has no parameter '{other}'"
                )))
            }
        }
        Ok(())
    }
}

impl Predictor for RandomForest {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("RandomForest::predict".into()))?;
        let x = as_dense(frame)?;
        let y = model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("RandomForest predict failed: {e}")))?;
        Ok(y.into_iter().map(|c| c as f64).collect())
    }
}

/// Ordinary least squares, backed by smartcore.
pub struct LinearRegression {
    model: Option<ScLinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { model: None }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::new()
    }
}

impl Estimator for LinearRegression {
    fn name(&self) -> &'static str {
        "LinearRegression"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y: Vec<f64> = dataset.target().to_vec();
        let model = ScLinearRegression::fit(&x, &y, LinearRegressionParameters::default())
            .map_err(|e| Error::Backend(format!("LinearRegression fit failed: {e}")))?;
        self.model = Some(model);
        Ok(())
    }
}

impl Predictor for LinearRegression {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("LinearRegression::predict".into()))?;
        let x = as_dense(frame)?;
        model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("LinearRegression predict failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_forest_separates_two_clusters() {
        // Two well-separated classes on a single feature.
        let x = Frame::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.5, 0.2],
                vec![0.1, 0.4],
                vec![9.0, 9.0],
                vec![9.5, 8.8],
                vec![8.9, 9.3],
            ],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let ds = Dataset::new(x.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut rf = RandomForest::new().n_trees(50);
        rf.fit(&ds).unwrap();

        let test = Frame::from_rows(
            vec![vec![0.2, 0.1], vec![9.2, 9.1]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(rf.predict(&test).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn linear_regression_recovers_a_line() {
        // y = 2x + 1
        let x = Frame::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec!["x".into()],
        )
        .unwrap();
        let ds = Dataset::new(x, vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();

        let test = Frame::from_rows(vec![vec![4.0]], vec!["x".into()]).unwrap();
        let pred = lr.predict(&test).unwrap()[0];
        assert!((pred - 9.0).abs() < 1e-6, "expected ~9.0, got {pred}");
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(RandomForest::new().predict(&f).is_err());
        assert!(LinearRegression::new().predict(&f).is_err());
    }
}
