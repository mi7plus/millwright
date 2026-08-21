//! Train-time balancers — the resampling stage of a pipeline.
//!
//! These adapt [`imbalance-rs`](https://docs.rs/imbalance-rs) samplers behind
//! the framework's [`Balancer`] trait. A balancer runs **only during `fit`**
//! (via [`Pipeline::balance`](crate::pipeline::Pipeline::balance)) — never at
//! predict time — because resampling changes the row set.
//!
//! Conversion to `ndarray` happens here, at the edge, so the imbalance-rs
//! array world never reaches user code.

use imbalance_rs::{RandomOverSampler as ImbRos, Sampler, Smote as ImbSmote};
use ndarray::{Array1, Array2};

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::traits::Balancer;

fn frame_to_array2(frame: &Frame) -> Result<Array2<f64>> {
    let (n, p) = frame.shape();
    Array2::from_shape_vec((n, p), frame.buf().to_vec())
        .map_err(|e| Error::Backend(format!("ndarray conversion failed: {e}")))
}

fn array2_to_frame(arr: &Array2<f64>, columns: &[String]) -> Result<Frame> {
    let (n, p) = arr.dim();
    // `.iter()` yields elements in row-major logical order, which is exactly
    // the layout `Frame` stores.
    let buf: Vec<f64> = arr.iter().copied().collect();
    Frame::new(buf, n, p, columns.to_vec())
}

/// SMOTE over-sampling (Synthetic Minority Over-sampling Technique).
///
/// Synthesises new minority-class rows by interpolating between a sample and its
/// nearest same-class neighbours. Backed by `imbalance_rs::Smote`.
#[derive(Clone, Debug)]
pub struct Smote {
    k_neighbors: usize,
    seed: u64,
}

impl Smote {
    /// SMOTE with the default 5 neighbours.
    pub fn new() -> Self {
        Smote {
            k_neighbors: 5,
            seed: 0,
        }
    }

    /// Number of nearest neighbours used to interpolate.
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Seed the RNG for reproducible synthesis.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for Smote {
    fn default() -> Self {
        Smote::new()
    }
}

impl Balancer for Smote {
    fn name(&self) -> &'static str {
        "Smote"
    }

    fn fit_resample(&self, features: &Frame, target: &[f64]) -> Result<(Frame, Vec<f64>)> {
        let x = frame_to_array2(features)?;
        let y: Array1<i64> =
            Array1::from(target.iter().map(|v| v.round() as i64).collect::<Vec<_>>());
        let sampler = ImbSmote::new()
            .k_neighbors(self.k_neighbors)
            .random_state(self.seed);
        let (xr, yr) = sampler
            .fit_resample(&x, &y)
            .map_err(|e| Error::Backend(format!("SMOTE failed: {e}")))?;
        let frame = array2_to_frame(&xr, features.columns())?;
        let target = yr.iter().map(|&l| l as f64).collect();
        Ok((frame, target))
    }
}

/// Random over-sampling: duplicate minority-class rows until balanced.
///
/// Backed by `imbalance_rs::RandomOverSampler`.
#[derive(Clone, Debug, Default)]
pub struct RandomOverSampler;

impl RandomOverSampler {
    pub fn new() -> Self {
        RandomOverSampler
    }
}

impl Balancer for RandomOverSampler {
    fn name(&self) -> &'static str {
        "RandomOverSampler"
    }

    fn fit_resample(&self, features: &Frame, target: &[f64]) -> Result<(Frame, Vec<f64>)> {
        let x = frame_to_array2(features)?;
        let y: Array1<i64> =
            Array1::from(target.iter().map(|v| v.round() as i64).collect::<Vec<_>>());
        let (xr, yr) = ImbRos::new()
            .fit_resample(&x, &y)
            .map_err(|e| Error::Backend(format!("RandomOverSampler failed: {e}")))?;
        let frame = array2_to_frame(&xr, features.columns())?;
        let target = yr.iter().map(|&l| l as f64).collect();
        Ok((frame, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smote_balances_the_minority_class() {
        // 6 majority (class 0), 2 minority (class 1).
        let x = Frame::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.1, 0.2],
                vec![0.2, 0.1],
                vec![0.3, 0.0],
                vec![0.0, 0.3],
                vec![0.2, 0.2],
                vec![9.0, 9.0],
                vec![9.1, 9.2],
            ],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let (xr, yr) = Smote::new()
            .k_neighbors(1)
            .random_state(7)
            .fit_resample(&x, &y)
            .unwrap();
        let zeros = yr.iter().filter(|&&v| v == 0.0).count();
        let ones = yr.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(zeros, 6);
        assert_eq!(ones, 6, "minority class should be oversampled to parity");
        assert_eq!(xr.nrows(), yr.len());
        assert_eq!(xr.ncols(), 2);
    }
}
