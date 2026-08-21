//! Core transformers.
//!
//! Phase 0 ships exactly one, [`StandardScaler`], to prove the `transform`
//! half of the contract and give a pipeline something to compose in front of
//! its estimator. The full preprocessing suite (imputers, encoders, SMOTE)
//! arrives in Phase 1 behind the `preprocessing` feature; they will implement
//! the very same [`Transformer`] trait.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::traits::{ParamValue, Transformer};

/// Standardize columns to zero mean and unit variance: `(x - mean) / std`.
///
/// Fitting learns per-column mean and (population) standard deviation.
/// Columns with zero variance are left unscaled (divided by 1).
#[derive(Clone, Debug)]
pub struct StandardScaler {
    with_mean: bool,
    with_std: bool,
    means: Vec<f64>,
    stds: Vec<f64>,
    columns: Vec<String>,
    fitted: bool,
}

impl StandardScaler {
    /// A scaler that centers and scales.
    pub fn new() -> Self {
        StandardScaler {
            with_mean: true,
            with_std: true,
            means: Vec::new(),
            stds: Vec::new(),
            columns: Vec::new(),
            fitted: false,
        }
    }

    /// Center to zero mean but do not scale.
    pub fn with_mean_only() -> Self {
        StandardScaler {
            with_std: false,
            ..StandardScaler::new()
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        StandardScaler::new()
    }
}

impl Transformer for StandardScaler {
    fn name(&self) -> &'static str {
        "StandardScaler"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (n, p) = frame.shape();
        if n == 0 {
            return Err(Error::Shape("cannot fit StandardScaler on 0 rows".into()));
        }
        let mut means = vec![0.0; p];
        let mut stds = vec![1.0; p];
        for c in 0..p {
            let col = frame.column(c);
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            means[c] = if self.with_mean { mean } else { 0.0 };
            let sd = var.sqrt();
            stds[c] = if self.with_std && sd > f64::EPSILON {
                sd
            } else {
                1.0
            };
        }
        self.means = means;
        self.stds = stds;
        self.columns = frame.columns().to_vec();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("StandardScaler::transform".into()));
        }
        frame.require_columns(&self.columns)?;
        let (n, p) = frame.shape();
        let mut buf = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                buf.push((frame.get(r, c) - self.means[c]) / self.stds[c]);
            }
        }
        Frame::new(buf, n, p, self.columns.clone())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "with_mean" => self.with_mean = value.as_bool()?,
            "with_std" => self.with_std = value.as_bool()?,
            other => {
                return Err(Error::Param(format!(
                    "StandardScaler has no parameter '{other}'"
                )))
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardizes_to_zero_mean_unit_std() {
        let f = Frame::from_rows(
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec!["x".into()],
        )
        .unwrap();
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&f).unwrap();
        let col = out.column(0);
        let mean: f64 = col.iter().sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-9);
        // population std of the standardized column is 1
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0;
        assert!((var - 1.0).abs() < 1e-9);
    }

    #[test]
    fn transform_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(StandardScaler::new().transform(&f).is_err());
    }
}
