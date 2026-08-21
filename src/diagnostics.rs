//! Regression diagnostics — OLS residual tests, influence, and `summary()`.
//!
//! Adapts [`regression-diagnostics`](https://docs.rs/regression-diagnostics)
//! behind the framework's [`Frame`]. [`Diagnostics::of`] fits an ordinary
//! least-squares model (with intercept) to a labelled dataset and exposes the
//! full statistical summary — R², per-coefficient VIF, Durbin–Watson, and the
//! Jarque–Bera / Breusch–Pagan / White tests — plus residuals and influence.
//!
//! Conversion to `ndarray` happens here, at the edge.

use ndarray::{Array1, Array2};
use regression_diagnostics::multicollinearity::vif;
use regression_diagnostics::{OlsFit, Summary};

use crate::error::{Error, Result};
use crate::frame::Dataset;

/// An OLS fit over a dataset, with its diagnostic summary.
pub struct Diagnostics {
    fit: OlsFit,
    columns: Vec<String>,
}

impl Diagnostics {
    /// Fit OLS (with an intercept) to `dataset` and prepare its diagnostics.
    pub fn of(dataset: &Dataset) -> Result<Diagnostics> {
        let (n, p) = dataset.features().shape();
        let x = Array2::from_shape_vec((n, p), dataset.features().buf().to_vec())
            .map_err(|e| Error::Backend(format!("ndarray conversion failed: {e}")))?;
        let y = Array1::from(dataset.target().to_vec());
        let fit = OlsFit::with_intercept(x, y)
            .map_err(|e| Error::Backend(format!("OLS fit failed: {e}")))?;
        Ok(Diagnostics {
            fit,
            columns: dataset.features().columns().to_vec(),
        })
    }

    /// The full statistical summary (implements `Display` for a printable table).
    pub fn summary(&self) -> Summary {
        self.fit.summary()
    }

    /// A formatted, human-readable summary table.
    pub fn summary_text(&self) -> String {
        self.fit.summary().to_string()
    }

    /// R² of the fit.
    pub fn r_squared(&self) -> f64 {
        self.fit.summary().r_squared
    }

    /// Per-column variance inflation factors, paired with column names.
    ///
    /// The underlying `vif` includes the intercept term first; it is dropped
    /// here so the values line up with the feature columns.
    pub fn vif(&self) -> Vec<(String, f64)> {
        let raw = vif(&self.fit);
        let feature_vifs = raw.iter().rev().take(self.columns.len()).rev();
        self.columns
            .iter()
            .cloned()
            .zip(feature_vifs.copied())
            .collect()
    }

    /// The residuals of the fit.
    pub fn residuals(&self) -> Vec<f64> {
        self.fit.residuals().to_vec()
    }

    /// The maximum Cook's distance — the most influential observation.
    pub fn max_cooks_distance(&self) -> f64 {
        Summary::max_cooks_distance(&self.fit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;

    #[test]
    fn recovers_a_linear_relationship() {
        // y = 2*x1 + 3 (x2 is noise-free but uncorrelated)
        let rows: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64, (i % 3) as f64])
            .collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();

        let diag = Diagnostics::of(&ds).unwrap();
        assert!(diag.r_squared() > 0.99, "R2 = {}", diag.r_squared());
        assert_eq!(diag.vif().len(), 2);
        assert_eq!(diag.residuals().len(), 20);
        assert!(diag.summary_text().contains("R"));
    }
}
