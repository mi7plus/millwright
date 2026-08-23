//! Drift monitoring — PSI on the prediction stream.
//!
//! [`DriftMonitor::psi`] builds a monitor from a reference (training)
//! prediction distribution via [`driftwatch`](https://docs.rs/driftwatch). As a
//! served model handles traffic, [`DriftMonitor::observe`] accumulates live
//! predictions; [`DriftMonitor::report`] scores them against the reference and
//! reports the population-stability index and whether it has drifted.

use std::sync::Mutex;

use driftwatch::{EqualWidthBinning, MetricKind, PredictionDriftMonitor};

use crate::error::{Error, Result};

fn drift_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("driftwatch: {e}"))
}

/// A snapshot of drift status for the observed prediction stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct DriftStatus {
    /// Whether the live stream has drifted from the reference.
    pub drifted: bool,
    /// Population-stability index of the live predictions vs. the reference.
    pub psi: f64,
    /// Number of live predictions observed so far.
    pub observed: usize,
}

/// A PSI drift monitor over a model's prediction stream.
pub struct DriftMonitor {
    inner: PredictionDriftMonitor,
    live: Mutex<Vec<f64>>,
}

impl DriftMonitor {
    /// Build a monitor from the reference (training) predictions, binned with
    /// the standard 10-bin PSI convention.
    pub fn psi(reference_predictions: &[f64]) -> Result<Self> {
        let inner =
            PredictionDriftMonitor::new(reference_predictions, EqualWidthBinning::default())
                .map_err(drift_err)?;
        Ok(DriftMonitor {
            inner,
            live: Mutex::new(Vec::new()),
        })
    }

    /// Build a monitor from the reference distribution a registry [`Version`]
    /// stored at registration time.
    ///
    /// [`Version`]: crate::registry::Version
    #[cfg(feature = "registry")]
    pub fn from_registry(version: &crate::registry::Version) -> Result<Self> {
        Self::psi(&version.metadata.reference)
    }

    /// Record a batch of live predictions.
    pub fn observe(&self, predictions: &[f64]) {
        self.live
            .lock()
            .expect("monitor lock")
            .extend_from_slice(predictions);
    }

    /// Score the accumulated live predictions against the reference.
    pub fn report(&self) -> Result<DriftStatus> {
        let live = self.live.lock().expect("monitor lock").clone();
        if live.is_empty() {
            return Ok(DriftStatus::default());
        }
        let report = self.inner.check(&live).map_err(drift_err)?;
        let psi = report
            .features
            .first()
            .and_then(|f| f.score(MetricKind::Psi))
            .map(|s| s.statistic)
            .unwrap_or(0.0);
        let drifted = report.drifted_features().next().is_some();
        Ok(DriftStatus {
            drifted,
            psi,
            observed: live.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_stream_does_not_drift_but_shifted_stream_does() {
        // reference predictions centred near 0
        let reference: Vec<f64> = (0..200).map(|i| (i % 10) as f64 * 0.1).collect();
        let monitor = DriftMonitor::psi(&reference).unwrap();

        // a like-distributed live batch: low PSI
        let stable: Vec<f64> = (0..200).map(|i| (i % 10) as f64 * 0.1).collect();
        monitor.observe(&stable);
        let stable_report = monitor.report().unwrap();
        assert!(!stable_report.drifted, "psi = {}", stable_report.psi);

        // a clearly shifted stream: high PSI, drift fires
        let shifted = DriftMonitor::psi(&reference).unwrap();
        let far: Vec<f64> = (0..200).map(|_| 100.0).collect();
        shifted.observe(&far);
        let shifted_report = shifted.report().unwrap();
        assert!(shifted_report.drifted, "psi = {}", shifted_report.psi);
        assert!(shifted_report.psi > stable_report.psi);
    }
}
