//! Time-series forecasting — auto-ARIMA via chronos-ts.
//!
//! [`AutoArima`] adapts [`chronos-ts`](https://docs.rs/chronos-ts)'s `auto_arima`
//! behind the framework's [`Forecaster`] contract. chronos-ts speaks
//! `ndarray 0.15` (distinct from the `0.16` the rest of the stack uses); the
//! conversion happens here, at the edge.

use chronos_ts::{auto_arima, AutoArimaOptions, SarimaModel};
use ndarray015::Array1;

use crate::error::{Error, Result};
use crate::traits::Forecaster;

/// An auto-ARIMA forecaster: searches `(p, d, q)` orders and fits the best.
pub struct AutoArima {
    max_p: usize,
    max_q: usize,
    max_d: usize,
    seasonal_period: usize,
    model: Option<SarimaModel>,
    history: Vec<f64>,
}

impl AutoArima {
    /// A non-seasonal auto-ARIMA searching up to order 5.
    pub fn new() -> Self {
        AutoArima {
            max_p: 5,
            max_q: 5,
            max_d: 2,
            seasonal_period: 1,
            model: None,
            history: Vec::new(),
        }
    }

    /// Cap the autoregressive order searched.
    pub fn max_p(mut self, p: usize) -> Self {
        self.max_p = p;
        self
    }

    /// Cap the moving-average order searched.
    pub fn max_q(mut self, q: usize) -> Self {
        self.max_q = q;
        self
    }

    /// Set the seasonal period `m` (1 for non-seasonal).
    pub fn seasonal_period(mut self, m: usize) -> Self {
        self.seasonal_period = m;
        self
    }
}

impl Default for AutoArima {
    fn default() -> Self {
        AutoArima::new()
    }
}

impl Forecaster for AutoArima {
    fn name(&self) -> &'static str {
        "AutoArima"
    }

    fn fit(&mut self, series: &[f64]) -> Result<()> {
        if series.len() < 3 {
            return Err(Error::Shape(
                "AutoArima needs at least 3 observations".into(),
            ));
        }
        let data = Array1::from(series.to_vec());
        let opts = AutoArimaOptions {
            max_p: self.max_p,
            max_q: self.max_q,
            max_d: self.max_d,
            m: self.seasonal_period,
            ..Default::default()
        };
        let model =
            auto_arima(&data, opts).map_err(|e| Error::Backend(format!("auto_arima: {e}")))?;
        self.model = Some(model);
        self.history = series.to_vec();
        Ok(())
    }

    fn forecast(&self, steps: usize) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("AutoArima::forecast".into()))?;
        let history = Array1::from(self.history.clone());
        Ok(model.forecast(&history, steps).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecasts_a_linear_trend_upward() {
        // A clear upward trend; the forecast should continue rising.
        let series: Vec<f64> = (0..40)
            .map(|i| 10.0 + i as f64 + (i % 3) as f64 * 0.3)
            .collect();
        let mut model = AutoArima::new().max_p(2).max_q(2);
        model.fit(&series).unwrap();
        let fc = model.forecast(4).unwrap();
        assert_eq!(fc.len(), 4);
        // continuation of an upward trend stays above the last observed value's
        // neighbourhood and keeps increasing
        assert!(fc[0] > series[series.len() - 1] - 3.0);
        assert!(fc[3] > fc[0]);
    }

    #[test]
    fn forecast_before_fit_errors() {
        assert!(AutoArima::new().forecast(3).is_err());
    }
}
