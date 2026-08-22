//! A native binary logistic-regression classifier — the framework's first model
//! that produces *real* class probabilities.
//!
//! Unlike the smartcore `RandomForest` (whose per-tree internals aren't exposed,
//! so a soft vote can only average vote shares), [`LogisticRegression`]
//! implements [`ProbaPredictor`] with a genuine `sigmoid` probability. It is
//! pure core — no backend feature required — and fits by gradient descent on the
//! log-loss with optional L2, standardizing features internally so it converges
//! without a scaler in front.

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::traits::{Estimator, ParamValue, Predictor, ProbaPredictor};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Binary logistic regression: `P(y = c1 | x) = sigmoid(w·x + b)`.
///
/// The target must have exactly two classes; they are sorted ascending, and the
/// larger is the positive class. Features are standardized internally.
#[derive(Clone, Debug)]
pub struct LogisticRegression {
    learning_rate: f64,
    epochs: usize,
    l2: f64,
    weights: Vec<f64>,
    bias: f64,
    // internal standardization, learned at fit
    mean: Vec<f64>,
    std: Vec<f64>,
    classes: Vec<i64>,
    fitted: bool,
}

impl LogisticRegression {
    /// A classifier with sensible defaults (lr 0.5, 500 epochs, no L2).
    pub fn new() -> Self {
        LogisticRegression {
            learning_rate: 0.5,
            epochs: 500,
            l2: 0.0,
            weights: Vec::new(),
            bias: 0.0,
            mean: Vec::new(),
            std: Vec::new(),
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Set the gradient-descent learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of gradient-descent epochs.
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    /// Set the L2 regularization strength.
    pub fn l2(mut self, l2: f64) -> Self {
        self.l2 = l2;
        self
    }

    /// Standardize one row with the learned mean/std.
    fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .zip(&self.mean)
            .zip(&self.std)
            .map(|((x, m), s)| (x - m) / s)
            .collect()
    }

    /// Positive-class probability for one (raw) row.
    fn proba_pos(&self, row: &[f64]) -> f64 {
        let z: f64 = self
            .standardize(row)
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;
        sigmoid(z)
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        LogisticRegression::new()
    }
}

impl Estimator for LogisticRegression {
    fn name(&self) -> &'static str {
        "LogisticRegression"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let frame = dataset.features();
        let (n, p) = frame.shape();
        if n == 0 {
            return Err(Error::Shape("LogisticRegression: empty dataset".into()));
        }
        let mut classes: Vec<i64> = dataset.target().iter().map(|v| v.round() as i64).collect();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() != 2 {
            return Err(Error::Pipeline(format!(
                "LogisticRegression is binary; found {} classes",
                classes.len()
            )));
        }
        let pos = classes[1];
        let y: Vec<f64> = dataset
            .target()
            .iter()
            .map(|v| if v.round() as i64 == pos { 1.0 } else { 0.0 })
            .collect();

        // learn standardization
        let mut mean = vec![0.0; p];
        let mut std = vec![1.0; p];
        for c in 0..p {
            let col = frame.column(c);
            let m = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
            mean[c] = m;
            let sd = var.sqrt();
            std[c] = if sd > f64::EPSILON { sd } else { 1.0 };
        }
        self.mean = mean;
        self.std = std;

        // standardized design matrix
        let x: Vec<Vec<f64>> = (0..n).map(|r| self.standardize(frame.row(r))).collect();

        let mut w = vec![0.0; p];
        let mut b = 0.0;
        let inv_n = 1.0 / n as f64;
        for _ in 0..self.epochs {
            let mut gw = vec![0.0; p];
            let mut gb = 0.0;
            for (row, &yi) in x.iter().zip(&y) {
                let z: f64 = row.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
                let d = sigmoid(z) - yi;
                for (g, xi) in gw.iter_mut().zip(row) {
                    *g += d * xi;
                }
                gb += d;
            }
            for (wi, g) in w.iter_mut().zip(&gw) {
                *wi -= self.learning_rate * (g * inv_n + self.l2 * *wi);
            }
            b -= self.learning_rate * gb * inv_n;
        }

        self.weights = w;
        self.bias = b;
        self.classes = classes;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "learning_rate" => self.learning_rate = value.as_f64()?,
            "epochs" => self.epochs = value.as_i64()? as usize,
            "l2" => self.l2 = value.as_f64()?,
            other => {
                return Err(Error::Param(format!(
                    "LogisticRegression has no parameter '{other}'"
                )))
            }
        }
        Ok(())
    }
}

impl Predictor for LogisticRegression {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("LogisticRegression::predict".into()));
        }
        Ok((0..frame.nrows())
            .map(|r| {
                let c = if self.proba_pos(frame.row(r)) >= 0.5 {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
                c as f64
            })
            .collect())
    }
}

impl ProbaPredictor for LogisticRegression {
    fn predict_proba(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("LogisticRegression::predict_proba".into()));
        }
        let cols = vec![
            format!("p{}", self.classes[0]),
            format!("p{}", self.classes[1]),
        ];
        let n = frame.nrows();
        let mut buf = Vec::with_capacity(n * 2);
        for r in 0..n {
            let p = self.proba_pos(frame.row(r));
            buf.push(1.0 - p);
            buf.push(p);
        }
        Frame::new(buf, n, 2, cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separable() -> Dataset {
        // one feature that cleanly separates the two classes
        let rows: Vec<Vec<f64>> = [-3.0, -2.0, -1.5, 1.5, 2.0, 3.0]
            .iter()
            .map(|&x| vec![x])
            .collect();
        Dataset::new(
            Frame::from_rows(rows, vec!["x".into()]).unwrap(),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn separates_and_scores_probabilities() {
        let ds = separable();
        let mut lr = LogisticRegression::new();
        lr.fit(&ds).unwrap();

        let probe = Frame::from_rows(vec![vec![-2.5], vec![2.5]], vec!["x".into()]).unwrap();
        assert_eq!(lr.predict(&probe).unwrap(), vec![0.0, 1.0]);

        let proba = lr.predict_proba(&probe).unwrap();
        assert_eq!(proba.columns(), &["p0".to_string(), "p1".into()]);
        // rows sum to 1, and each is confident toward its class
        for r in 0..2 {
            assert!((proba.get(r, 0) + proba.get(r, 1) - 1.0).abs() < 1e-9);
        }
        assert!(proba.get(0, 0) > 0.5); // negative row -> class 0
        assert!(proba.get(1, 1) > 0.5); // positive row -> class 1
    }

    #[test]
    fn rejects_non_binary_target() {
        let x = Frame::from_rows(vec![vec![0.0], vec![1.0], vec![2.0]], vec!["x".into()]).unwrap();
        let ds = Dataset::new(x, vec![0.0, 1.0, 2.0]).unwrap();
        assert!(LogisticRegression::new().fit(&ds).is_err());
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(LogisticRegression::new().predict(&f).is_err());
    }
}
