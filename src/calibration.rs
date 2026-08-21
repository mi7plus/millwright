//! Probability calibration — turn raw classifier scores into calibrated
//! probabilities, and measure how well-calibrated they already are.
//!
//! [`PlattScaling`] fits a logistic `sigmoid(a·s + b)`; [`IsotonicRegression`]
//! fits a monotone step function via pool-adjacent-violators (PAV).
//! [`reliability_curve`] bins predictions against outcomes for a reliability
//! diagram. All three operate on `(score, label)` vectors, independent of any
//! model — so they calibrate a soft-vote, a decision function, or any score.

use crate::error::{Error, Result};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn check(scores: &[f64], labels: &[f64]) -> Result<()> {
    if scores.is_empty() || scores.len() != labels.len() {
        return Err(Error::Shape(format!(
            "calibration: {} scores vs {} labels",
            scores.len(),
            labels.len()
        )));
    }
    Ok(())
}

/// Platt scaling: a logistic map `p = sigmoid(a·s + b)` fit to binary outcomes.
#[derive(Clone, Debug)]
pub struct PlattScaling {
    a: f64,
    b: f64,
}

impl PlattScaling {
    /// Fit `a`, `b` to binary `labels` (0/1) from `scores` by gradient descent
    /// on the log-loss.
    pub fn fit(scores: &[f64], labels: &[f64]) -> Result<PlattScaling> {
        check(scores, labels)?;
        let n = scores.len() as f64;
        let (mut a, mut b) = (1.0, 0.0);
        let lr = 0.1;
        for _ in 0..3000 {
            let (mut ga, mut gb) = (0.0, 0.0);
            for (s, y) in scores.iter().zip(labels) {
                let d = sigmoid(a * s + b) - y; // d(logloss)/d(a·s+b)
                ga += d * s;
                gb += d;
            }
            a -= lr * ga / n;
            b -= lr * gb / n;
        }
        Ok(PlattScaling { a, b })
    }

    /// Map raw scores to calibrated probabilities.
    pub fn transform(&self, scores: &[f64]) -> Vec<f64> {
        scores
            .iter()
            .map(|s| sigmoid(self.a * s + self.b))
            .collect()
    }
}

/// Isotonic regression: a non-decreasing step function fit by
/// pool-adjacent-violators — a non-parametric calibrator.
#[derive(Clone, Debug)]
pub struct IsotonicRegression {
    /// Right edge (score) of each pooled block, ascending.
    edges: Vec<f64>,
    /// Fitted probability of each block.
    values: Vec<f64>,
}

impl IsotonicRegression {
    /// Fit a monotone map from `scores` to binary `labels`.
    pub fn fit(scores: &[f64], labels: &[f64]) -> Result<IsotonicRegression> {
        check(scores, labels)?;
        let mut pairs: Vec<(f64, f64)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let (mut edges, mut values, mut weights): (Vec<f64>, Vec<f64>, Vec<f64>) =
            (Vec::new(), Vec::new(), Vec::new());
        for (s, y) in pairs {
            edges.push(s);
            values.push(y);
            weights.push(1.0);
            // pool while the sequence violates monotonicity
            while values.len() >= 2 && values[values.len() - 2] >= values[values.len() - 1] {
                let n = values.len();
                let w = weights[n - 2] + weights[n - 1];
                let v = (values[n - 2] * weights[n - 2] + values[n - 1] * weights[n - 1]) / w;
                let e = edges[n - 1];
                values.truncate(n - 2);
                weights.truncate(n - 2);
                edges.truncate(n - 2);
                values.push(v);
                weights.push(w);
                edges.push(e);
            }
        }
        Ok(IsotonicRegression { edges, values })
    }

    /// Map raw scores to calibrated probabilities via the step function.
    pub fn transform(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.interp(s)).collect()
    }

    fn interp(&self, s: f64) -> f64 {
        match self.edges.iter().rposition(|&e| e <= s) {
            Some(i) => self.values[i],
            None => self.values.first().copied().unwrap_or(0.0),
        }
    }
}

/// One bin of a reliability diagram.
#[derive(Clone, Copy, Debug)]
pub struct ReliabilityBin {
    /// Mean predicted probability in the bin.
    pub mean_predicted: f64,
    /// Observed fraction of positives in the bin.
    pub fraction_positive: f64,
    /// Number of samples in the bin.
    pub count: usize,
}

/// Bin `probs` into `bins` equal-width buckets over `[0, 1]` and report the mean
/// prediction vs. the observed positive rate in each — a perfectly calibrated
/// model has `mean_predicted == fraction_positive` in every bin.
pub fn reliability_curve(probs: &[f64], labels: &[f64], bins: usize) -> Vec<ReliabilityBin> {
    let bins = bins.max(1);
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); bins];
    for (p, y) in probs.iter().zip(labels) {
        let mut idx = (p * bins as f64) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        acc[idx].0 += p;
        acc[idx].1 += y;
        acc[idx].2 += 1;
    }
    acc.into_iter()
        .filter(|(_, _, c)| *c > 0)
        .map(|(sp, pos, c)| ReliabilityBin {
            mean_predicted: sp / c as f64,
            fraction_positive: pos / c as f64,
            count: c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // scores that separate the classes: low scores -> 0, high -> 1
    fn data() -> (Vec<f64>, Vec<f64>) {
        let scores = vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
        let labels = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (scores, labels)
    }

    #[test]
    fn platt_is_monotone_and_bounded() {
        let (s, y) = data();
        let cal = PlattScaling::fit(&s, &y).unwrap();
        let p = cal.transform(&s);
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(p[0] < 0.5 && *p.last().unwrap() > 0.5);
        // monotone increasing in score
        assert!(p.windows(2).all(|w| w[0] <= w[1] + 1e-9));
    }

    #[test]
    fn isotonic_is_monotone_and_fits() {
        let (s, y) = data();
        let cal = IsotonicRegression::fit(&s, &y).unwrap();
        let p = cal.transform(&s);
        assert!(p.windows(2).all(|w| w[0] <= w[1] + 1e-9));
        assert!(p[0] < 0.5 && *p.last().unwrap() > 0.5);
    }

    #[test]
    fn reliability_curve_bins_counts() {
        let probs = vec![0.05, 0.15, 0.85, 0.95];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let curve = reliability_curve(&probs, &labels, 10);
        assert_eq!(curve.iter().map(|b| b.count).sum::<usize>(), 4);
        // low-prob bin is all negatives, high-prob bin all positives
        assert_eq!(curve.first().unwrap().fraction_positive, 0.0);
        assert_eq!(curve.last().unwrap().fraction_positive, 1.0);
    }
}
