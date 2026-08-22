//! Unsupervised outlier detection.
//!
//! [`Mahalanobis`] scores each row by its covariance-aware distance from the
//! fitted centre; [`KnnScore`] by the distance to its k-th nearest training
//! neighbour. Higher score = more anomalous. Both fit on a [`Frame`] and score a
//! [`Frame`], with a `threshold` turning scores into boolean flags.
//!
//! (Isolation Forest is deferred until the tree ecosystem exposes the internals
//! it needs; these two cover the common covariance / density cases.)

// Small dense linear algebra by hand — indexed loops read more clearly here than
// iterator adapters.
#![allow(clippy::needless_range_loop)]

use crate::error::{Error, Result};
use crate::frame::Frame;

/// A common contract for unsupervised outlier scorers: fit on data, score each
/// row (higher = more anomalous), and flag rows past a threshold. Implemented by
/// [`Mahalanobis`] and [`KnnScore`], so they are interchangeable behind
/// `Box<dyn OutlierDetector>`.
pub trait OutlierDetector {
    /// Learn the detector from a feature frame.
    fn fit(&mut self, frame: &Frame) -> Result<()>;

    /// The anomaly score of each row (higher = more anomalous).
    fn score(&self, frame: &Frame) -> Result<Vec<f64>>;

    /// Flag rows whose score exceeds `threshold`.
    fn is_outlier(&self, frame: &Frame, threshold: f64) -> Result<Vec<bool>> {
        Ok(self
            .score(frame)?
            .into_iter()
            .map(|s| s > threshold)
            .collect())
    }
}

/// Mahalanobis-distance outlier scorer: `d(x) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))`.
#[derive(Clone, Debug, Default)]
pub struct Mahalanobis {
    mean: Vec<f64>,
    inv_cov: Vec<Vec<f64>>,
    fitted: bool,
}

impl Mahalanobis {
    /// A new, unfitted scorer.
    pub fn new() -> Self {
        Mahalanobis::default()
    }

    /// Learn the centre and (ridge-regularized) inverse covariance.
    pub fn fit(&mut self, frame: &Frame) -> Result<()> {
        let (n, p) = frame.shape();
        if n < 2 {
            return Err(Error::Shape("Mahalanobis::fit needs >= 2 rows".into()));
        }
        let mean: Vec<f64> = (0..p)
            .map(|c| frame.column(c).iter().sum::<f64>() / n as f64)
            .collect();
        // population covariance with a small ridge for invertibility
        let mut cov = vec![vec![0.0; p]; p];
        for r in 0..n {
            for i in 0..p {
                let di = frame.get(r, i) - mean[i];
                for j in 0..p {
                    cov[i][j] += di * (frame.get(r, j) - mean[j]);
                }
            }
        }
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v /= n as f64;
                if i == j {
                    *v += 1e-6;
                }
            }
        }
        self.inv_cov = invert(cov).ok_or_else(|| Error::Backend("singular covariance".into()))?;
        self.mean = mean;
        self.fitted = true;
        Ok(())
    }

    /// The Mahalanobis distance of each row.
    pub fn score(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Mahalanobis::score".into()));
        }
        let p = self.mean.len();
        if frame.ncols() != p {
            return Err(Error::Shape(format!(
                "Mahalanobis: expected {p} columns, got {}",
                frame.ncols()
            )));
        }
        let mut out = Vec::with_capacity(frame.nrows());
        for r in 0..frame.nrows() {
            let d: Vec<f64> = (0..p).map(|i| frame.get(r, i) - self.mean[i]).collect();
            // dᵀ Σ⁻¹ d
            let mut q = 0.0;
            for i in 0..p {
                let mut row = 0.0;
                for j in 0..p {
                    row += self.inv_cov[i][j] * d[j];
                }
                q += d[i] * row;
            }
            out.push(q.max(0.0).sqrt());
        }
        Ok(out)
    }

    /// Flag rows whose score exceeds `threshold`.
    pub fn is_outlier(&self, frame: &Frame, threshold: f64) -> Result<Vec<bool>> {
        Ok(self
            .score(frame)?
            .into_iter()
            .map(|s| s > threshold)
            .collect())
    }
}

/// k-nearest-neighbour distance scorer: each row scores as its distance to the
/// k-th nearest training point. Isolated points score high.
#[derive(Clone, Debug)]
pub struct KnnScore {
    k: usize,
    train: Vec<Vec<f64>>,
}

impl KnnScore {
    /// A scorer using the k-th nearest neighbour distance.
    pub fn new(k: usize) -> Self {
        KnnScore {
            k: k.max(1),
            train: Vec::new(),
        }
    }

    /// Store the training points.
    pub fn fit(&mut self, frame: &Frame) -> Result<()> {
        if frame.nrows() == 0 {
            return Err(Error::Shape("KnnScore::fit needs >= 1 row".into()));
        }
        self.train = frame.as_rows();
        Ok(())
    }

    /// The k-th nearest-neighbour distance of each row.
    pub fn score(&self, frame: &Frame) -> Result<Vec<f64>> {
        if self.train.is_empty() {
            return Err(Error::NotFitted("KnnScore::score".into()));
        }
        let p = self.train[0].len();
        if frame.ncols() != p {
            return Err(Error::Shape(format!(
                "KnnScore: expected {p} columns, got {}",
                frame.ncols()
            )));
        }
        let k = self.k.min(self.train.len());
        let mut out = Vec::with_capacity(frame.nrows());
        for r in 0..frame.nrows() {
            let row = frame.row(r);
            let mut dists: Vec<f64> = self
                .train
                .iter()
                .map(|t| {
                    row.iter()
                        .zip(t)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            dists.sort_by(f64::total_cmp);
            out.push(dists[k - 1]);
        }
        Ok(out)
    }

    /// Flag rows whose score exceeds `threshold`.
    pub fn is_outlier(&self, frame: &Frame, threshold: f64) -> Result<Vec<bool>> {
        Ok(self
            .score(frame)?
            .into_iter()
            .map(|s| s > threshold)
            .collect())
    }
}

impl OutlierDetector for Mahalanobis {
    fn fit(&mut self, frame: &Frame) -> Result<()> {
        Mahalanobis::fit(self, frame)
    }
    fn score(&self, frame: &Frame) -> Result<Vec<f64>> {
        Mahalanobis::score(self, frame)
    }
}

impl OutlierDetector for KnnScore {
    fn fit(&mut self, frame: &Frame) -> Result<()> {
        KnnScore::fit(self, frame)
    }
    fn score(&self, frame: &Frame) -> Result<Vec<f64>> {
        KnnScore::score(self, frame)
    }
}

/// Invert a square matrix by Gauss-Jordan elimination; `None` if singular.
fn invert(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    for col in 0..n {
        // partial pivot
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..n {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = a[r][col];
                for j in 0..n {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_with_outlier() -> Frame {
        // a tight blob around the origin, plus one far-off point (row 5)
        Frame::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.1, -0.1],
                vec![-0.1, 0.1],
                vec![0.05, 0.05],
                vec![-0.05, -0.05],
                vec![8.0, 9.0], // the outlier
            ],
            vec!["a".into(), "b".into()],
        )
        .unwrap()
    }

    #[test]
    fn mahalanobis_flags_the_outlier() {
        let f = cluster_with_outlier();
        let mut m = Mahalanobis::new();
        m.fit(&f).unwrap();
        let scores = m.score(&f).unwrap();
        // the last row is by far the most anomalous
        let max_idx = (0..scores.len())
            .max_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap())
            .unwrap();
        assert_eq!(max_idx, 5);
    }

    #[test]
    fn knn_scores_isolated_point_highest() {
        let f = cluster_with_outlier();
        let mut knn = KnnScore::new(2);
        knn.fit(&f).unwrap();
        let scores = knn.score(&f).unwrap();
        let max_idx = (0..scores.len())
            .max_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap())
            .unwrap();
        assert_eq!(max_idx, 5);
        assert!(knn.is_outlier(&f, 1.0).unwrap()[5]);
    }

    #[test]
    fn invert_recovers_identity() {
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert(m.clone()).unwrap();
        // m * inv ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let v: f64 = (0..2).map(|k| m[i][k] * inv[k][j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((v - expect).abs() < 1e-9);
            }
        }
    }
}
