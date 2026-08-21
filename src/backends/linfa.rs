//! The linfa backend — Millwright's second engine.
//!
//! Proves the [`Frame`] boundary conversion against a whole other array world:
//! linfa speaks `ndarray`, so every model here converts `Frame → Array2` (and
//! back) *inside the adapter*, exactly as the smartcore adapter converts to
//! `DenseMatrix`. Nothing above this module names an `ndarray` or linfa type.
//!
//! Ships the design brief's linfa line-up:
//! - [`KMeans`], [`GaussianMixture`] — inductive [`Clusterer`]s (fit + predict).
//! - [`Dbscan`] — transductive density clustering (`fit_predict` only).
//! - [`Pca`] — dimensionality reduction as a [`Transformer`].

use linfa::traits::{Fit, Predict, Transformer as LinfaTransformer};
use linfa::DatasetBase;
use linfa_clustering::{Dbscan as LinfaDbscan, GaussianMixtureModel, KMeans as LinfaKMeans};
use linfa_reduction::Pca as LinfaPca;
use ndarray::{Array2, ArrayBase, OwnedRepr};

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::traits::{Clusterer, Transformer};

/// `Frame → ndarray Array2<f64>` — the linfa-side conversion edge.
fn to_array2(frame: &Frame) -> Result<Array2<f64>> {
    let (n, p) = frame.shape();
    Array2::from_shape_vec((n, p), frame.buf().to_vec())
        .map_err(|e| Error::Backend(format!("ndarray conversion failed: {e}")))
}

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

/// K-means clustering, backed by linfa.
#[derive(Clone)]
pub struct KMeans {
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    model: Option<LinfaKMeans<f64, linfa_nn::distance::L2Dist>>,
}

impl KMeans {
    /// K-means seeking `n_clusters` clusters.
    pub fn new(n_clusters: usize) -> Self {
        KMeans {
            n_clusters,
            max_iter: 200,
            tolerance: 1e-4,
            model: None,
        }
    }

    /// Maximum Lloyd's-iteration count.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    /// Convergence tolerance on centroid movement.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }
}

impl Clusterer for KMeans {
    fn name(&self) -> &'static str {
        "KMeans"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let x = to_array2(frame)?;
        let dataset = DatasetBase::from(x);
        let model = LinfaKMeans::params(self.n_clusters)
            .max_n_iterations(self.max_iter)
            .tolerance(self.tolerance)
            .fit(&dataset)
            .map_err(|e| Error::Backend(format!("KMeans fit failed: {e}")))?;
        self.model = Some(model);
        Ok(())
    }

    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("KMeans::predict".into()))?;
        let x = to_array2(frame)?;
        let labels = model.predict(&x);
        Ok(labels.into_iter().map(|c| c as f64).collect())
    }
}

// ---------------------------------------------------------------------------
// Gaussian mixture
// ---------------------------------------------------------------------------

/// Gaussian-mixture clustering, backed by linfa.
#[derive(Clone)]
pub struct GaussianMixture {
    n_clusters: usize,
    max_iter: u64,
    model: Option<GaussianMixtureModel<f64>>,
}

impl GaussianMixture {
    /// A mixture of `n_clusters` Gaussian components.
    pub fn new(n_clusters: usize) -> Self {
        GaussianMixture {
            n_clusters,
            max_iter: 100,
            model: None,
        }
    }

    /// Maximum EM-iteration count.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }
}

impl Clusterer for GaussianMixture {
    fn name(&self) -> &'static str {
        "GaussianMixture"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let x = to_array2(frame)?;
        let dataset = DatasetBase::from(x);
        let model = GaussianMixtureModel::params(self.n_clusters)
            .max_n_iterations(self.max_iter)
            .fit(&dataset)
            .map_err(|e| Error::Backend(format!("GaussianMixture fit failed: {e}")))?;
        self.model = Some(model);
        Ok(())
    }

    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("GaussianMixture::predict".into()))?;
        let x = to_array2(frame)?;
        let labels = model.predict(&x);
        Ok(labels.into_iter().map(|c| c as f64).collect())
    }
}

// ---------------------------------------------------------------------------
// DBSCAN (transductive)
// ---------------------------------------------------------------------------

/// DBSCAN density clustering, backed by linfa.
///
/// DBSCAN is transductive — it labels the data it is given and cannot assign
/// unseen points — so it exposes [`Dbscan::fit_predict`] rather than the
/// [`Clusterer`] fit/predict split. Noise points are labelled `-1.0`.
#[derive(Clone)]
pub struct Dbscan {
    min_points: usize,
    tolerance: f64,
}

impl Dbscan {
    /// DBSCAN requiring at least `min_points` in an ε-neighbourhood to form a
    /// dense region.
    pub fn new(min_points: usize) -> Self {
        Dbscan {
            min_points,
            tolerance: 1.0,
        }
    }

    /// The ε neighbourhood radius.
    pub fn tolerance(mut self, eps: f64) -> Self {
        self.tolerance = eps;
        self
    }

    /// Cluster `frame`, returning one label per row (`-1.0` for noise).
    pub fn fit_predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let x = to_array2(frame)?;
        let labels = LinfaDbscan::params(self.min_points)
            .tolerance(self.tolerance)
            .transform(&x)
            .map_err(|e| Error::Backend(format!("DBSCAN failed: {e}")))?;
        Ok(labels
            .into_iter()
            .map(|c| c.map(|i| i as f64).unwrap_or(-1.0))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Principal-component analysis as a [`Transformer`], backed by linfa.
///
/// Fits on a frame and projects it onto its top `n_components` principal axes;
/// the output frame has columns `pc0..pc{k-1}`.
#[derive(Clone)]
pub struct Pca {
    n_components: usize,
    model: Option<LinfaPca<f64>>,
    out_cols: Vec<String>,
}

impl Pca {
    /// PCA keeping `n_components` principal components.
    pub fn new(n_components: usize) -> Self {
        Pca {
            n_components,
            model: None,
            out_cols: Vec::new(),
        }
    }
}

impl Transformer for Pca {
    fn name(&self) -> &'static str {
        "Pca"
    }

    fn fit(&mut self, frame: &Frame) -> Result<()> {
        let x = to_array2(frame)?;
        let dataset = DatasetBase::from(x);
        let model = LinfaPca::params(self.n_components)
            .fit(&dataset)
            .map_err(|e| Error::Backend(format!("PCA fit failed: {e}")))?;
        self.model = Some(model);
        self.out_cols = (0..self.n_components).map(|i| format!("pc{i}")).collect();
        Ok(())
    }

    fn transform(&self, frame: &Frame) -> Result<Frame> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("Pca::transform".into()))?;
        let x = to_array2(frame)?;
        let reduced: ArrayBase<OwnedRepr<f64>, _> = model.predict(&x);
        let (n, p) = reduced.dim();
        let buf: Vec<f64> = reduced.iter().copied().collect();
        Frame::new(buf, n, p, self.out_cols.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tight, far-apart blobs on two features.
    fn two_blobs() -> Frame {
        let mut rows = Vec::new();
        for i in 0..8 {
            rows.push(vec![0.0 + (i as f64) * 0.01, 0.0 + (i as f64) * 0.01]);
        }
        for i in 0..8 {
            rows.push(vec![10.0 + (i as f64) * 0.01, 10.0 + (i as f64) * 0.01]);
        }
        Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap()
    }

    #[test]
    fn kmeans_separates_two_blobs() {
        let f = two_blobs();
        let mut km = KMeans::new(2);
        km.fit(&f).unwrap();
        let labels = km.predict(&f).unwrap();
        // First blob shares a label; the two blobs differ. (Label ids are
        // arbitrary, so compare structure, not exact values.)
        assert_eq!(labels[0], labels[7]);
        assert_ne!(labels[0], labels[8]);
        assert_eq!(labels[8], labels[15]);
    }

    #[test]
    fn gaussian_mixture_separates_two_blobs() {
        let f = two_blobs();
        let mut gmm = GaussianMixture::new(2);
        gmm.fit(&f).unwrap();
        let labels = gmm.predict(&f).unwrap();
        assert_eq!(labels[0], labels[7]);
        assert_ne!(labels[0], labels[8]);
    }

    #[test]
    fn dbscan_finds_two_clusters() {
        let labels = Dbscan::new(3).tolerance(1.0).fit_predict(&two_blobs()).unwrap();
        assert_eq!(labels[0], labels[7]);
        assert_ne!(labels[0], labels[8]);
    }

    #[test]
    fn pca_reduces_dimensionality() {
        let f = two_blobs();
        let mut pca = Pca::new(1);
        let out = pca.fit_transform(&f).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_eq!(out.nrows(), f.nrows());
        assert_eq!(out.columns(), &["pc0".to_string()]);
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = two_blobs();
        assert!(KMeans::new(2).predict(&f).is_err());
        assert!(Pca::new(1).transform(&f).is_err());
    }
}
