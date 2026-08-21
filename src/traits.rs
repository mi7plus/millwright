//! The trait contract.
//!
//! Everything composes because everything speaks these traits, and they are
//! **object-safe**, so a [`Pipeline`](crate::pipeline::Pipeline) can hold a
//! heterogeneous `Vec<Box<dyn Transformer>>` and a `Box<dyn Model>`.
//!
//! Four traits are the core — the whole supervised lifecycle rides on them:
//!
//! | trait | shape |
//! |-------|-------|
//! | [`Transformer`] | `transform(&Frame) -> Frame` |
//! | [`Estimator`]   | `fit(&Dataset)` |
//! | [`Predictor`]   | `predict(&Frame) -> Vec<f64>` |
//! | [`ProbaPredictor`] | `predict_proba(&Frame) -> Frame` |
//!
//! A blanket [`Model`] ties `Estimator + Predictor` together. A few specialized
//! traits cover the shapes that don't fit the supervised mould:
//! [`Clusterer`] (unsupervised labels), [`Forecaster`] (time series),
//! [`PartialFit`] (out-of-core), and [`Balancer`] (train-time resampling).

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};

/// A hyperparameter value, addressable by path (the `"step__param"`
/// convention). Kept deliberately small for the spine.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::Int(v)
    }
}
impl From<i32> for ParamValue {
    fn from(v: i32) -> Self {
        ParamValue::Int(v as i64)
    }
}
impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        ParamValue::Int(v as i64)
    }
}
impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::Float(v)
    }
}
impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        ParamValue::Bool(v)
    }
}

impl ParamValue {
    /// Interpret as an integer, accepting an integral float.
    pub fn as_i64(&self) -> Result<i64> {
        match self {
            ParamValue::Int(i) => Ok(*i),
            ParamValue::Float(f) if f.fract() == 0.0 => Ok(*f as i64),
            other => Err(Error::Param(format!("expected an integer, got {other:?}"))),
        }
    }

    /// Interpret as a float.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            ParamValue::Float(f) => Ok(*f),
            ParamValue::Int(i) => Ok(*i as f64),
            other => Err(Error::Param(format!("expected a float, got {other:?}"))),
        }
    }

    /// Interpret as a boolean.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            ParamValue::Bool(b) => Ok(*b),
            other => Err(Error::Param(format!("expected a bool, got {other:?}"))),
        }
    }
}

/// Learns parameters from a frame, then maps `Frame -> Frame`.
///
/// A transformer is fitted in place with `&mut self`, which keeps the trait
/// object-safe so pipelines can own `Box<dyn Transformer>` steps.
///
/// The [`TransformerClone`] supertrait lets a boxed transformer be cloned, so a
/// search can re-fit a fresh copy of a pipeline on every CV fold.
pub trait Transformer: TransformerClone {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Learn any parameters needed to transform (means, encodings, …).
    fn fit(&mut self, frame: &Frame) -> Result<()>;

    /// Map an input frame to an output frame using the fitted parameters.
    fn transform(&self, frame: &Frame) -> Result<Frame>;

    /// Fit then transform in one pass. Override for a cheaper combined path.
    fn fit_transform(&mut self, frame: &Frame) -> Result<Frame> {
        self.fit(frame)?;
        self.transform(frame)
    }

    /// If (once fitted) this transformer is an affine map
    /// `y = (x - shift) / scale` per column, return `(shift, scale)`.
    ///
    /// Scalers implement this so a [`Pipeline`](crate::pipeline::Pipeline) can be
    /// folded into a single ONNX graph. Non-affine transformers return `None`.
    fn as_affine(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        None
    }

    /// Set a hyperparameter by name. Unknown names are an error.
    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        Err(Error::Param(format!(
            "{} has no parameter '{name}'",
            self.name()
        )))
    }
}

/// Clone support for boxed transformers (the object-safe half of `Clone`).
pub trait TransformerClone {
    /// Clone `self` into a fresh box.
    fn clone_box(&self) -> Box<dyn Transformer>;
}

impl<T> TransformerClone for T
where
    T: Transformer + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Transformer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Fits a model on a labelled [`Dataset`].
pub trait Estimator {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Fit the model on features + target.
    fn fit(&mut self, dataset: &Dataset) -> Result<()>;

    /// Set a hyperparameter by name. Unknown names are an error.
    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        Err(Error::Param(format!(
            "{} has no parameter '{name}'",
            self.name()
        )))
    }

    /// Build this estimator's ONNX graph, if it supports export. Overridden by
    /// backends that are ONNX-exportable; the default reports the estimator is
    /// not exportable.
    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        Err(Error::Backend(format!(
            "{} is not ONNX-exportable",
            self.name()
        )))
    }
}

/// Produces point predictions for a frame.
pub trait Predictor {
    /// Predict one value per row.
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>>;
}

/// Produces class-probability predictions.
///
/// The returned [`Frame`] has one column per class.
pub trait ProbaPredictor: Predictor {
    fn predict_proba(&self, frame: &Frame) -> Result<Frame>;
}

/// A time-series forecaster: fit on a one-dimensional series, then predict the
/// next `steps` values. A *different data shape* than the row/target contract,
/// so it has its own trait (as clustering does).
pub trait Forecaster {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Fit the forecaster on a historical series.
    fn fit(&mut self, series: &[f64]) -> Result<()>;

    /// Forecast the next `steps` values beyond the fitted history.
    fn forecast(&self, steps: usize) -> Result<Vec<f64>>;
}

/// An out-of-core estimator: learn from a stream of batches that never fully
/// load into memory. `partial_fit` updates the model with one batch at a time;
/// the estimator predicts through the usual [`Predictor`] contract.
pub trait PartialFit {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Update the model with one batch of `(features, target)`.
    fn partial_fit(&mut self, batch: &Dataset) -> Result<()>;
}

/// An unsupervised cluster model: fit on features alone (no target), then
/// assign each row a cluster label.
///
/// This is the contract for inductive clusterers (k-means, GMM) that can label
/// unseen data. Transductive methods that only label their training data (e.g.
/// DBSCAN) expose a `fit_predict` inherent method instead.
pub trait Clusterer {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Learn the clustering from the feature frame.
    fn fit(&mut self, frame: &Frame) -> Result<()>;

    /// Assign each row of `frame` a cluster label.
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>>;
}

/// A fittable, predicting model — the shape a pipeline's final step must have.
///
/// Blanket-implemented for anything that is an [`Estimator`], a [`Predictor`],
/// and `Clone`, so backends never implement it directly. The `Clone` bound (via
/// [`ModelClone`]) lets a search re-fit fresh copies across CV folds and lets
/// bagging/stacking clone their base estimators.
pub trait Model: Estimator + Predictor + ModelClone {}
impl<T: Estimator + Predictor + Clone + 'static> Model for T {}

/// Clone support for boxed models (the object-safe half of `Clone`).
pub trait ModelClone {
    /// Clone `self` into a fresh box.
    fn clone_box(&self) -> Box<dyn Model>;
}

impl<T> ModelClone for T
where
    T: Model + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Model> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A train-time resampler: given features and a target, produce a rebalanced
/// `(Frame, target)`. Unlike a [`Transformer`], a balancer runs **only during
/// `fit`** — never at predict time — because it changes the row set (e.g. SMOTE
/// synthesises minority-class rows).
pub trait Balancer: BalancerClone {
    /// A short, stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Resample `(features, target)` into a rebalanced pair.
    fn fit_resample(&self, features: &Frame, target: &[f64]) -> Result<(Frame, Vec<f64>)>;
}

/// Clone support for boxed balancers.
pub trait BalancerClone {
    /// Clone `self` into a fresh box.
    fn clone_box(&self) -> Box<dyn Balancer>;
}

impl<T> BalancerClone for T
where
    T: Balancer + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Balancer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Balancer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
