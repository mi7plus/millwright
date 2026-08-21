//! The four traits — the whole contract.
//!
//! Everything composes because everything speaks these traits, and they are
//! **object-safe**, so a [`Pipeline`](crate::pipeline::Pipeline) can hold a
//! heterogeneous `Vec<Box<dyn Transformer>>` and a `Box<dyn Model>`.
//!
//! | trait | shape |
//! |-------|-------|
//! | [`Transformer`] | `transform(&Frame) -> Frame` |
//! | [`Estimator`]   | `fit(&Dataset)` |
//! | [`Predictor`]   | `predict(&Frame) -> Vec<f64>` |
//! | [`ProbaPredictor`] | `predict_proba(&Frame) -> Frame` |

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
