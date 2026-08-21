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
pub trait Transformer {
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
/// Blanket-implemented for anything that is both an [`Estimator`] and a
/// [`Predictor`], so backends never implement it directly.
pub trait Model: Estimator + Predictor {}
impl<T: Estimator + Predictor> Model for T {}
