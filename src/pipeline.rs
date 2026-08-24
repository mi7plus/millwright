//! `Pipeline` — the composition that ties the contract together.
//!
//! A pipeline is a named sequence of [`Transformer`] steps followed by one
//! final [`Model`] estimator. Fitting threads the frame through each
//! transformer's `fit_transform`, then fits the estimator on the result;
//! predicting replays the fitted transforms and calls the estimator.
//!
//! Steps are addressable by name, so a search can tune any parameter anywhere
//! in the chain with the scikit-learn `"step__param"` convention (see
//! [`Pipeline::set_param`]). Because a `Pipeline` is itself an [`Estimator`]
//! and [`Predictor`], pipelines nest.

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::traits::{Balancer, Estimator, Model, ParamValue, Predictor, Transformer};

/// A preprocessing-plus-model pipeline.
#[derive(Default, Clone)]
pub struct Pipeline {
    steps: Vec<(String, Box<dyn Transformer>)>,
    balancer: Option<Box<dyn Balancer>>,
    estimator: Option<(String, Box<dyn Model>)>,
    fitted: bool,
}

impl Pipeline {
    /// An empty pipeline.
    pub fn new() -> Self {
        Pipeline {
            steps: Vec::new(),
            balancer: None,
            estimator: None,
            fitted: false,
        }
    }

    /// Append a named transformer step. Builder-style; chainable.
    pub fn step(mut self, name: impl Into<String>, t: impl Transformer + 'static) -> Self {
        self.steps.push((name.into(), Box::new(t)));
        self
    }

    /// Set a train-time [`Balancer`] (e.g. SMOTE), applied after the transforms
    /// during `fit` and skipped entirely at predict time. Builder-style.
    pub fn balance(mut self, b: impl Balancer + 'static) -> Self {
        self.balancer = Some(Box::new(b));
        self
    }

    /// Set the final estimator. Builder-style; chainable.
    pub fn estimator(mut self, name: impl Into<String>, e: impl Model + 'static) -> Self {
        self.estimator = Some((name.into(), Box::new(e)));
        self
    }

    /// The step / estimator names, in order — handy for diagnostics.
    pub fn step_names(&self) -> Vec<&str> {
        self.steps
            .iter()
            .map(|(n, _)| n.as_str())
            .chain(self.estimator.iter().map(|(n, _)| n.as_str()))
            .collect()
    }

    /// Route a `"step__param"` path to the named step or estimator.
    ///
    /// The name before the first `__` selects the step; the remainder is the
    /// parameter passed on to that step (recursing for nested pipelines).
    pub fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        let (step, rest) = path
            .split_once("__")
            .ok_or_else(|| Error::Param(format!("'{path}' is not a 'step__param' path")))?;
        for (name, t) in &mut self.steps {
            if name == step {
                return t.set_param(rest, value);
            }
        }
        if let Some((name, e)) = &mut self.estimator {
            if name == step {
                return e.set_param(rest, value);
            }
        }
        Err(Error::Param(format!("no step named '{step}' in pipeline")))
    }

    fn require_estimator(&self) -> Result<&(String, Box<dyn Model>)> {
        self.estimator
            .as_ref()
            .ok_or_else(|| Error::Pipeline("pipeline has no estimator".into()))
    }

    /// Apply every fitted transformer to `frame`, returning the transformed
    /// frame the estimator sees.
    fn forward(&self, frame: &Frame) -> Result<Frame> {
        let mut current = frame.clone();
        for (_, t) in &self.steps {
            current = t.transform(&current)?;
        }
        Ok(current)
    }
}

impl Estimator for Pipeline {
    fn name(&self) -> &'static str {
        "Pipeline"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        self.require_estimator()?;
        let mut current = dataset.features().clone();
        for (_, t) in &mut self.steps {
            current = t.fit_transform(&current)?;
        }
        // Train-time resampling (SMOTE, etc.) runs here — never at predict time.
        let transformed = match &self.balancer {
            Some(b) => {
                let (bx, by) = b.fit_resample(&current, dataset.target())?;
                Dataset::new(bx, by)?
            }
            None => dataset.with_features(current),
        };
        let (_, est) = self.estimator.as_mut().ok_or_else(|| {
            Error::Pipeline("pipeline has no estimator; add one before fit".into())
        })?;
        est.fit(&transformed)?;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        // Delegating to the inherent method lets a Pipeline nest inside another
        // Pipeline and still be reached by a `"outer__inner__param"` path.
        Pipeline::set_param(self, name, value)
    }

    fn supports_proba(&self) -> bool {
        self.estimator
            .as_ref()
            .map(|(_, estimator)| estimator.supports_proba())
            .unwrap_or(false)
    }

    fn predict_proba_dyn(&self, frame: &Frame) -> Result<Frame> {
        if !self.fitted {
            return Err(Error::NotFitted("Pipeline::predict_proba".into()));
        }
        let transformed = self.forward(frame)?;
        let (_, estimator) = self.require_estimator()?;
        estimator.predict_proba_dyn(&transformed)
    }

    // The object-safe ONNX hook, so a boxed pipeline (e.g. a search winner held
    // as `Box<dyn Model>`) can still be exported. Delegates to the pipeline's
    // own `ExportOnnx` implementation.
    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        crate::onnx::ExportOnnx::to_onnx(self)
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proba_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        let (_, estimator) = self.require_estimator()?;
        let mut proto = estimator.to_onnx_proba_proto()?;
        let prefixes: Vec<crate::onnx::Prefix> = self
            .steps
            .iter()
            .map(|(name, transformer)| {
                transformer.onnx_prefix().ok_or_else(|| {
                    Error::Backend(format!(
                        "pipeline step '{name}' ({}) is not ONNX-exportable",
                        transformer.name()
                    ))
                })
            })
            .collect::<Result<_>>()?;
        if !prefixes.is_empty() {
            crate::onnx::prepend_prefixes(&mut proto, &prefixes)?;
        }
        Ok(proto)
    }
}

impl Predictor for Pipeline {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Pipeline::predict".into()));
        }
        let transformed = self.forward(frame)?;
        let (_, est) = self.require_estimator()?;
        est.predict(&transformed)
    }
}

/// Export the whole pipeline as one ONNX graph: each leading transformer that
/// can express itself as ONNX nodes (scalers as an affine map, imputers as a
/// `Where(IsNaN, fill, x)`, one-hot encoders as a `Gather`/`Equal`/`Concat`
/// expansion) is spliced in front of the estimator's graph, in order. A step
/// with no ONNX form is reported as an error naming it; the train-time balancer
/// is inference-irrelevant and skipped.
#[cfg(feature = "onnx")]
impl crate::onnx::ExportOnnx for Pipeline {
    fn to_onnx(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        let (_, est) = self.require_estimator()?;
        let mut proto = est.to_onnx_proto()?;

        let prefixes: Vec<crate::onnx::Prefix> = self
            .steps
            .iter()
            .map(|(name, t)| {
                t.onnx_prefix().ok_or_else(|| {
                    Error::Backend(format!(
                        "pipeline step '{name}' ({}) is not ONNX-exportable",
                        t.name()
                    ))
                })
            })
            .collect::<Result<_>>()?;

        if !prefixes.is_empty() {
            crate::onnx::prepend_prefixes(&mut proto, &prefixes)?;
        }
        Ok(proto)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::StandardScaler;

    /// A trivial estimator that predicts the mean of the training target,
    /// letting us test pipeline plumbing without a backend.
    #[derive(Default, Clone)]
    struct MeanBaseline {
        mean: f64,
    }
    impl Estimator for MeanBaseline {
        fn name(&self) -> &'static str {
            "MeanBaseline"
        }
        fn fit(&mut self, d: &Dataset) -> Result<()> {
            self.mean = d.target().iter().sum::<f64>() / d.target().len() as f64;
            Ok(())
        }
    }
    impl Predictor for MeanBaseline {
        fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
            Ok(vec![self.mean; frame.nrows()])
        }
    }

    #[test]
    fn fits_and_predicts_through_a_transform() {
        let x = Frame::from_rows(
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec!["x".into()],
        )
        .unwrap();
        let ds = Dataset::new(x.clone(), vec![10.0, 10.0, 20.0, 20.0]).unwrap();

        let mut pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("mean", MeanBaseline::default());

        pipe.fit(&ds).unwrap();
        let preds = pipe.predict(&x).unwrap();
        assert_eq!(preds, vec![15.0, 15.0, 15.0, 15.0]);
        assert_eq!(pipe.step_names(), vec!["scale", "mean"]);
    }

    #[test]
    fn routes_params_by_path() {
        let mut pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("mean", MeanBaseline::default());
        // known param on a real step
        assert!(pipe
            .set_param("scale__with_mean", ParamValue::Bool(false))
            .is_ok());
        // unknown step
        assert!(pipe.set_param("nope__x", ParamValue::Int(1)).is_err());
        // unknown param on a known step
        assert!(pipe.set_param("scale__bogus", ParamValue::Int(1)).is_err());
        // malformed path
        assert!(pipe.set_param("noseparator", ParamValue::Int(1)).is_err());
    }

    #[test]
    fn predict_before_fit_errors() {
        let x = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        let pipe = Pipeline::new().estimator("mean", MeanBaseline::default());
        assert!(pipe.predict(&x).is_err());
    }
}
