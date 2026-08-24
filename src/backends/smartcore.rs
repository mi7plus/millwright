//! The smartcore backend — Millwright's first engine.
//!
//! Wraps a couple of smartcore models behind the framework's traits. The
//! `Frame -> DenseMatrix` conversion happens here, at the edge, via
//! [`as_dense`]; nothing above this module ever names a smartcore type.
//!
//! The models adapted here prove the contract across a real backend:
//! - [`RandomForest`] — the classifier from the design brief's API example.
//! - [`LinearRegression`] — the regression path.
//! - [`Knn`], [`Svc`], [`NaiveBayes`] — k-nearest-neighbours, a (one-vs-one)
//!   support vector classifier, and Gaussian naive Bayes.

use std::sync::Arc;

use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression as ScLinearRegression, LinearRegressionParameters,
};
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::svm::svc::{MultiClassSVC, SVCParameters};
use smartcore::svm::Kernels;

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::traits::{Estimator, ParamValue, Predictor};

// The concrete smartcore model types, spelled once.
type ScForest = RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>>;
type ScLinReg = ScLinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>;
type ScKnn = KNNClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>, Euclidian<f64>>;
type ScNb = GaussianNB<f64, u64, DenseMatrix<f64>, Vec<u64>>;
type ScSvcParams = SVCParameters<f64, i64, DenseMatrix<f64>, Vec<i64>>;
type ScMultiSvc = MultiClassSVC<'static, f64, i64, DenseMatrix<f64>, Vec<i64>>;

/// Validate and convert a target vector to integer class labels.
fn int_labels(dataset: &Dataset) -> Result<Vec<i64>> {
    dataset
        .target()
        .iter()
        .map(|&value| {
            if value.is_finite()
                && value.fract() == 0.0
                && value >= i64::MIN as f64
                && value <= i64::MAX as f64
            {
                Ok(value as i64)
            } else {
                Err(Error::Shape(format!(
                    "classifier targets must be finite integer labels; got {value}"
                )))
            }
        })
        .collect()
}

/// Convert a [`Frame`] to smartcore's native `DenseMatrix<f64>`.
///
/// This is the one place the boundary type crosses into an engine's world.
pub fn as_dense(frame: &Frame) -> Result<DenseMatrix<f64>> {
    DenseMatrix::from_2d_vec(&frame.as_rows())
        .map_err(|e| Error::Backend(format!("DenseMatrix conversion failed: {e}")))
}

/// A random-forest classifier backed by smartcore.
///
/// Class labels are the (integral) values of the [`Dataset`] target, cast to
/// integers for smartcore and back to `f64` on predict.
#[derive(Clone)]
pub struct RandomForest {
    n_trees: u16,
    max_depth: Option<u16>,
    // Arc so the wrapper is cheaply Clone (the fitted model is immutable and
    // smartcore's own type is not Clone); a re-fit replaces the Arc.
    model: Option<Arc<ScForest>>,
}

impl RandomForest {
    /// A forest with smartcore's defaults (100 trees, unbounded depth).
    pub fn new() -> Self {
        RandomForest {
            n_trees: 100,
            max_depth: None,
            model: None,
        }
    }

    /// Set the number of trees.
    pub fn n_trees(mut self, n: u16) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the maximum tree depth.
    pub fn max_depth(mut self, d: u16) -> Self {
        self.max_depth = Some(d);
        self
    }
}

impl Default for RandomForest {
    fn default() -> Self {
        RandomForest::new()
    }
}

impl Estimator for RandomForest {
    fn name(&self) -> &'static str {
        "RandomForest"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y = int_labels(dataset)?;

        let mut params = RandomForestClassifierParameters::default().with_n_trees(self.n_trees);
        if let Some(d) = self.max_depth {
            params = params.with_max_depth(d);
        }

        let model = RandomForestClassifier::fit(&x, &y, params)
            .map_err(|e| Error::Backend(format!("RandomForest fit failed: {e}")))?;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_trees" => self.n_trees = value.as_i64()? as u16,
            "max_depth" => self.max_depth = Some(value.as_i64()? as u16),
            other => {
                return Err(Error::Param(format!(
                    "RandomForest has no parameter '{other}'"
                )))
            }
        }
        Ok(())
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        crate::onnx::ExportOnnx::to_onnx(self)
    }
}

impl Predictor for RandomForest {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("RandomForest::predict".into()))?;
        let x = as_dense(frame)?;
        let y = model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("RandomForest predict failed: {e}")))?;
        Ok(y.into_iter().map(|c| c as f64).collect())
    }
}

/// Ordinary least squares, backed by smartcore.
#[derive(Clone)]
pub struct LinearRegression {
    model: Option<Arc<ScLinReg>>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { model: None }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::new()
    }
}

impl Estimator for LinearRegression {
    fn name(&self) -> &'static str {
        "LinearRegression"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y: Vec<f64> = dataset.target().to_vec();
        let model = ScLinearRegression::fit(&x, &y, LinearRegressionParameters::default())
            .map_err(|e| Error::Backend(format!("LinearRegression fit failed: {e}")))?;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        crate::onnx::ExportOnnx::to_onnx(self)
    }
}

impl Predictor for LinearRegression {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("LinearRegression::predict".into()))?;
        let x = as_dense(frame)?;
        model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("LinearRegression predict failed: {e}")))
    }
}

/// A k-nearest-neighbours classifier backed by smartcore (Euclidean distance).
#[derive(Clone)]
pub struct Knn {
    k: usize,
    model: Option<Arc<ScKnn>>,
}

impl Knn {
    /// A KNN classifier with `k = 5`.
    pub fn new() -> Self {
        Knn { k: 5, model: None }
    }
    /// A KNN classifier with a given number of neighbours.
    pub fn k(k: usize) -> Self {
        Knn { k, model: None }
    }
}

impl Default for Knn {
    fn default() -> Self {
        Knn::new()
    }
}

impl Estimator for Knn {
    fn name(&self) -> &'static str {
        "Knn"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y = int_labels(dataset)?;
        let params = KNNClassifierParameters::default().with_k(self.k);
        let model = KNNClassifier::fit(&x, &y, params)
            .map_err(|e| Error::Backend(format!("Knn fit failed: {e}")))?;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "k" => self.k = value.as_i64()? as usize,
            other => return Err(Error::Param(format!("Knn has no parameter '{other}'"))),
        }
        Ok(())
    }
}

impl Predictor for Knn {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("Knn::predict".into()))?;
        let x = as_dense(frame)?;
        let y = model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("Knn predict failed: {e}")))?;
        Ok(y.into_iter().map(|c| c as f64).collect())
    }
}

/// Gaussian naive Bayes, backed by smartcore.
#[derive(Clone)]
pub struct NaiveBayes {
    model: Option<Arc<ScNb>>,
}

impl NaiveBayes {
    pub fn new() -> Self {
        NaiveBayes { model: None }
    }
}

impl Default for NaiveBayes {
    fn default() -> Self {
        NaiveBayes::new()
    }
}

impl Estimator for NaiveBayes {
    fn name(&self) -> &'static str {
        "NaiveBayes"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        // smartcore's Gaussian NB wants unsigned class labels.
        let y: Vec<u64> = int_labels(dataset)?
            .into_iter()
            .map(|label| {
                u64::try_from(label).map_err(|_| {
                    Error::Shape("NaiveBayes targets must be non-negative integer labels".into())
                })
            })
            .collect::<Result<_>>()?;
        let model = GaussianNB::fit(&x, &y, Default::default())
            .map_err(|e| Error::Backend(format!("NaiveBayes fit failed: {e}")))?;
        self.model = Some(Arc::new(model));
        Ok(())
    }
}

impl Predictor for NaiveBayes {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("NaiveBayes::predict".into()))?;
        let x = as_dense(frame)?;
        let y = model
            .predict(&x)
            .map_err(|e| Error::Backend(format!("NaiveBayes predict failed: {e}")))?;
        Ok(y.into_iter().map(|c| c as f64).collect())
    }
}

#[derive(Clone, Copy)]
enum SvcKernel {
    Linear,
    Rbf { gamma: f64 },
}

/// A fitted SVC together with the parameters it borrows.
///
/// smartcore's `SVC` binds one lifetime across its inputs, but the fitted model
/// copies its support vectors and only retains a reference to the *parameters*
/// (the kernel). This holder owns those parameters at a stable heap address and
/// keeps them alive for the model's whole life, so the borrow is sound.
struct FittedSvc {
    // `model` borrows `_params`; declared first so it is dropped first.
    model: ScMultiSvc,
    _params: Box<ScSvcParams>,
}

impl FittedSvc {
    fn fit(x: DenseMatrix<f64>, y: Vec<i64>, params: ScSvcParams) -> Result<FittedSvc> {
        let params = Box::new(params);
        // SAFETY: `MultiClassSVC` stores only a `&parameters` reference (support
        // vectors are copied in) and does not retain `x`/`y`. We box `params`
        // for a stable address, hand `fit` a `'static` view of it, and keep the
        // box alive for the model's whole life, dropping `model` before
        // `_params` (field order). `x`/`y` outlive the `fit` call and are not
        // referenced by the returned model, so widening their borrow for the
        // call only is sound.
        let params_ref: &'static ScSvcParams = unsafe { &*(params.as_ref() as *const ScSvcParams) };
        let x_ref: &'static DenseMatrix<f64> = unsafe { &*(&x as *const DenseMatrix<f64>) };
        let y_ref: &'static Vec<i64> = unsafe { &*(&y as *const Vec<i64>) };
        let model = MultiClassSVC::fit(x_ref, y_ref, params_ref)
            .map_err(|e| Error::Backend(format!("Svc fit failed: {e}")))?;
        Ok(FittedSvc {
            model,
            _params: params,
        })
    }

    fn predict(&self, x: &DenseMatrix<f64>) -> Result<Vec<f64>> {
        self.model
            .predict(x)
            .map_err(|e| Error::Backend(format!("Svc predict failed: {e}")))
    }
}

// SAFETY: `FittedSvc` is immutable after construction. Its internal reference
// points into its own boxed `_params` (they move together — no aliasing across
// the boundary), and smartcore's kernels hold only plain numeric config, so the
// value is effectively `Send`/`Sync` even though `dyn Kernel` is not bounded as
// such. This gives `Svc` the same thread-safety as the other backend models.
unsafe impl Send for FittedSvc {}
unsafe impl Sync for FittedSvc {}

/// A support vector classifier backed by smartcore (one-vs-one for multiclass).
#[derive(Clone)]
pub struct Svc {
    c: f64,
    kernel: SvcKernel,
    fitted: Option<Arc<FittedSvc>>,
}

impl Svc {
    /// A linear-kernel SVC (`C = 1`).
    pub fn new() -> Self {
        Svc {
            c: 1.0,
            kernel: SvcKernel::Linear,
            fitted: None,
        }
    }
    /// A linear-kernel SVC.
    pub fn linear() -> Self {
        Svc::new()
    }
    /// An RBF-kernel SVC (default `gamma = 0.5`).
    pub fn rbf() -> Self {
        Svc {
            c: 1.0,
            kernel: SvcKernel::Rbf { gamma: 0.5 },
            fitted: None,
        }
    }
    /// Set the regularization parameter `C`.
    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    /// Set the RBF kernel bandwidth `gamma` (switches to an RBF kernel).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.kernel = SvcKernel::Rbf { gamma };
        self
    }
}

impl Default for Svc {
    fn default() -> Self {
        Svc::new()
    }
}

impl Estimator for Svc {
    fn name(&self) -> &'static str {
        "Svc"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        let x = as_dense(dataset.features())?;
        let y = int_labels(dataset)?;
        let params = ScSvcParams::default().with_c(self.c);
        let params = match self.kernel {
            SvcKernel::Linear => params.with_kernel(Kernels::linear()),
            SvcKernel::Rbf { gamma } => params.with_kernel(Kernels::rbf().with_gamma(gamma)),
        };
        self.fitted = Some(Arc::new(FittedSvc::fit(x, y, params)?));
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => self.c = value.as_f64()?,
            "gamma" => {
                self.kernel = SvcKernel::Rbf {
                    gamma: value.as_f64()?,
                }
            }
            other => return Err(Error::Param(format!("Svc has no parameter '{other}'"))),
        }
        Ok(())
    }
}

impl Predictor for Svc {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| Error::NotFitted("Svc::predict".into()))?;
        let x = as_dense(frame)?;
        fitted.predict(&x)
    }
}

/// Export a fitted [`RandomForest`] to an ONNX tree-ensemble classifier.
///
/// Uses onnx-export-rs's serde-based compat adapter, so it is independent of the
/// smartcore version onnx-export itself was built against.
#[cfg(feature = "onnx")]
impl crate::onnx::ExportOnnx for RandomForest {
    fn to_onnx(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        use onnx_export_rs::adapters::smartcore_compat::random_forest_classifier;
        use onnx_export_rs::canonical::TreeTask;
        use onnx_export_rs::exporters::export_tree_ensemble;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("RandomForest::to_onnx".into()))?;
        let (forest, _labels) = random_forest_classifier(&**model)
            .map_err(|e| Error::Backend(format!("RandomForest ONNX adapter failed: {e}")))?;
        export_tree_ensemble(&forest, TreeTask::Classification)
            .map_err(|e| Error::Backend(format!("RandomForest ONNX export failed: {e}")))
    }
}

/// Export a fitted [`LinearRegression`] to an ONNX `Gemm` graph.
///
/// The canonical weights are read from smartcore's own coefficient/intercept
/// accessors, so this is independent of onnx-export's smartcore version. Unlike
/// the tree ensemble, the resulting graph runs in tract.
#[cfg(feature = "onnx")]
impl crate::onnx::ExportOnnx for LinearRegression {
    fn to_onnx(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        use onnx_export_rs::canonical::LinearModelWeights;
        use onnx_export_rs::exporters::export_linear;
        use smartcore::linalg::basic::arrays::Array;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::NotFitted("LinearRegression::to_onnx".into()))?;
        let coef = model.coefficients();
        let (nr, nc) = coef.shape();
        let mut coefficients = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                coefficients.push(*coef.get((r, c)));
            }
        }
        let weights =
            LinearModelWeights::new(ndarray::Array1::from(coefficients), *model.intercept());
        Ok(export_linear(&weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_forest_separates_two_clusters() {
        // Two well-separated classes on a single feature.
        let x = Frame::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.5, 0.2],
                vec![0.1, 0.4],
                vec![9.0, 9.0],
                vec![9.5, 8.8],
                vec![8.9, 9.3],
            ],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let ds = Dataset::new(x.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut rf = RandomForest::new().n_trees(50);
        rf.fit(&ds).unwrap();

        let test = Frame::from_rows(
            vec![vec![0.2, 0.1], vec![9.2, 9.1]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(rf.predict(&test).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn linear_regression_recovers_a_line() {
        // y = 2x + 1
        let x = Frame::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec!["x".into()],
        )
        .unwrap();
        let ds = Dataset::new(x, vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();

        let test = Frame::from_rows(vec![vec![4.0]], vec!["x".into()]).unwrap();
        let pred = lr.predict(&test).unwrap()[0];
        assert!((pred - 9.0).abs() < 1e-6, "expected ~9.0, got {pred}");
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
        assert!(RandomForest::new().predict(&f).is_err());
        assert!(LinearRegression::new().predict(&f).is_err());
        assert!(Knn::new().predict(&f).is_err());
        assert!(Svc::new().predict(&f).is_err());
        assert!(NaiveBayes::new().predict(&f).is_err());
    }

    // Two separable clusters shared by the classifier tests below.
    fn two_clusters() -> (Dataset, Frame) {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..12 {
            rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
            y.push(1.0);
        }
        let cols = vec!["a".to_string(), "b".to_string()];
        let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
        let probe = Frame::from_rows(vec![vec![0.2, 0.1], vec![9.2, 9.1]], cols).unwrap();
        (ds, probe)
    }

    #[test]
    fn knn_separates_two_clusters() {
        let (ds, probe) = two_clusters();
        let mut m = Knn::k(3);
        m.fit(&ds).unwrap();
        assert_eq!(m.predict(&probe).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn svc_separates_two_clusters() {
        let (ds, probe) = two_clusters();
        for mut m in [Svc::linear(), Svc::rbf()] {
            m.fit(&ds).unwrap();
            assert_eq!(m.predict(&probe).unwrap(), vec![0.0, 1.0], "kernel differs");
        }
    }

    #[test]
    fn naive_bayes_separates_two_clusters() {
        let (ds, probe) = two_clusters();
        let mut m = NaiveBayes::new();
        m.fit(&ds).unwrap();
        assert_eq!(m.predict(&probe).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn svc_survives_drop_after_fit() {
        // Exercises the self-referential holder: the frame the model was fit on
        // is dropped, then we still predict.
        let mut m = Svc::rbf();
        {
            let (ds, _) = two_clusters();
            m.fit(&ds).unwrap();
        } // ds dropped here
        let probe = Frame::from_rows(
            vec![vec![0.2, 0.1], vec![9.2, 9.1]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(m.predict(&probe).unwrap(), vec![0.0, 1.0]);
    }
}
