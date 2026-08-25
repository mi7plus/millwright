//! Ensembles — combine models, even across backends.
//!
//! Because every model is a [`Predictor`], combining them is just another
//! model that holds several — no new machinery, no new crate. These live in the
//! core and compose over the four traits, so a linfa model, a smartcore forest,
//! and any future backend can sit in one ensemble.
//!
//! - [`Voting`] — hard majority or genuine probability-averaged soft vote.
//! - [`Bagging`] — bootstrap-resample, fit a base estimator per sample (in
//!   parallel over rayon), aggregate.
//! - [`Boosting`] — SAMME adaptive boosting: fit weak learners in sequence, each
//!   reweighted toward the last ensemble's mistakes, then `alpha`-weighted vote.
//! - [`Stacking`] — a meta-learner over the base models' leak-free out-of-fold
//!   predictions (requires the `model-selection` feature for the CV engine).
//!
//! Classification vs. regression can be selected explicitly with
//! [`EnsembleTask`]. The default [`EnsembleTask::Infer`] mode preserves the
//! convenience of inferring integral targets as class labels; use
//! [`EnsembleTask::Regression`] for integer-valued regression targets.
//!
//! Soft voting requires every member to expose class probabilities and averages
//! those probabilities. Use hard voting for estimators that only predict labels.

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::rng::Rng;
use crate::traits::{Estimator, Model, ParamValue, Predictor, ProbaPredictor};

fn is_classification(target: &[f64]) -> bool {
    !target.is_empty() && target.iter().all(|v| v.is_finite() && v.fract() == 0.0)
}

fn classes_of(target: &[f64]) -> Vec<i64> {
    let mut c: Vec<i64> = target.iter().map(|v| v.round() as i64).collect();
    c.sort_unstable();
    c.dedup();
    c
}

/// How an ensemble interprets its training target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnsembleTask {
    /// Infer classification from an all-integral target (backward compatible).
    #[default]
    Infer,
    /// Treat the target as class labels and aggregate with voting.
    Classification,
    /// Treat the target as a continuous response and aggregate with a mean.
    Regression,
}

fn target_classes(task: EnsembleTask, target: &[f64]) -> Option<Vec<i64>> {
    match task {
        EnsembleTask::Infer => is_classification(target).then(|| classes_of(target)),
        EnsembleTask::Classification => Some(classes_of(target)),
        EnsembleTask::Regression => None,
    }
}

/// Majority vote (classification) or mean (regression) across members' row
/// predictions. `members` is `[member][row]`.
fn aggregate(members: &[Vec<f64>], n_rows: usize, classes: &Option<Vec<i64>>) -> Vec<f64> {
    (0..n_rows)
        .map(|r| {
            let col: Vec<f64> = members.iter().map(|m| m[r]).collect();
            match classes {
                Some(cs) => majority(&col, cs),
                None => col.iter().sum::<f64>() / col.len() as f64,
            }
        })
        .collect()
}

fn majority(votes: &[f64], classes: &[i64]) -> f64 {
    let mut best = classes[0];
    let mut best_count = 0usize;
    for &c in classes {
        let count = votes.iter().filter(|v| v.round() as i64 == c).count();
        if count > best_count {
            best_count = count;
            best = c;
        }
    }
    best as f64
}

/// Average aligned member probabilities. Columns are matched by their `p<class>`
/// names, so members may expose classes in a different order.
fn mean_proba(members: &[Frame], n_rows: usize, classes: &[i64]) -> Result<Frame> {
    let m = members.len() as f64;
    let mut buf = Vec::with_capacity(n_rows * classes.len());
    for r in 0..n_rows {
        for &c in classes {
            let name = format!("p{c}");
            let mut sum = 0.0;
            for member in members {
                let col = member
                    .columns()
                    .iter()
                    .position(|column| column == &name)
                    .ok_or_else(|| {
                        Error::Shape(format!(
                            "soft-voting member has no '{name}' probability column"
                        ))
                    })?;
                sum += member.get(r, col);
            }
            buf.push(sum / m);
        }
    }
    let cols: Vec<String> = classes.iter().map(|c| format!("p{c}")).collect();
    Frame::new(buf, n_rows, classes.len(), cols)
}

// ---------------------------------------------------------------------------
// Voting
// ---------------------------------------------------------------------------

/// How a [`Voting`] ensemble combines its members.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VotingKind {
    /// Majority vote over hard predictions.
    Hard,
    /// Argmax of the mean class probability across members.
    Soft,
}

/// A voting ensemble: several fitted models, combined by (weighted) vote.
#[derive(Clone)]
pub struct Voting {
    kind: VotingKind,
    members: Vec<(String, Box<dyn Model>)>,
    classes: Option<Vec<i64>>,
    fitted: bool,
    task: EnsembleTask,
}

impl Voting {
    /// A hard-voting ensemble.
    pub fn hard() -> Self {
        Voting {
            kind: VotingKind::Hard,
            members: Vec::new(),
            classes: None,
            fitted: false,
            task: EnsembleTask::Infer,
        }
    }

    /// A soft-voting ensemble (mean class probabilities).
    pub fn soft() -> Self {
        Voting {
            kind: VotingKind::Soft,
            members: Vec::new(),
            classes: None,
            fitted: false,
            task: EnsembleTask::Infer,
        }
    }

    /// Add a named member model. Builder-style.
    pub fn add(mut self, name: impl Into<String>, model: impl Model + 'static) -> Self {
        self.members.push((name.into(), Box::new(model)));
        self
    }

    /// Select classification or regression semantics explicitly.
    pub fn task(mut self, task: EnsembleTask) -> Self {
        self.task = task;
        self
    }

    fn member_preds(&self, frame: &Frame) -> Result<Vec<Vec<f64>>> {
        self.members.iter().map(|(_, m)| m.predict(frame)).collect()
    }

    fn member_probas(&self, frame: &Frame) -> Result<Vec<Frame>> {
        self.members
            .iter()
            .map(|(name, model)| {
                if !model.supports_proba() {
                    return Err(Error::Pipeline(format!(
                        "soft-voting member '{name}' ({}) does not support probabilities",
                        model.name()
                    )));
                }
                model.predict_proba_dyn(frame)
            })
            .collect()
    }
}

impl Estimator for Voting {
    fn name(&self) -> &'static str {
        "Voting"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        if self.members.is_empty() {
            return Err(Error::Pipeline("Voting has no members".into()));
        }
        self.classes = target_classes(self.task, dataset.target());
        if self.kind == VotingKind::Soft && self.classes.is_none() {
            return Err(Error::Pipeline(
                "soft voting is only available for classification; use hard voting for regression"
                    .into(),
            ));
        }
        for (_, m) in &mut self.members {
            m.fit(dataset)?;
        }
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        route(&mut self.members, path, value)
    }

    fn supports_proba(&self) -> bool {
        self.kind == VotingKind::Soft && self.members.iter().all(|(_, m)| m.supports_proba())
    }

    fn predict_proba_dyn(&self, frame: &Frame) -> Result<Frame> {
        <Self as ProbaPredictor>::predict_proba(self, frame)
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        if !self.fitted {
            return Err(Error::NotFitted("Voting::to_onnx".into()));
        }
        let classes = self.classes.as_deref();
        let weights = vec![1.0; self.members.len()];
        let protos = self
            .members
            .iter()
            .map(|(_, member)| match self.kind {
                VotingKind::Soft => member.to_onnx_proba_proto(),
                VotingKind::Hard => member.to_onnx_proto(),
            })
            .collect::<Result<Vec<_>>>()?;
        let aggregation = match (self.kind, classes) {
            (VotingKind::Soft, Some(classes)) => crate::onnx::EnsembleAggregation::SoftVote {
                classes,
                weights: &weights,
            },
            (VotingKind::Hard, Some(classes)) => crate::onnx::EnsembleAggregation::HardVote {
                classes,
                weights: &weights,
            },
            (_, None) => crate::onnx::EnsembleAggregation::Mean,
        };
        crate::onnx::combine_onnx(protos, aggregation)
    }
}

impl Predictor for Voting {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Voting::predict".into()));
        }
        match (self.kind, &self.classes) {
            (VotingKind::Soft, Some(classes)) => {
                let proba = mean_proba(&self.member_probas(frame)?, frame.nrows(), classes)?;
                Ok(argmax_rows(&proba, classes))
            }
            // Hard vote, or regression (mean) via `aggregate`.
            _ => Ok(aggregate(
                &self.member_preds(frame)?,
                frame.nrows(),
                &self.classes,
            )),
        }
    }
}

impl ProbaPredictor for Voting {
    fn predict_proba(&self, frame: &Frame) -> Result<Frame> {
        let classes = self.classes.as_ref().ok_or_else(|| {
            Error::Pipeline("predict_proba requires a classification target".into())
        })?;
        match self.kind {
            VotingKind::Soft => mean_proba(&self.member_probas(frame)?, frame.nrows(), classes),
            VotingKind::Hard => Err(Error::Pipeline(
                "predict_proba is only defined for soft voting".into(),
            )),
        }
    }
}

fn argmax_rows(proba: &Frame, classes: &[i64]) -> Vec<f64> {
    (0..proba.nrows())
        .map(|r| {
            let mut best = 0usize;
            for c in 1..proba.ncols() {
                if proba.get(r, c) > proba.get(r, best) {
                    best = c;
                }
            }
            classes[best] as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bagging
// ---------------------------------------------------------------------------

/// Bootstrap aggregating: fit `n` clones of a base estimator on bootstrap
/// resamples, then aggregate their predictions. Works for any estimator.
#[derive(Clone)]
pub struct Bagging {
    base: Box<dyn Model>,
    n_estimators: usize,
    seed: u64,
    members: Vec<Box<dyn Model>>,
    classes: Option<Vec<i64>>,
    fitted: bool,
    task: EnsembleTask,
}

impl Bagging {
    /// Bag the given base estimator (10 estimators by default).
    pub fn of(base: impl Model + 'static) -> Self {
        Bagging {
            base: Box::new(base),
            n_estimators: 10,
            seed: 0,
            members: Vec::new(),
            classes: None,
            fitted: false,
            task: EnsembleTask::Infer,
        }
    }

    /// Number of bootstrap estimators.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Seed the bootstrap RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Select classification or regression aggregation explicitly.
    pub fn task(mut self, task: EnsembleTask) -> Self {
        self.task = task;
        self
    }
}

impl Estimator for Bagging {
    fn name(&self) -> &'static str {
        "Bagging"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(Error::Pipeline("Bagging needs n_estimators >= 1".into()));
        }
        self.classes = target_classes(self.task, dataset.target());
        let n = dataset.features().nrows();

        // Draw the bootstrap index sets sequentially (deterministic RNG), then
        // fit the base estimator on each in parallel over rayon. `par_iter`
        // preserves order, so the fitted members are seed-reproducible.
        let mut rng = Rng::new(self.seed);
        let bootstraps: Vec<Vec<usize>> = (0..self.n_estimators)
            .map(|_| (0..n).map(|_| rng.below(n)).collect())
            .collect();

        use rayon::prelude::*;
        self.members = bootstraps
            .par_iter()
            .map(|idx| {
                let mut m = self.base.clone();
                m.fit(&dataset.select(idx))?;
                Ok(m)
            })
            .collect::<Result<Vec<_>>>()?;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        // "base__param" (or a bare param name) tunes the base estimator.
        let param = path.strip_prefix("base__").unwrap_or(path);
        self.base.set_param(param, value)
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        if !self.fitted {
            return Err(Error::NotFitted("Bagging::to_onnx".into()));
        }
        let protos = self
            .members
            .iter()
            .map(|member| member.to_onnx_proto())
            .collect::<Result<Vec<_>>>()?;
        let weights = vec![1.0; self.members.len()];
        let aggregation = match self.classes.as_deref() {
            Some(classes) => crate::onnx::EnsembleAggregation::HardVote {
                classes,
                weights: &weights,
            },
            None => crate::onnx::EnsembleAggregation::Mean,
        };
        crate::onnx::combine_onnx(protos, aggregation)
    }
}

impl Predictor for Bagging {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Bagging::predict".into()));
        }
        let preds: Vec<Vec<f64>> = self
            .members
            .iter()
            .map(|m| m.predict(frame))
            .collect::<Result<_>>()?;
        Ok(aggregate(&preds, frame.nrows(), &self.classes))
    }
}

// ---------------------------------------------------------------------------
// Boosting (SAMME AdaBoost for classification)
// ---------------------------------------------------------------------------

/// A weighted bootstrap: draw `weights.len()` indices with probability
/// proportional to `weights`. Lets a base estimator that takes no sample
/// weights still be trained on a reweighted distribution.
fn weighted_bootstrap(weights: &[f64], rng: &mut Rng) -> Vec<usize> {
    let n = weights.len();
    let mut cum = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        cum.push(acc);
    }
    let total = acc.max(f64::MIN_POSITIVE);
    (0..n)
        .map(|_| {
            let u = (rng.next_u64() as f64 / u64::MAX as f64) * total;
            match cum.binary_search_by(|c| c.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)) {
                Ok(i) | Err(i) => i.min(n - 1),
            }
        })
        .collect()
}

/// Adaptive boosting (SAMME) for classification: fit a sequence of weak
/// learners, each on a distribution reweighted toward the previous ensemble's
/// mistakes, then take an `alpha`-weighted vote. Works for any classifier;
/// shallow trees (`RandomForest::new().max_depth(1)`) make classic stumps.
#[derive(Clone)]
pub struct Boosting {
    base: Box<dyn Model>,
    n_estimators: usize,
    learning_rate: f64,
    seed: u64,
    members: Vec<(Box<dyn Model>, f64)>,
    classes: Vec<i64>,
    fitted: bool,
}

impl Boosting {
    /// Boost the given base estimator (50 rounds by default).
    pub fn of(base: impl Model + 'static) -> Self {
        Boosting {
            base: Box::new(base),
            n_estimators: 50,
            learning_rate: 1.0,
            seed: 0,
            members: Vec::new(),
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Number of boosting rounds.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Shrink each round's contribution (`< 1.0` regularizes).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Seed the weighted-resampling RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Estimator for Boosting {
    fn name(&self) -> &'static str {
        "Boosting"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(Error::Pipeline("Boosting needs n_estimators >= 1".into()));
        }
        let y: Vec<i64> = dataset.target().iter().map(|v| v.round() as i64).collect();
        let n = y.len();
        self.classes = classes_of(dataset.target());
        let k = self.classes.len();
        if k < 2 {
            return Err(Error::Pipeline("Boosting needs >= 2 classes".into()));
        }

        let x = dataset.features();
        let mut w = vec![1.0 / n as f64; n];
        let mut rng = Rng::new(self.seed);
        let mut members: Vec<(Box<dyn Model>, f64)> = Vec::new();

        for _ in 0..self.n_estimators {
            let idx = weighted_bootstrap(&w, &mut rng);
            let mut m = self.base.clone();
            m.fit(&dataset.select(&idx))?;
            let preds = m.predict(x)?;
            let miss: Vec<bool> = preds
                .iter()
                .zip(&y)
                .map(|(p, t)| p.round() as i64 != *t)
                .collect();

            let wsum: f64 = w.iter().sum();
            let err = (w
                .iter()
                .zip(&miss)
                .filter(|(_, &m)| m)
                .map(|(wi, _)| *wi)
                .sum::<f64>()
                / wsum)
                .clamp(1e-10, 1.0 - 1e-10);

            // A perfect round: keep it with a dominant weight and stop.
            if err <= 1e-10 {
                members.push((m, 1.0));
                break;
            }
            // SAMME: worse than random for K classes — stop boosting.
            if err >= 1.0 - 1.0 / k as f64 {
                break;
            }

            let alpha = self.learning_rate * (((1.0 - err) / err).ln() + (k as f64 - 1.0).ln());
            for i in 0..n {
                if miss[i] {
                    w[i] *= alpha.exp();
                }
            }
            let sum: f64 = w.iter().sum();
            for wi in &mut w {
                *wi /= sum;
            }
            members.push((m, alpha));
        }

        if members.is_empty() {
            return Err(Error::Pipeline(
                "Boosting produced no usable weak learners (base worse than chance)".into(),
            ));
        }
        self.members = members;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        let param = path.strip_prefix("base__").unwrap_or(path);
        self.base.set_param(param, value)
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        if !self.fitted {
            return Err(Error::NotFitted("Boosting::to_onnx".into()));
        }
        let protos = self
            .members
            .iter()
            .map(|(member, _)| member.to_onnx_proto())
            .collect::<Result<Vec<_>>>()?;
        let weights: Vec<f64> = self.members.iter().map(|(_, weight)| *weight).collect();
        crate::onnx::combine_onnx(
            protos,
            crate::onnx::EnsembleAggregation::HardVote {
                classes: &self.classes,
                weights: &weights,
            },
        )
    }
}

impl Predictor for Boosting {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Boosting::predict".into()));
        }
        let n = frame.nrows();
        let k = self.classes.len();
        let mut scores = vec![vec![0.0f64; k]; n];
        for (m, alpha) in &self.members {
            let preds = m.predict(frame)?;
            for (r, p) in preds.iter().enumerate() {
                let cls = p.round() as i64;
                if let Some(ci) = self.classes.iter().position(|c| *c == cls) {
                    scores[r][ci] += *alpha;
                }
            }
        }
        Ok(scores
            .iter()
            .map(|row| {
                let mut best = 0;
                for c in 1..k {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                self.classes[best] as f64
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Stacking (needs the model-selection CV engine)
// ---------------------------------------------------------------------------

/// A stacking ensemble: a meta-learner trained on the base models' leak-free
/// out-of-fold predictions. The out-of-fold folds come from the same
/// `model-selection` CV engine used everywhere else, so the meta-learner never
/// sees a base model's own training rows.
#[cfg(feature = "model-selection")]
#[derive(Clone)]
pub struct Stacking {
    bases: Vec<(String, Box<dyn Model>)>,
    meta: Box<dyn Model>,
    cv: Box<dyn crate::selection::CrossValidator>,
    fitted_bases: Vec<Box<dyn Model>>,
    fitted: bool,
}

#[cfg(feature = "model-selection")]
impl Stacking {
    /// A stacking ensemble with the given meta-learner. Defaults to 5-fold CV
    /// for generating out-of-fold meta-features.
    pub fn meta(meta: impl Model + 'static) -> Self {
        Stacking {
            bases: Vec::new(),
            meta: Box::new(meta),
            cv: Box::new(crate::selection::KFold::new(5)),
            fitted_bases: Vec::new(),
            fitted: false,
        }
    }

    /// Add a named base model. Builder-style.
    pub fn base(mut self, name: impl Into<String>, model: impl Model + 'static) -> Self {
        self.bases.push((name.into(), Box::new(model)));
        self
    }

    /// Set the CV strategy used to build out-of-fold meta-features.
    pub fn cv(mut self, cv: impl crate::selection::CrossValidator + 'static) -> Self {
        self.cv = Box::new(cv);
        self
    }

    #[cfg(feature = "automl")]
    pub(crate) fn boxed_cv(mut self, cv: Box<dyn crate::selection::CrossValidator>) -> Self {
        self.cv = cv;
        self
    }

    /// Build the `(n_rows, n_bases)` meta-feature frame from fitted-base
    /// predictions on `frame`.
    fn meta_features(&self, frame: &Frame) -> Result<Frame> {
        let cols: Vec<String> = self.bases.iter().map(|(n, _)| n.clone()).collect();
        let n = frame.nrows();
        let mut buf = vec![0.0; n * self.fitted_bases.len()];
        for (j, m) in self.fitted_bases.iter().enumerate() {
            let preds = m.predict(frame)?;
            for (r, p) in preds.into_iter().enumerate() {
                buf[r * self.fitted_bases.len() + j] = p;
            }
        }
        Frame::new(buf, n, cols.len(), cols)
    }
}

#[cfg(feature = "model-selection")]
impl Estimator for Stacking {
    fn name(&self) -> &'static str {
        "Stacking"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        if self.bases.is_empty() {
            return Err(Error::Pipeline("Stacking has no base models".into()));
        }
        let n = dataset.features().nrows();
        let splits = self.cv.splits(dataset)?;

        // Out-of-fold predictions: one meta-feature column per base model.
        let n_bases = self.bases.len();
        let mut oof = vec![0.0; n * n_bases];
        for (j, (_, base)) in self.bases.iter().enumerate() {
            for (train, test) in &splits {
                let mut m = base.clone();
                m.fit(&dataset.select(train))?;
                let preds = m.predict(&dataset.features().select_rows(test))?;
                for (&row, p) in test.iter().zip(preds) {
                    oof[row * n_bases + j] = p;
                }
            }
        }
        let cols: Vec<String> = self.bases.iter().map(|(nm, _)| nm.clone()).collect();
        let meta_frame = Frame::new(oof, n, n_bases, cols)?;
        let meta_ds = Dataset::new(meta_frame, dataset.target().to_vec())?;
        self.meta.fit(&meta_ds)?;

        // Refit each base on the full data for inference.
        let mut fitted_bases = Vec::with_capacity(n_bases);
        for (_, base) in &self.bases {
            let mut m = base.clone();
            m.fit(dataset)?;
            fitted_bases.push(m);
        }
        self.fitted_bases = fitted_bases;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        if let Some(param) = path.strip_prefix("meta__") {
            return self.meta.set_param(param, value);
        }
        route(&mut self.bases, path, value)
    }

    #[cfg(feature = "onnx")]
    fn to_onnx_proto(&self) -> Result<onnx_export_rs::proto::ModelProto> {
        if !self.fitted {
            return Err(Error::NotFitted("Stacking::to_onnx".into()));
        }
        let bases = self
            .fitted_bases
            .iter()
            .map(|base| base.to_onnx_proto())
            .collect::<Result<Vec<_>>>()?;
        crate::onnx::stack_onnx(bases, self.meta.to_onnx_proto()?)
    }
}

#[cfg(feature = "model-selection")]
impl Predictor for Stacking {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Stacking::predict".into()));
        }
        let meta_frame = self.meta_features(frame)?;
        self.meta.predict(&meta_frame)
    }
}

/// Route a `"name__param"` path to the named member in `members`.
fn route(members: &mut [(String, Box<dyn Model>)], path: &str, value: ParamValue) -> Result<()> {
    let (name, rest) = path
        .split_once("__")
        .ok_or_else(|| Error::Param(format!("'{path}' is not a 'member__param' path")))?;
    for (n, m) in members.iter_mut() {
        if n == name {
            return m.set_param(rest, value);
        }
    }
    Err(Error::Param(format!("no member named '{name}'")))
}

#[cfg(all(test, feature = "smartcore-backend"))]
#[path = "ensemble_tests.rs"]
mod tests;
