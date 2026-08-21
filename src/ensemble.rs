//! Ensembles — combine models, even across backends.
//!
//! Because every model is a [`Predictor`], combining them is just another
//! model that holds several — no new machinery, no new crate. These live in the
//! core and compose over the four traits, so a linfa model, a smartcore forest,
//! and any future backend can sit in one ensemble.
//!
//! - [`Voting`] — hard (majority) or soft (mean class-fraction) vote.
//! - [`Bagging`] — bootstrap-resample, fit a base estimator per sample, aggregate.
//! - [`Stacking`] — a meta-learner over the base models' leak-free out-of-fold
//!   predictions (requires the `model-selection` feature for the CV engine).
//!
//! Classification vs. regression is inferred from the training target: an
//! all-integral target is treated as class labels (aggregated by vote); any
//! other target is treated as regression (aggregated by mean).

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

/// Class-fraction probabilities: for each row, the fraction of members voting
/// each class. One column per class, in ascending class order.
fn vote_proba(members: &[Vec<f64>], n_rows: usize, classes: &[i64]) -> Result<Frame> {
    let m = members.len() as f64;
    let mut buf = Vec::with_capacity(n_rows * classes.len());
    for r in 0..n_rows {
        for &c in classes {
            let count = members.iter().filter(|mem| mem[r].round() as i64 == c).count();
            buf.push(count as f64 / m);
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
    /// Argmax of the mean class-fraction across members.
    Soft,
}

/// A voting ensemble: several fitted models, combined by (weighted) vote.
#[derive(Clone)]
pub struct Voting {
    kind: VotingKind,
    members: Vec<(String, Box<dyn Model>)>,
    classes: Option<Vec<i64>>,
    fitted: bool,
}

impl Voting {
    /// A hard-voting ensemble.
    pub fn hard() -> Self {
        Voting {
            kind: VotingKind::Hard,
            members: Vec::new(),
            classes: None,
            fitted: false,
        }
    }

    /// A soft-voting ensemble (mean class-fraction).
    pub fn soft() -> Self {
        Voting {
            kind: VotingKind::Soft,
            members: Vec::new(),
            classes: None,
            fitted: false,
        }
    }

    /// Add a named member model. Builder-style.
    pub fn add(mut self, name: impl Into<String>, model: impl Model + 'static) -> Self {
        self.members.push((name.into(), Box::new(model)));
        self
    }

    fn member_preds(&self, frame: &Frame) -> Result<Vec<Vec<f64>>> {
        self.members.iter().map(|(_, m)| m.predict(frame)).collect()
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
        self.classes = is_classification(dataset.target()).then(|| classes_of(dataset.target()));
        for (_, m) in &mut self.members {
            m.fit(dataset)?;
        }
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        route(&mut self.members, path, value)
    }
}

impl Predictor for Voting {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::NotFitted("Voting::predict".into()));
        }
        let preds = self.member_preds(frame)?;
        match (self.kind, &self.classes) {
            (VotingKind::Soft, Some(classes)) => {
                let proba = vote_proba(&preds, frame.nrows(), classes)?;
                Ok(argmax_rows(&proba, classes))
            }
            // Hard vote, or regression (mean) via `aggregate`.
            _ => Ok(aggregate(&preds, frame.nrows(), &self.classes)),
        }
    }
}

impl ProbaPredictor for Voting {
    fn predict_proba(&self, frame: &Frame) -> Result<Frame> {
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| Error::Pipeline("predict_proba requires a classification target".into()))?;
        let preds = self.member_preds(frame)?;
        vote_proba(&preds, frame.nrows(), classes)
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
}

impl Estimator for Bagging {
    fn name(&self) -> &'static str {
        "Bagging"
    }

    fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(Error::Pipeline("Bagging needs n_estimators >= 1".into()));
        }
        self.classes = is_classification(dataset.target()).then(|| classes_of(dataset.target()));
        let n = dataset.features().nrows();
        let mut rng = Rng::new(self.seed);
        let mut members = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let idx: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            let mut m = self.base.clone();
            m.fit(&dataset.select(&idx))?;
            members.push(m);
        }
        self.members = members;
        self.fitted = true;
        Ok(())
    }

    fn set_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        // "base__param" (or a bare param name) tunes the base estimator.
        let param = path.strip_prefix("base__").unwrap_or(path);
        self.base.set_param(param, value)
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
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;

    fn two_class() -> Dataset {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..15 {
            rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
            y.push(1.0);
        }
        Dataset::new(Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(), y).unwrap()
    }

    fn probe() -> Frame {
        Frame::from_rows(
            vec![vec![0.2, 0.2], vec![9.3, 9.3]],
            vec!["a".into(), "b".into()],
        )
        .unwrap()
    }

    #[test]
    fn hard_voting_predicts_clusters() {
        let mut v = Voting::hard()
            .add("rf1", RandomForest::new().n_trees(10))
            .add("rf2", RandomForest::new().n_trees(20));
        v.fit(&two_class()).unwrap();
        assert_eq!(v.predict(&probe()).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn soft_voting_exposes_probabilities() {
        let mut v = Voting::soft().add("rf", RandomForest::new().n_trees(10));
        v.fit(&two_class()).unwrap();
        let proba = v.predict_proba(&probe()).unwrap();
        assert_eq!(proba.shape(), (2, 2)); // 2 rows, 2 classes
        assert_eq!(v.predict(&probe()).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn bagging_predicts_clusters() {
        let mut b = Bagging::of(RandomForest::new().n_trees(10)).n_estimators(5).seed(1);
        b.fit(&two_class()).unwrap();
        assert_eq!(b.predict(&probe()).unwrap(), vec![0.0, 1.0]);
    }

    #[cfg(feature = "model-selection")]
    #[test]
    fn stacking_predicts_clusters() {
        use crate::selection::StratifiedKFold;
        let mut s = Stacking::meta(RandomForest::new().n_trees(10))
            .base("rf", RandomForest::new().n_trees(10))
            .base("rf2", RandomForest::new().n_trees(15))
            .cv(StratifiedKFold::new(3));
        s.fit(&two_class()).unwrap();
        assert_eq!(s.predict(&probe()).unwrap(), vec![0.0, 1.0]);
    }
}
