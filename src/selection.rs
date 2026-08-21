//! Model selection — cross-validation, scoring, and search over a pipeline.
//!
//! This module adapts [`model-selection-rs`](https://docs.rs/model-selection-rs)
//! (the CV splitters and scorers) into the framework, then layers grid and
//! random search over any [`Model`] — including a whole [`Pipeline`], tuned by
//! `"step__param"` path.
//!
//! ```no_run
//! use millwright::prelude::*;
//! use millwright::grid;
//!
//! # fn main() -> millwright::Result<()> {
//! # let train: Dataset = todo!();
//! let pipe = Pipeline::new()
//!     .step("scale", StandardScaler::new())
//!     .estimator("rf", RandomForest::new());
//!
//! let search = GridSearch::new(pipe, grid! { "rf__max_depth" => [4, 8, 16] })
//!     .cv(StratifiedKFold::new(5))
//!     .scoring(Metric::F1)
//!     .fit(&train)?;
//!
//! println!("best F1 = {:.3}", search.best_score());
//! let preds = search.predict(train.features())?;
//! # let _ = preds;
//! # Ok(())
//! # }
//! ```

use ndarray::Array1;

use model_selection_rs::scoring::smartcore_adapter::SmartcoreF1;
use model_selection_rs::scoring::{
    Accuracy as MsAccuracy, MeanAbsoluteError, MeanSquaredError, R2Score, RootMeanSquaredError,
    Scorer,
};
use model_selection_rs::splitters::{
    CvSplitter, KFold as MsKFold, StratifiedKFold as MsStratifiedKFold,
};

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::rng::Rng;
use crate::traits::{Model, ParamValue, Predictor};

fn ms_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("model-selection: {e}"))
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// A scoring metric, backed by `model-selection-rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    /// Classification accuracy (higher is better).
    Accuracy,
    /// Binary F1 (higher is better); labels encoded `0.0` / `1.0`.
    F1,
    /// Mean absolute error (lower is better).
    Mae,
    /// Mean squared error (lower is better).
    Mse,
    /// Root mean squared error (lower is better).
    Rmse,
    /// Coefficient of determination R² (higher is better).
    R2,
}

impl Metric {
    fn scorer(&self) -> Box<dyn Scorer> {
        match self {
            Metric::Accuracy => Box::new(MsAccuracy),
            Metric::F1 => Box::new(SmartcoreF1::default()),
            Metric::Mae => Box::new(MeanAbsoluteError),
            Metric::Mse => Box::new(MeanSquaredError),
            Metric::Rmse => Box::new(RootMeanSquaredError),
            Metric::R2 => Box::new(R2Score),
        }
    }

    /// Whether a larger score is an improvement.
    pub fn greater_is_better(&self) -> bool {
        self.scorer().greater_is_better()
    }

    /// Score aligned truth / prediction vectors.
    pub fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let t = Array1::from(y_true.to_vec());
        let p = Array1::from(y_pred.to_vec());
        self.scorer().score(&t, &p)
    }
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

/// A cross-validation strategy that yields `(train, test)` row-index splits for
/// a dataset. Wraps the `model-selection-rs` splitters, supplying labels
/// automatically where a splitter needs them.
pub trait CrossValidator: CrossValidatorClone {
    /// Produce the `(train_indices, test_indices)` splits for `dataset`.
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>>;
    /// The number of splits.
    fn n_splits(&self) -> usize;
}

/// Clone support for boxed cross-validators (so a stacking ensemble that owns
/// one can itself be `Clone`, and thus a [`Model`]).
pub trait CrossValidatorClone {
    /// Clone into a fresh box.
    fn clone_box(&self) -> Box<dyn CrossValidator>;
}

impl<T> CrossValidatorClone for T
where
    T: CrossValidator + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn CrossValidator> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CrossValidator> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Plain K-fold cross-validation.
#[derive(Clone, Copy, Debug)]
pub struct KFold {
    k: usize,
}

impl KFold {
    /// K-fold with `k` folds.
    pub fn new(k: usize) -> Self {
        KFold { k }
    }
}

impl CrossValidator for KFold {
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        MsKFold::new(self.k)
            .map_err(ms_err)?
            .split(dataset.features().nrows())
            .map_err(ms_err)
    }
    fn n_splits(&self) -> usize {
        self.k
    }
}

/// Stratified K-fold — preserves per-class proportions in every fold.
///
/// Labels come from the dataset's target automatically (integral-coded), so
/// this reads exactly like the design brief: `StratifiedKFold::new(5)`.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedKFold {
    k: usize,
}

impl StratifiedKFold {
    /// Stratified K-fold with `k` folds.
    pub fn new(k: usize) -> Self {
        StratifiedKFold { k }
    }
}

impl CrossValidator for StratifiedKFold {
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let labels: Array1<i64> =
            Array1::from(dataset.target().iter().map(|v| v.round() as i64).collect::<Vec<_>>());
        MsStratifiedKFold::new(self.k, &labels)
            .map_err(ms_err)?
            .split(labels.len())
            .map_err(ms_err)
    }
    fn n_splits(&self) -> usize {
        self.k
    }
}

/// Cross-validate one already-configured model, returning the mean fold score.
///
/// A fresh clone of `model` is fit on each fold's training rows and scored on
/// its test rows, so the passed-in model is left untouched.
pub fn cross_val_score(
    model: &dyn Model,
    dataset: &Dataset,
    cv: &dyn CrossValidator,
    metric: Metric,
) -> Result<f64> {
    let splits = cv.splits(dataset)?;
    if splits.is_empty() {
        return Err(Error::Pipeline("cross-validation produced no splits".into()));
    }
    let mut total = 0.0;
    for (train, test) in &splits {
        let mut m = model.clone_box();
        m.fit(&dataset.select(train))?;
        let preds = m.predict(&dataset.features().select_rows(test))?;
        let truth: Vec<f64> = test.iter().map(|&i| dataset.target()[i]).collect();
        total += metric.score(&truth, &preds);
    }
    Ok(total / splits.len() as f64)
}

// ---------------------------------------------------------------------------
// Parameter grids
// ---------------------------------------------------------------------------

/// A search space: a set of named axes, each a list of candidate values.
#[derive(Clone, Debug, Default)]
pub struct ParamGrid {
    axes: Vec<(String, Vec<ParamValue>)>,
}

impl ParamGrid {
    /// An empty grid (a search over it evaluates the base config once).
    pub fn new() -> Self {
        ParamGrid::default()
    }

    /// Add an axis: a parameter path and its candidate values.
    pub fn add(&mut self, path: impl Into<String>, values: Vec<ParamValue>) -> &mut Self {
        self.axes.push((path.into(), values));
        self
    }

    /// Builder form of [`ParamGrid::add`].
    pub fn with(mut self, path: impl Into<String>, values: Vec<ParamValue>) -> Self {
        self.add(path, values);
        self
    }

    /// Whether the grid has any axes.
    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// Every combination as a list of `(path, value)` assignments (the
    /// Cartesian product of the axes). An empty grid yields one empty combo.
    pub fn combinations(&self) -> Vec<Vec<(String, ParamValue)>> {
        let mut combos: Vec<Vec<(String, ParamValue)>> = vec![Vec::new()];
        for (path, values) in &self.axes {
            let mut next = Vec::with_capacity(combos.len() * values.len());
            for base in &combos {
                for v in values {
                    let mut c = base.clone();
                    c.push((path.clone(), v.clone()));
                    next.push(c);
                }
            }
            combos = next;
        }
        combos
    }
}

/// Build a [`ParamGrid`] with scikit-learn-style `"step__param" => [values]`.
///
/// ```
/// use millwright::grid;
/// let g = grid! {
///     "rf__max_depth" => [4, 8, 16],
///     "scale__with_mean" => [true, false],
/// };
/// assert_eq!(g.combinations().len(), 6);
/// ```
#[macro_export]
macro_rules! grid {
    ( $( $path:expr => [ $( $val:expr ),* $(,)? ] ),* $(,)? ) => {{
        let mut g = $crate::selection::ParamGrid::new();
        $(
            g.add($path, vec![ $( $crate::traits::ParamValue::from($val) ),* ]);
        )*
        g
    }};
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// The outcome of a search: the best refit model plus a leaderboard.
pub struct SearchResult {
    best_model: Box<dyn Model>,
    best_params: Vec<(String, ParamValue)>,
    best_score: f64,
    leaderboard: Vec<(Vec<(String, ParamValue)>, f64)>,
}

impl SearchResult {
    /// The best cross-validated score.
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// The winning parameter assignment.
    pub fn best_params(&self) -> &[(String, ParamValue)] {
        &self.best_params
    }

    /// Every evaluated `(params, mean_score)`, best first.
    pub fn leaderboard(&self) -> &[(Vec<(String, ParamValue)>, f64)] {
        &self.leaderboard
    }

    /// Predict with the best refit model.
    pub fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        self.best_model.predict(frame)
    }

    /// Take ownership of the best refit model, e.g. to nest it further.
    pub fn into_model(self) -> Box<dyn Model> {
        self.best_model
    }
}

impl Predictor for SearchResult {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        self.best_model.predict(frame)
    }
}

// Shared evaluation: score every combo by CV, refit the winner on all data.
fn run_search(
    template: Box<dyn Model>,
    combos: Vec<Vec<(String, ParamValue)>>,
    cv: &dyn CrossValidator,
    metric: Metric,
    dataset: &Dataset,
) -> Result<SearchResult> {
    let greater_is_better = metric.greater_is_better();
    let mut leaderboard: Vec<(Vec<(String, ParamValue)>, f64)> = Vec::with_capacity(combos.len());

    for combo in combos {
        let mut configured = template.clone();
        for (path, value) in &combo {
            configured.set_param(path, value.clone())?;
        }
        let score = cross_val_score(configured.as_ref(), dataset, cv, metric)?;
        leaderboard.push((combo, score));
    }

    leaderboard.sort_by(|a, b| {
        if greater_is_better {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let (best_params, best_score) = leaderboard
        .first()
        .cloned()
        .ok_or_else(|| Error::Pipeline("search evaluated no candidates".into()))?;

    // Refit the winner on the full dataset.
    let mut best_model = template.clone();
    for (path, value) in &best_params {
        best_model.set_param(path, value.clone())?;
    }
    best_model.fit(dataset)?;

    Ok(SearchResult {
        best_model,
        best_params,
        best_score,
        leaderboard,
    })
}

/// Exhaustive grid search over a model or pipeline.
pub struct GridSearch {
    model: Box<dyn Model>,
    grid: ParamGrid,
    cv: Box<dyn CrossValidator>,
    metric: Metric,
}

impl GridSearch {
    /// Search `model` (often a [`Pipeline`]) over `grid`. Defaults: 5-fold CV,
    /// accuracy scoring — override with [`GridSearch::cv`] / [`GridSearch::scoring`].
    pub fn new(model: impl Model + 'static, grid: ParamGrid) -> Self {
        GridSearch {
            model: Box::new(model),
            grid,
            cv: Box::new(KFold::new(5)),
            metric: Metric::Accuracy,
        }
    }

    /// Set the cross-validation strategy.
    pub fn cv(mut self, cv: impl CrossValidator + 'static) -> Self {
        self.cv = Box::new(cv);
        self
    }

    /// Set the scoring metric.
    pub fn scoring(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Run the search and refit the winner on the full dataset.
    pub fn fit(self, dataset: &Dataset) -> Result<SearchResult> {
        let combos = self.grid.combinations();
        run_search(self.model, combos, self.cv.as_ref(), self.metric, dataset)
    }
}

/// Randomized search: evaluate a random subset of the grid's combinations.
pub struct RandomSearch {
    model: Box<dyn Model>,
    grid: ParamGrid,
    cv: Box<dyn CrossValidator>,
    metric: Metric,
    n_iter: usize,
    seed: u64,
}

impl RandomSearch {
    /// Randomized search over `grid`. Defaults: 10 iterations, 5-fold CV,
    /// accuracy scoring, seed 0.
    pub fn new(model: impl Model + 'static, grid: ParamGrid) -> Self {
        RandomSearch {
            model: Box::new(model),
            grid,
            cv: Box::new(KFold::new(5)),
            metric: Metric::Accuracy,
            n_iter: 10,
            seed: 0,
        }
    }

    /// Number of random combinations to try.
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Seed the sampling RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the cross-validation strategy.
    pub fn cv(mut self, cv: impl CrossValidator + 'static) -> Self {
        self.cv = Box::new(cv);
        self
    }

    /// Set the scoring metric.
    pub fn scoring(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Run the search and refit the winner on the full dataset.
    pub fn fit(self, dataset: &Dataset) -> Result<SearchResult> {
        let mut combos = self.grid.combinations();
        let mut rng = Rng::new(self.seed);
        rng.shuffle(&mut combos);
        combos.truncate(self.n_iter.max(1));
        run_search(self.model, combos, self.cv.as_ref(), self.metric, dataset)
    }
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;
    use crate::pipeline::Pipeline;
    use crate::transform::StandardScaler;

    fn two_class_dataset() -> Dataset {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
            y.push(1.0);
        }
        let cols = vec!["a".into(), "b".into()];
        Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap()
    }

    #[test]
    fn grid_search_over_pipeline_finds_a_good_model() {
        let ds = two_class_dataset();
        let pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("rf", RandomForest::new());
        let search = GridSearch::new(pipe, crate::grid! { "rf__max_depth" => [2, 4] })
            .cv(StratifiedKFold::new(4))
            .scoring(Metric::F1)
            .fit(&ds)
            .unwrap();
        assert!(search.best_score() > 0.9, "F1 was {}", search.best_score());
        assert_eq!(search.leaderboard().len(), 2);
        // best model predicts the two clusters correctly
        let test = Frame::from_rows(
            vec![vec![0.1, 0.1], vec![9.2, 9.2]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(search.predict(&test).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn random_search_respects_n_iter() {
        let ds = two_class_dataset();
        let search = RandomSearch::new(
            RandomForest::new(),
            crate::grid! { "max_depth" => [1, 2, 4, 8] },
        )
        .n_iter(2)
        .cv(KFold::new(3))
        .scoring(Metric::Accuracy)
        .fit(&ds)
        .unwrap();
        assert_eq!(search.leaderboard().len(), 2);
    }
}
