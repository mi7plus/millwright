//! Parameter grids and the search strategies (grid, random, and Bayesian).

use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::rng::Rng;
use crate::traits::{Model, ParamValue, Predictor};

use super::cv::{cross_val_score, CrossValidator, KFold};
use super::scoring::Metric;

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

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// The outcome of a search: the best model (refit on the full data), its
/// parameters and score, and the full leaderboard.
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

// Shared evaluation: score every combo by CV, then finalize.
fn run_search(
    template: Box<dyn Model>,
    combos: Vec<Vec<(String, ParamValue)>>,
    cv: &dyn CrossValidator,
    metric: Metric,
    dataset: &Dataset,
) -> Result<SearchResult> {
    let mut leaderboard: Vec<(Vec<(String, ParamValue)>, f64)> = Vec::with_capacity(combos.len());
    for combo in combos {
        let mut configured = template.clone();
        for (path, value) in &combo {
            configured.set_param(path, value.clone())?;
        }
        let score = cross_val_score(configured.as_ref(), dataset, cv, metric)?;
        leaderboard.push((combo, score));
    }
    finalize_search(
        template.as_ref(),
        leaderboard,
        metric.greater_is_better(),
        dataset,
    )
}

/// Sort a leaderboard, pick the winner, and refit it on the full dataset.
/// Shared by grid, random, and (behind `hpo`) Bayesian search.
fn finalize_search(
    template: &dyn Model,
    mut leaderboard: Vec<(Vec<(String, ParamValue)>, f64)>,
    greater_is_better: bool,
    dataset: &Dataset,
) -> Result<SearchResult> {
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

    let mut best_model = template.clone_box();
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
    /// Search `model` (often a [`Pipeline`](crate::pipeline::Pipeline)) over
    /// `grid`. Defaults: 5-fold CV, accuracy scoring — override with
    /// [`GridSearch::cv`] / [`GridSearch::scoring`].
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

// ---------------------------------------------------------------------------
// Bayesian / TPE search (behind the `hpo` feature) — same search API
// ---------------------------------------------------------------------------

/// One parameter's distribution in a [`SearchSpace`].
#[cfg(feature = "hpo")]
#[derive(Clone, Debug)]
enum Dist {
    Int(i64, i64),
    Float(f64, f64),
    Choice(Vec<ParamValue>),
}

/// A continuous/discrete search space for [`BayesSearch`].
///
/// Unlike a [`ParamGrid`] (a fixed set of points), a space describes *ranges*
/// the TPE sampler explores adaptively.
#[cfg(feature = "hpo")]
#[derive(Clone, Debug, Default)]
pub struct SearchSpace {
    params: Vec<(String, Dist)>,
}

#[cfg(feature = "hpo")]
impl SearchSpace {
    /// An empty space.
    pub fn new() -> Self {
        SearchSpace::default()
    }

    /// An integer parameter in `[low, high]` (inclusive).
    pub fn int(mut self, path: impl Into<String>, low: i64, high: i64) -> Self {
        self.params.push((path.into(), Dist::Int(low, high)));
        self
    }

    /// A float parameter in `[low, high]`.
    pub fn float(mut self, path: impl Into<String>, low: f64, high: f64) -> Self {
        self.params.push((path.into(), Dist::Float(low, high)));
        self
    }

    /// A categorical parameter chosen from `values`.
    pub fn choice(mut self, path: impl Into<String>, values: Vec<ParamValue>) -> Self {
        self.params.push((path.into(), Dist::Choice(values)));
        self
    }
}

/// Bayesian hyperparameter search via `hyperopt-rs`'s TPE sampler.
///
/// Returns the same [`SearchResult`] as [`GridSearch`] / [`RandomSearch`], so a
/// tuned model flows into the rest of the lifecycle identically — only the
/// search strategy differs.
#[cfg(feature = "hpo")]
pub struct BayesSearch {
    model: Box<dyn Model>,
    space: SearchSpace,
    cv: Box<dyn CrossValidator>,
    metric: Metric,
    n_trials: usize,
    seed: u64,
}

#[cfg(feature = "hpo")]
impl BayesSearch {
    /// TPE search over `space`. Defaults: 25 trials, 5-fold CV, accuracy, seed 0.
    pub fn new(model: impl Model + 'static, space: SearchSpace) -> Self {
        BayesSearch {
            model: Box::new(model),
            space,
            cv: Box::new(KFold::new(5)),
            metric: Metric::Accuracy,
            n_trials: 25,
            seed: 0,
        }
    }

    /// Number of trials the sampler runs.
    pub fn n_trials(mut self, n: usize) -> Self {
        self.n_trials = n;
        self
    }

    /// Seed the TPE sampler for reproducibility.
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
        use hyperopt_rs::{Direction, StudyBuilder, TpeSampler};

        let direction = if self.metric.greater_is_better() {
            Direction::Maximize
        } else {
            Direction::Minimize
        };
        let study = StudyBuilder::new("millwright")
            .direction(direction)
            .sampler(TpeSampler::seeded(self.seed))
            .build()
            .map_err(|e| Error::Backend(format!("hyperopt study: {e}")))?;

        let mut leaderboard: Vec<(Vec<(String, ParamValue)>, f64)> = Vec::new();
        {
            let objective = |ctx: &mut hyperopt_rs::TrialContext| -> hyperopt_rs::ObjectiveResult {
                let mut combo = Vec::with_capacity(self.space.params.len());
                for (path, dist) in &self.space.params {
                    let value = match dist {
                        Dist::Int(lo, hi) => ParamValue::Int(ctx.suggest_int(path, *lo, *hi)),
                        Dist::Float(lo, hi) => ParamValue::Float(ctx.suggest_float(path, *lo, *hi)),
                        Dist::Choice(choices) => {
                            let idx = ctx.suggest_int(path, 0, (choices.len() - 1) as i64) as usize;
                            choices[idx].clone()
                        }
                    };
                    combo.push((path.clone(), value));
                }
                let mut m = self.model.clone();
                for (p, v) in &combo {
                    m.set_param(p, v.clone())?;
                }
                let score = cross_val_score(m.as_ref(), dataset, self.cv.as_ref(), self.metric)?;
                leaderboard.push((combo, score));
                Ok(score)
            };
            study
                .optimize(objective, self.n_trials)
                .map_err(|e| Error::Backend(format!("hyperopt optimize: {e}")))?;
        }

        finalize_search(
            self.model.as_ref(),
            leaderboard,
            self.metric.greater_is_better(),
            dataset,
        )
    }
}
