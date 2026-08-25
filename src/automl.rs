//! AutoML — the framework, pointed at itself.
//!
//! Everything AutoML needs already exists: preprocessing transformers, the CV
//! engine, and the ensemble core. [`AutoML`] orchestrates them — searching
//! preprocessing × model × hyperparameters under a budget, optionally
//! auto-ensembling the top candidates — and returns the best *deployable*
//! model plus a leaderboard. No new crate; it reuses
//! [`selection`](crate::selection) and [`ensemble`](crate::ensemble).
//!
//! When the `eda` engine is enabled, the search is *seeded* from
//! [`Profile::suggest_pipeline`](crate::profile::Profile::suggest_pipeline): the
//! preprocessing is fixed to EDA's suggestion and only the model varies on top,
//! pruning the space before a single model is fit. Without `eda` it falls back
//! to searching the scaler as well.
//!
//! ```no_run
//! use millwright::prelude::*;
//! # fn main() -> millwright::Result<()> {
//! # let train: Dataset = todo!();
//! let result = AutoML::classifier()
//!     .budget(Budget::trials(40))
//!     .metric(Metric::F1)
//!     .cv(StratifiedKFold::new(5))
//!     .fit(&train)?;
//!
//! println!("{}", result.leaderboard());
//! # Ok(())
//! # }
//! ```

use crate::backends::smartcore::{Knn, NaiveBayes, Svc};
use crate::backends::smartcore::{LinearRegression, RandomForest};
use crate::ensemble::{Bagging, Boosting, EnsembleTask, Stacking, Voting};
use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
use crate::logistic::LogisticRegression;
use crate::pipeline::Pipeline;
use crate::rng::Rng;
use crate::selection::{cross_val_score, CrossValidator, KFold, Metric, StratifiedKFold};
use crate::traits::{Estimator, Model, Predictor};

/// A search budget.
#[derive(Clone, Copy, Debug)]
pub enum Budget {
    /// Evaluate at most this many candidate configurations.
    Trials(usize),
    /// Soft wall-clock search budget. The current CV evaluation is allowed to
    /// finish, so the search may overrun by one trial; final refitting is not
    /// counted because returning an unfitted winner would be unusable.
    Minutes(f64),
}

impl Budget {
    /// A trial-count budget.
    pub fn trials(n: usize) -> Self {
        Budget::Trials(n)
    }
    /// A wall-clock budget in minutes.
    pub fn minutes(m: f64) -> Self {
        Budget::Minutes(m)
    }
}

#[derive(Clone, Copy)]
enum Task {
    Classifier,
    Regressor,
}

/// Ensemble families that can compete with single pipelines during AutoML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsembleKind {
    Voting,
    Bagging,
    Boosting,
    Stacking,
}

/// Controls whether AutoML may consider models that cannot be exported to ONNX.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deployability {
    /// Search every available model family and optimize predictive score only.
    Any,
    /// Restrict the search to models that can be exported to ONNX.
    Onnx,
}

/// An automated model search.
pub struct AutoML {
    task: Task,
    budget: Budget,
    metric: Metric,
    cv: Box<dyn CrossValidator>,
    seed: u64,
    ensemble: bool,
    ensemble_size: usize,
    ensemble_kinds: Vec<EnsembleKind>,
    prefer_ensemble_on_tie: bool,
    parallel: bool,
    deployability: Deployability,
}

impl AutoML {
    /// A classification search (defaults: 40 trials, accuracy, 5-fold stratified
    /// CV, auto-ensembling on).
    pub fn classifier() -> Self {
        AutoML {
            task: Task::Classifier,
            budget: Budget::Trials(40),
            metric: Metric::Accuracy,
            cv: Box::new(StratifiedKFold::new(5)),
            seed: 0,
            ensemble: true,
            ensemble_size: 3,
            ensemble_kinds: vec![
                EnsembleKind::Voting,
                EnsembleKind::Bagging,
                EnsembleKind::Boosting,
                EnsembleKind::Stacking,
            ],
            prefer_ensemble_on_tie: false,
            parallel: false,
            deployability: if cfg!(feature = "onnx") {
                Deployability::Onnx
            } else {
                Deployability::Any
            },
        }
    }

    /// A regression search (defaults: 40 trials, R², 5-fold CV).
    pub fn regressor() -> Self {
        AutoML {
            task: Task::Regressor,
            budget: Budget::Trials(40),
            metric: Metric::R2,
            cv: Box::new(KFold::new(5)),
            seed: 0,
            ensemble: true,
            ensemble_size: 3,
            ensemble_kinds: vec![
                EnsembleKind::Voting,
                EnsembleKind::Bagging,
                EnsembleKind::Stacking,
            ],
            prefer_ensemble_on_tie: false,
            parallel: false,
            deployability: if cfg!(feature = "onnx") {
                Deployability::Onnx
            } else {
                Deployability::Any
            },
        }
    }

    /// Set the search budget.
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }
    /// Set the scoring metric.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }
    /// Set the cross-validation strategy.
    pub fn cv(mut self, cv: impl CrossValidator + 'static) -> Self {
        self.cv = Box::new(cv);
        self
    }
    /// Seed the candidate shuffling.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Disable auto-ensembling of the top candidates.
    pub fn no_ensemble(mut self) -> Self {
        self.ensemble = false;
        self
    }

    /// Configure ensemble breadth/rounds. Values below two are raised to two.
    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.ensemble_size = size.max(2);
        self
    }

    /// Choose which ensemble families compete with individual pipelines.
    pub fn ensemble_kinds(mut self, kinds: impl IntoIterator<Item = EnsembleKind>) -> Self {
        self.ensemble_kinds = kinds.into_iter().collect();
        self
    }

    /// Prefer an ensemble when it ties the best single-pipeline score.
    pub fn prefer_ensemble_on_tie(mut self) -> Self {
        self.prefer_ensemble_on_tie = true;
        self
    }

    /// Evaluate candidate configurations in parallel over rayon (each CV run is
    /// already fold-parallel; this adds parallelism across candidates). Parallel
    /// search requires a trial budget because already-running work cannot obey a
    /// strict wall-clock cutoff.
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Choose whether non-ONNX-exportable candidates may participate.
    pub fn deployability(mut self, deployability: Deployability) -> Self {
        self.deployability = deployability;
        self
    }

    /// Run the search and return the best deployable model with a leaderboard.
    pub fn fit(self, dataset: &Dataset) -> Result<AutoMLResult> {
        match self.budget {
            Budget::Trials(0) => {
                return Err(Error::Param("AutoML trial budget must be >= 1".into()))
            }
            Budget::Minutes(minutes) if !minutes.is_finite() || minutes <= 0.0 => {
                return Err(Error::Param(
                    "AutoML minute budget must be finite and > 0".into(),
                ))
            }
            Budget::Minutes(_) if self.parallel => {
                return Err(Error::Param(
                    "parallel AutoML requires a trial budget, not a minute budget".into(),
                ))
            }
            _ => {}
        }
        let mut candidates = match self.task {
            Task::Classifier => classifier_candidates(dataset, self.deployability),
            Task::Regressor => regressor_candidates(dataset),
        };
        Rng::new(self.seed).shuffle(&mut candidates);

        let start = std::time::Instant::now();
        let trial_cap = match self.budget {
            Budget::Trials(n) => n,
            Budget::Minutes(_) => usize::MAX,
        };

        let mut candidate_failures = Vec::new();
        let mut board: Vec<(String, f64, Pipeline)> = if self.parallel {
            // Candidate-level parallelism: evaluate the (capped) candidate set
            // concurrently. Each CV run is itself fold-parallel; rayon nests fine.
            use rayon::prelude::*;
            candidates
                .into_iter()
                .take(trial_cap)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(label, pipe)| {
                    let score = cross_val_score(&pipe, dataset, self.cv.as_ref(), self.metric);
                    (label, pipe, score)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|(label, pipe, score)| match score {
                    Ok(score) => Some((label, score, pipe)),
                    Err(error) => {
                        candidate_failures.push((label, error.to_string()));
                        None
                    }
                })
                .collect()
        } else {
            let mut board = Vec::new();
            for (label, pipe) in candidates.into_iter().take(trial_cap) {
                if let Budget::Minutes(m) = self.budget {
                    if start.elapsed().as_secs_f64() > m * 60.0 {
                        break;
                    }
                }
                match cross_val_score(&pipe, dataset, self.cv.as_ref(), self.metric) {
                    Ok(score) => board.push((label, score, pipe)),
                    Err(error) => candidate_failures.push((label, error.to_string())),
                }
            }
            board
        };
        if board.is_empty() {
            let details = candidate_failures
                .iter()
                .map(|(label, error)| format!("{label}: {error}"))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(Error::Pipeline(format!(
                "AutoML found no viable candidates{}",
                if details.is_empty() {
                    String::new()
                } else {
                    format!(": {details}")
                }
            )));
        }
        sort_board(&mut board, self.metric.greater_is_better());

        let ensemble_models =
            if self.ensemble && board.len() >= 2 && !budget_expired(start, self.budget) {
                build_ensembles(
                    self.task,
                    &board,
                    self.ensemble_size,
                    self.seed,
                    &self.ensemble_kinds,
                    self.cv.clone(),
                )
            } else {
                Vec::new()
            };
        let mut ensemble_entries = Vec::new();
        let mut ensemble_failures = Vec::new();
        for (label, model) in &ensemble_models {
            if budget_expired(start, self.budget) {
                break;
            }
            match cross_val_score(model.as_ref(), dataset, self.cv.as_ref(), self.metric) {
                Ok(score) => ensemble_entries.push((label.clone(), score)),
                Err(error) => ensemble_failures.push((label.clone(), error.to_string())),
            }
        }

        // The leaderboard is every single candidate plus the ensemble entry.
        let mut leaderboard: Vec<(String, f64)> =
            board.iter().map(|(l, s, _)| (l.clone(), *s)).collect();
        leaderboard.extend(ensemble_entries.iter().cloned());
        sort_pairs(&mut leaderboard, self.metric.greater_is_better());
        if self.prefer_ensemble_on_tie {
            leaderboard.sort_by(|a, b| {
                cmp(a.1, b.1, self.metric.greater_is_better()).then_with(|| {
                    b.0.starts_with("ensemble:")
                        .cmp(&a.0.starts_with("ensemble:"))
                })
            });
        }

        // Refit in leaderboard order. A candidate can pass every CV fold and
        // still fail on the full dataset, so fall back instead of discarding a
        // completed search.
        let (winner, label, score, refit_failures) =
            refit_ranked(&leaderboard, &board, &ensemble_models, dataset)?;

        Ok(AutoMLResult {
            winner,
            label,
            score,
            board: leaderboard,
            candidate_failures,
            ensemble_failures,
            refit_failures,
        })
    }
}

enum Winner {
    Single(Pipeline),
    Ensemble(Box<dyn Model>),
}

/// The outcome of an [`AutoML`] search: the fitted winner and a leaderboard.
pub struct AutoMLResult {
    winner: Winner,
    label: String,
    score: f64,
    board: Vec<(String, f64)>,
    candidate_failures: Vec<(String, String)>,
    ensemble_failures: Vec<(String, String)>,
    refit_failures: Vec<(String, String)>,
}

impl AutoMLResult {
    /// The winning configuration's label.
    pub fn best_label(&self) -> &str {
        &self.label
    }

    /// The winner's cross-validated score.
    pub fn best_score(&self) -> f64 {
        self.score
    }

    /// The best single pipeline, if a pipeline (not an ensemble) won. Use
    /// [`AutoMLResult::best_ensemble`] for an ensemble winner.
    pub fn best_pipeline(&self) -> Option<&Pipeline> {
        match &self.winner {
            Winner::Single(p) => Some(p),
            Winner::Ensemble(_) => None,
        }
    }

    /// Whether an ensemble won the search.
    pub fn is_ensemble(&self) -> bool {
        matches!(&self.winner, Winner::Ensemble(_))
    }

    /// Return the fitted ensemble winner, if one won.
    pub fn best_ensemble(&self) -> Option<&dyn Model> {
        match &self.winner {
            Winner::Ensemble(model) => Some(model.as_ref()),
            Winner::Single(_) => None,
        }
    }

    /// A ranked, printable leaderboard.
    pub fn leaderboard(&self) -> String {
        let mut out = String::from("rank  score    config\n");
        for (i, (label, score)) in self.board.iter().enumerate() {
            out.push_str(&format!("{:>4}  {score:.4}  {label}\n", i + 1));
        }
        if !self.candidate_failures.is_empty() {
            out.push_str("\nfailed candidates\n");
            for (label, error) in &self.candidate_failures {
                out.push_str(&format!("  {label}: {error}\n"));
            }
        }
        if !self.ensemble_failures.is_empty() {
            out.push_str("\nfailed ensembles\n");
            for (label, error) in &self.ensemble_failures {
                out.push_str(&format!("  {label}: {error}\n"));
            }
        }
        if !self.refit_failures.is_empty() {
            out.push_str("\nfailed full-data refits\n");
            for (label, error) in &self.refit_failures {
                out.push_str(&format!("  {label}: {error}\n"));
            }
        }
        out
    }

    /// Ranked leaderboard entries as structured `(label, score)` values.
    pub fn leaderboard_entries(&self) -> &[(String, f64)] {
        &self.board
    }

    /// Individual pipeline candidates that could not be evaluated.
    pub fn candidate_failures(&self) -> &[(String, String)] {
        &self.candidate_failures
    }

    /// Ranked candidates that passed CV but failed while refitting on all data.
    pub fn refit_failures(&self) -> &[(String, String)] {
        &self.refit_failures
    }

    /// Clone the fitted winning model for type-erased integrations.
    pub fn clone_best_model(&self) -> Box<dyn Model> {
        match &self.winner {
            Winner::Single(pipeline) => Box::new(pipeline.clone()),
            Winner::Ensemble(model) => model.clone(),
        }
    }

    /// Ensemble candidates that could not be evaluated, with their errors.
    pub fn ensemble_failures(&self) -> &[(String, String)] {
        &self.ensemble_failures
    }

    /// Export the winner to ONNX when it and all of its components support it.
    #[cfg(feature = "onnx")]
    pub fn export_onnx(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::onnx::ExportOnnx;
        match &self.winner {
            Winner::Single(p) => p.export_onnx(path),
            Winner::Ensemble(model) => {
                let proto = model.to_onnx_proto()?;
                onnx_export_rs::graph_builder::save_to_file(&proto, path)
                    .map_err(|e| Error::Backend(format!("ONNX save failed: {e}")))
            }
        }
    }
}

fn build_ensembles(
    task: Task,
    board: &[(String, f64, Pipeline)],
    requested_size: usize,
    seed: u64,
    kinds: &[EnsembleKind],
    cv: Box<dyn CrossValidator>,
) -> Vec<(String, Box<dyn Model>)> {
    let size = requested_size.max(2);
    let k = board.len().min(size);
    let mut out: Vec<(String, Box<dyn Model>)> = Vec::new();
    for kind in kinds {
        let candidate = match kind {
            EnsembleKind::Voting => Some(voting_candidate(task, board, k)),
            EnsembleKind::Bagging => Some(bagging_candidate(task, &board[0].2, size, seed)),
            EnsembleKind::Boosting => boosting_candidate(task, &board[0].2, size, seed),
            EnsembleKind::Stacking => Some(stacking_candidate(task, board, k, cv.clone())),
        };
        if let Some(candidate) = candidate {
            out.push(candidate);
        }
    }
    out
}

fn ensemble_task(task: Task) -> EnsembleTask {
    match task {
        Task::Classifier => EnsembleTask::Classification,
        Task::Regressor => EnsembleTask::Regression,
    }
}

fn voting_candidate(
    task: Task,
    board: &[(String, f64, Pipeline)],
    k: usize,
) -> (String, Box<dyn Model>) {
    let soft = matches!(task, Task::Classifier)
        && board.iter().take(k).all(|(_, _, p)| p.supports_proba());
    let mut model = if soft { Voting::soft() } else { Voting::hard() }.task(ensemble_task(task));
    for (i, (_, _, pipeline)) in board.iter().take(k).enumerate() {
        model = model.add(format!("c{i}"), pipeline.clone());
    }
    (
        format!(
            "ensemble:voting-{}(top-{k})",
            if soft { "soft" } else { "hard" }
        ),
        Box::new(model),
    )
}

fn bagging_candidate(
    task: Task,
    base: &Pipeline,
    size: usize,
    seed: u64,
) -> (String, Box<dyn Model>) {
    (
        format!("ensemble:bagging(n={size})"),
        Box::new(
            Bagging::of(base.clone())
                .n_estimators(size)
                .seed(seed)
                .task(ensemble_task(task)),
        ),
    )
}

fn boosting_candidate(
    task: Task,
    base: &Pipeline,
    size: usize,
    seed: u64,
) -> Option<(String, Box<dyn Model>)> {
    matches!(task, Task::Classifier).then(|| {
        (
            format!("ensemble:boosting(n={size})"),
            Box::new(Boosting::of(base.clone()).n_estimators(size).seed(seed)) as Box<dyn Model>,
        )
    })
}

fn stacking_candidate(
    task: Task,
    board: &[(String, f64, Pipeline)],
    k: usize,
    cv: Box<dyn CrossValidator>,
) -> (String, Box<dyn Model>) {
    let mut model = match task {
        Task::Classifier => Stacking::meta(LogisticRegression::new()),
        Task::Regressor => Stacking::meta(LinearRegression::new()),
    };
    for (i, (_, _, pipeline)) in board.iter().take(k).enumerate() {
        model = model.base(format!("c{i}"), pipeline.clone());
    }
    (
        format!("ensemble:stacking(top-{k})"),
        Box::new(model.boxed_cv(cv)),
    )
}

fn budget_expired(start: std::time::Instant, budget: Budget) -> bool {
    matches!(budget, Budget::Minutes(minutes) if start.elapsed().as_secs_f64() >= minutes * 60.0)
}

type RefitOutcome = (Winner, String, f64, Vec<(String, String)>);

fn refit_ranked(
    leaderboard: &[(String, f64)],
    board: &[(String, f64, Pipeline)],
    ensembles: &[(String, Box<dyn Model>)],
    dataset: &Dataset,
) -> Result<RefitOutcome> {
    let mut failures = Vec::new();
    for (label, score) in leaderboard {
        let winner = if let Some((_, model)) = ensembles.iter().find(|(name, _)| name == label) {
            let mut model = model.clone();
            match model.fit(dataset) {
                Ok(()) => Some(Winner::Ensemble(model)),
                Err(error) => {
                    failures.push((label.clone(), error.to_string()));
                    None
                }
            }
        } else if let Some((_, _, pipeline)) = board.iter().find(|(name, _, _)| name == label) {
            let mut pipeline = pipeline.clone();
            match pipeline.fit(dataset) {
                Ok(()) => Some(Winner::Single(pipeline)),
                Err(error) => {
                    failures.push((label.clone(), error.to_string()));
                    None
                }
            }
        } else {
            failures.push((label.clone(), "ranked candidate was not retained".into()));
            None
        };
        if let Some(winner) = winner {
            return Ok((winner, label.clone(), *score, failures));
        }
    }
    let details = failures
        .iter()
        .map(|(label, error)| format!("{label}: {error}"))
        .collect::<Vec<_>>()
        .join("; ");
    Err(Error::Pipeline(format!(
        "AutoML could not refit any ranked candidate: {details}"
    )))
}

impl Predictor for AutoMLResult {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        match &self.winner {
            Winner::Single(p) => p.predict(frame),
            Winner::Ensemble(m) => m.predict(frame),
        }
    }
}

fn sort_board(board: &mut [(String, f64, Pipeline)], greater_is_better: bool) {
    board.sort_by(|a, b| cmp(a.1, b.1, greater_is_better));
}
fn sort_pairs(pairs: &mut [(String, f64)], greater_is_better: bool) {
    pairs.sort_by(|a, b| cmp(a.1, b.1, greater_is_better));
}
fn cmp(a: f64, b: f64, greater_is_better: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    // NaN scores (degenerate folds) always sort last.
    match (a.is_nan(), b.is_nan()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }
    if greater_is_better {
        b.partial_cmp(&a).unwrap_or(Ordering::Equal)
    } else {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

fn class_count(target: &[f64]) -> usize {
    let mut classes: Vec<i64> = target.iter().map(|value| value.round() as i64).collect();
    classes.sort_unstable();
    classes.dedup();
    classes.len()
}

/// The preprocessing pipeline EDA suggests for `dataset`, when the `eda` engine
/// is available — the search then varies only the model on top of it, rather
/// than blindly trying every scaler. Returns `None` (fall back to the scaler
/// sweep) when `eda` is off or profiling fails.
#[cfg(feature = "eda")]
fn seeded_base(dataset: &Dataset) -> Option<Pipeline> {
    let table = crate::table::Table::from_frame(dataset.features()).ok()?;
    let profile = crate::profile::Profile::of(&table).ok()?;
    Some(profile.suggest_pipeline())
}

/// Preprocessing × classifier-family × hyperparameter candidates.
#[cfg_attr(not(feature = "eda"), allow(unused_variables))]
fn classifier_candidates(
    dataset: &Dataset,
    deployability: Deployability,
) -> Vec<(String, Pipeline)> {
    let mut out = Vec::new();

    // Seed the preprocessing from EDA's suggestion; vary only the model on top.
    #[cfg(feature = "eda")]
    if let Some(base) = seeded_base(dataset) {
        let prep = base.step_names().join("+");
        let prep = if prep.is_empty() { "raw".into() } else { prep };
        push_classifier_models(
            &mut out,
            &base,
            &format!("profile[{prep}]"),
            dataset,
            deployability,
        );
        return out;
    }

    // Fallback: search the scaler as well as the model.
    for scaler in ["none", "standard", "minmax"] {
        let base = preprocessing_candidate(scaler);
        push_classifier_models(&mut out, &base, scaler, dataset, deployability);
    }
    out
}

/// Add every classifier family for one preprocessing strategy. Keeping this
/// matrix in one place prevents EDA/scaler and ONNX feature paths from drifting.
fn push_classifier_models(
    out: &mut Vec<(String, Pipeline)>,
    base: &Pipeline,
    prefix: &str,
    dataset: &Dataset,
    deployability: Deployability,
) {
    let depths = [Some(2u16), Some(4), Some(8), None];
    let trees = [50u16, 100];
    for &depth in &depths {
        for &n in &trees {
            let mut rf = RandomForest::new().n_trees(n);
            if let Some(d) = depth {
                rf = rf.max_depth(d);
            }
            let depth_s = depth
                .map(|d| d.to_string())
                .unwrap_or_else(|| "none".into());
            out.push((
                format!("{prefix} | rf(trees={n}, depth={depth_s})"),
                base.clone().estimator("rf", rf),
            ));
        }
    }
    if deployability == Deployability::Any {
        for &k in &[3usize, 5, 9] {
            out.push((
                format!("{prefix} | knn(k={k})"),
                base.clone().estimator("knn", Knn::k(k)),
            ));
        }
        out.push((
            format!("{prefix} | naive_bayes"),
            base.clone().estimator("nb", NaiveBayes::new()),
        ));
        out.push((
            format!("{prefix} | svc(linear)"),
            base.clone().estimator("svc", Svc::linear()),
        ));
    }
    if class_count(dataset.target()) == 2 {
        for &l2 in &[0.0, 0.01, 0.1] {
            out.push((
                format!("{prefix} | logistic(l2={l2})"),
                base.clone()
                    .estimator("logistic", LogisticRegression::new().l2(l2)),
            ));
        }
    }
}

fn preprocessing_candidate(scaler: &str) -> Pipeline {
    use crate::transform::{MinMaxScaler, StandardScaler};
    match scaler {
        "standard" => Pipeline::new().step("scale", StandardScaler::new()),
        "minmax" => Pipeline::new().step("scale", MinMaxScaler::new()),
        _ => Pipeline::new(),
    }
}

/// Preprocessing × LinearRegression candidates.
#[cfg_attr(not(feature = "eda"), allow(unused_variables))]
fn regressor_candidates(dataset: &Dataset) -> Vec<(String, Pipeline)> {
    let mut out = Vec::new();

    #[cfg(feature = "eda")]
    if let Some(base) = seeded_base(dataset) {
        let prep = base.step_names().join("+");
        let prep = if prep.is_empty() { "raw".into() } else { prep };
        out.push((
            format!("profile[{prep}] | linear"),
            base.estimator("lr", LinearRegression::new()),
        ));
        return out;
    }

    for scaler in ["none", "standard", "minmax"] {
        let pipe = preprocessing_candidate(scaler).estimator("lr", LinearRegression::new());
        out.push((format!("{scaler} | linear"), pipe));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;

    fn two_class() -> Dataset {
        // Two clearly separable clusters (both features informative).
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..25 {
            rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
            y.push(1.0);
        }
        Dataset::new(
            Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(),
            y,
        )
        .unwrap()
    }

    #[test]
    fn classifier_search_finds_a_strong_model() {
        let ds = two_class();
        let result = AutoML::classifier()
            .budget(Budget::trials(12))
            .cv(StratifiedKFold::new(4))
            .seed(1)
            .fit(&ds)
            .unwrap();
        assert!(
            result.best_score() > 0.9,
            "score {}\n{}",
            result.best_score(),
            result.leaderboard()
        );
        assert!(!result.leaderboard().is_empty());

        let probe = Frame::from_rows(
            vec![vec![0.1, 0.1], vec![9.2, 9.2]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let preds = result.predict(&probe).unwrap();
        assert_eq!(preds.len(), 2);
        assert!(
            preds[0] < preds[1],
            "clusters should separate, got {preds:?} (winner: {})",
            result.best_label()
        );
    }

    #[test]
    fn ensemble_winner_branch_is_explicit_and_inspectable() {
        let result = AutoML::classifier()
            .budget(Budget::trials(12))
            .cv(StratifiedKFold::new(4))
            .seed(1)
            .ensemble_size(4)
            .ensemble_kinds([EnsembleKind::Voting])
            .prefer_ensemble_on_tie()
            .fit(&two_class())
            .unwrap();

        assert!(result.is_ensemble(), "winner: {}", result.best_label());
        assert!(result.best_ensemble().is_some());
        assert!(result.best_pipeline().is_none());
        assert!(result.best_label().starts_with("ensemble:voting-"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn ensemble_winner_exports_to_onnx() {
        let dataset = two_class();
        let mut voting = Voting::soft()
            .add("lr", LogisticRegression::new())
            .add("lr_l2", LogisticRegression::new().l2(0.01));
        voting.fit(&dataset).unwrap();
        let result = AutoMLResult {
            winner: Winner::Ensemble(Box::new(voting)),
            label: "ensemble:voting-soft(top-2)".into(),
            score: 1.0,
            board: vec![("ensemble:voting-soft(top-2)".into(), 1.0)],
            candidate_failures: Vec::new(),
            ensemble_failures: Vec::new(),
            refit_failures: Vec::new(),
        };
        let path = std::env::temp_dir().join(format!(
            "millwright-automl-ensemble-{}.onnx",
            std::process::id()
        ));
        result.export_onnx(&path).unwrap();
        let loaded = crate::onnx::InferenceModel::load(&path).unwrap();
        let probe = Frame::from_rows(
            vec![vec![0.1, 0.1], vec![9.2, 9.2]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(
            loaded.predict(&probe).unwrap(),
            result.predict(&probe).unwrap()
        );
        std::fs::remove_file(path).ok();
    }

    #[cfg(feature = "eda")]
    #[test]
    fn classifier_search_is_seeded_from_profile() {
        let ds = two_class();
        let result = AutoML::classifier()
            .budget(Budget::trials(6))
            .cv(StratifiedKFold::new(4))
            .seed(1)
            .fit(&ds)
            .unwrap();
        // Every candidate is built on the profile's suggested preprocessing.
        assert!(
            result.best_label().starts_with("profile["),
            "label: {}",
            result.best_label()
        );
    }

    #[test]
    fn parallel_search_matches_sequential() {
        let ds = two_class();
        let run = |parallel: bool| {
            let mut a = AutoML::classifier()
                .budget(Budget::trials(8))
                .cv(StratifiedKFold::new(4))
                .seed(3);
            if parallel {
                a = a.parallel();
            }
            a.fit(&ds).unwrap()
        };
        let seq = run(false);
        let par = run(true);
        assert_eq!(seq.best_label(), par.best_label());
        assert!((seq.best_score() - par.best_score()).abs() < 1e-12);
    }

    #[test]
    fn invalid_budgets_are_rejected() {
        let ds = two_class();
        assert!(AutoML::classifier()
            .budget(Budget::trials(0))
            .fit(&ds)
            .is_err());
        assert!(AutoML::classifier()
            .budget(Budget::minutes(f64::NAN))
            .fit(&ds)
            .is_err());
        assert!(AutoML::classifier()
            .budget(Budget::minutes(1.0))
            .parallel()
            .fit(&ds)
            .is_err());
    }

    #[test]
    fn deployability_policy_controls_non_onnx_candidates() {
        let dataset = two_class();
        let any = classifier_candidates(&dataset, Deployability::Any);
        let onnx = classifier_candidates(&dataset, Deployability::Onnx);
        assert!(any.iter().any(|(label, _)| label.contains("knn(")));
        assert!(any.iter().any(|(label, _)| label.contains("naive_bayes")));
        assert!(any.iter().any(|(label, _)| label.contains("svc(")));
        assert!(!onnx.iter().any(|(label, _)| {
            label.contains("knn(") || label.contains("naive_bayes") || label.contains("svc(")
        }));
    }

    #[test]
    fn full_data_refit_falls_back_to_next_ranked_candidate() {
        let dataset = two_class();
        let bad =
            Pipeline::new().estimator("logistic", LogisticRegression::new().learning_rate(0.0));
        let good = Pipeline::new().estimator("logistic", LogisticRegression::new());
        let board = vec![
            ("bad".to_string(), 1.0, bad),
            ("good".to_string(), 0.9, good),
        ];
        let leaderboard = vec![("bad".to_string(), 1.0), ("good".to_string(), 0.9)];
        let (winner, label, score, failures) =
            refit_ranked(&leaderboard, &board, &[], &dataset).unwrap();
        assert!(matches!(winner, Winner::Single(_)));
        assert_eq!(label, "good");
        assert_eq!(score, 0.9);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "bad");
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn candidate_failures_do_not_abort_search() {
        let original = two_class();
        let labels = original
            .target()
            .iter()
            .map(|label| if *label == 0.0 { -1.0 } else { 1.0 })
            .collect();
        let dataset = Dataset::new(original.features().clone(), labels).unwrap();
        let result = AutoML::classifier()
            .budget(Budget::trials(100))
            .cv(StratifiedKFold::new(3))
            .no_ensemble()
            .fit(&dataset)
            .unwrap();
        assert!(!result.candidate_failures().is_empty());
        assert!(!result.leaderboard_entries().is_empty());
    }

    #[test]
    fn regressor_search_runs() {
        let rows: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();
        let result = AutoML::regressor().cv(KFold::new(3)).fit(&ds).unwrap();
        assert!(result.best_score() > 0.95, "r2 {}", result.best_score());
    }
}
