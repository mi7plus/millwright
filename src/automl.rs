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

use crate::backends::smartcore::{Knn, LinearRegression, NaiveBayes, RandomForest, Svc};
use crate::ensemble::{Bagging, Boosting, Stacking, Voting};
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
    /// Search until this many minutes of wall-clock time elapse.
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
    /// already fold-parallel; this adds parallelism across candidates). Uses the
    /// trials cap and ignores the wall-clock budget's early cutoff.
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Run the search and return the best deployable model with a leaderboard.
    pub fn fit(self, dataset: &Dataset) -> Result<AutoMLResult> {
        let mut candidates = match self.task {
            Task::Classifier => classifier_candidates(dataset),
            Task::Regressor => regressor_candidates(dataset),
        };
        Rng::new(self.seed).shuffle(&mut candidates);

        let start = std::time::Instant::now();
        let trial_cap = match self.budget {
            Budget::Trials(n) => n,
            Budget::Minutes(_) => usize::MAX,
        };

        let mut board: Vec<(String, f64, Pipeline)> = if self.parallel {
            // Candidate-level parallelism: evaluate the (capped) candidate set
            // concurrently. Each CV run is itself fold-parallel; rayon nests fine.
            use rayon::prelude::*;
            candidates
                .into_iter()
                .take(trial_cap)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(label, pipe)| -> Result<(String, f64, Pipeline)> {
                    let score = cross_val_score(&pipe, dataset, self.cv.as_ref(), self.metric)?;
                    Ok((label, score, pipe))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let mut board = Vec::new();
            for (label, pipe) in candidates {
                if board.len() >= trial_cap {
                    break;
                }
                if let Budget::Minutes(m) = self.budget {
                    if start.elapsed().as_secs_f64() > m * 60.0 {
                        break;
                    }
                }
                let score = cross_val_score(&pipe, dataset, self.cv.as_ref(), self.metric)?;
                board.push((label, score, pipe));
            }
            board
        };
        if board.is_empty() {
            return Err(Error::Pipeline("AutoML evaluated no candidates".into()));
        }
        sort_board(&mut board, self.metric.greater_is_better());

        let ensemble_models = if self.ensemble && board.len() >= 2 {
            build_ensembles(
                self.task,
                &board,
                self.ensemble_size,
                self.seed,
                &self.ensemble_kinds,
            )
        } else {
            Vec::new()
        };
        let ensemble_entries: Vec<(String, f64)> = ensemble_models
            .iter()
            .filter_map(|(label, model)| {
                cross_val_score(model.as_ref(), dataset, self.cv.as_ref(), self.metric)
                    .ok()
                    .map(|score| (label.clone(), score))
            })
            .collect();

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

        // Decide and refit the winner on the full dataset.
        let ensemble_wins = ensemble_entries
            .iter()
            .any(|(label, score)| label == &leaderboard[0].0 && *score == leaderboard[0].1);

        let (winner, label, score) = if ensemble_wins {
            let mut model = ensemble_models
                .into_iter()
                .find(|(label, _)| label == &leaderboard[0].0)
                .expect("winning ensemble must exist")
                .1;
            model.fit(dataset)?;
            (
                Winner::Ensemble(model),
                leaderboard[0].0.clone(),
                leaderboard[0].1,
            )
        } else {
            let (best_label, best_score, best_pipe) = &board[0];
            let mut pipe = best_pipe.clone();
            pipe.fit(dataset)?;
            (Winner::Single(pipe), best_label.clone(), *best_score)
        };

        Ok(AutoMLResult {
            winner,
            label,
            score,
            board: leaderboard,
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
        out
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
) -> Vec<(String, Box<dyn Model>)> {
    let size = requested_size.max(2);
    let k = board.len().min(size);
    let mut out: Vec<(String, Box<dyn Model>)> = Vec::new();
    for kind in kinds {
        match kind {
            EnsembleKind::Voting => {
                let soft = matches!(task, Task::Classifier)
                    && board.iter().take(k).all(|(_, _, p)| p.supports_proba());
                let mut model = if soft { Voting::soft() } else { Voting::hard() };
                for (i, (_, _, pipeline)) in board.iter().take(k).enumerate() {
                    model = model.add(format!("c{i}"), pipeline.clone());
                }
                out.push((
                    format!(
                        "ensemble:voting-{}(top-{k})",
                        if soft { "soft" } else { "hard" }
                    ),
                    Box::new(model),
                ));
            }
            EnsembleKind::Bagging => out.push((
                format!("ensemble:bagging(n={size})"),
                Box::new(
                    Bagging::of(board[0].2.clone())
                        .n_estimators(size)
                        .seed(seed),
                ),
            )),
            EnsembleKind::Boosting if matches!(task, Task::Classifier) => out.push((
                format!("ensemble:boosting(n={size})"),
                Box::new(
                    Boosting::of(board[0].2.clone())
                        .n_estimators(size)
                        .seed(seed),
                ),
            )),
            EnsembleKind::Stacking => {
                let mut model = match task {
                    Task::Classifier => Stacking::meta(LogisticRegression::new()),
                    Task::Regressor => Stacking::meta(LinearRegression::new()),
                };
                for (i, (_, _, pipeline)) in board.iter().take(k).enumerate() {
                    model = model.base(format!("c{i}"), pipeline.clone());
                }
                out.push((format!("ensemble:stacking(top-{k})"), Box::new(model)));
            }
            EnsembleKind::Boosting => {}
        }
    }
    out
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
fn classifier_candidates(dataset: &Dataset) -> Vec<(String, Pipeline)> {
    use crate::transform::{MinMaxScaler, StandardScaler};
    let depths = [Some(2u16), Some(4), Some(8), None];
    let trees = [50u16, 100];
    let mut out = Vec::new();

    // Seed the preprocessing from EDA's suggestion; vary only the model on top.
    #[cfg(feature = "eda")]
    if let Some(base) = seeded_base(dataset) {
        let prep = base.step_names().join("+");
        let prep = if prep.is_empty() { "raw".into() } else { prep };
        for &depth in &depths {
            for &n in &trees {
                let mut rf = RandomForest::new().n_trees(n);
                if let Some(d) = depth {
                    rf = rf.max_depth(d);
                }
                let pipe = base.clone().estimator("rf", rf);
                let depth_s = depth
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "none".into());
                out.push((
                    format!("profile[{prep}] | rf(trees={n}, depth={depth_s})"),
                    pipe,
                ));
            }
        }
        for &k in &[3usize, 5, 9] {
            out.push((
                format!("profile[{prep}] | knn(k={k})"),
                base.clone().estimator("knn", Knn::k(k)),
            ));
        }
        out.push((
            format!("profile[{prep}] | naive_bayes"),
            base.clone().estimator("nb", NaiveBayes::new()),
        ));
        out.push((
            format!("profile[{prep}] | svc(linear)"),
            base.clone().estimator("svc", Svc::linear()),
        ));
        if class_count(dataset.target()) == 2 {
            for &l2 in &[0.0, 0.01, 0.1] {
                out.push((
                    format!("profile[{prep}] | logistic(l2={l2})"),
                    base.clone()
                        .estimator("logistic", LogisticRegression::new().l2(l2)),
                ));
            }
        }
        return out;
    }

    // Fallback: search the scaler as well as the model.
    for scaler in ["none", "standard", "minmax"] {
        for &depth in &depths {
            for &n in &trees {
                let mut rf = RandomForest::new().n_trees(n);
                if let Some(d) = depth {
                    rf = rf.max_depth(d);
                }
                let mut pipe = Pipeline::new();
                pipe = match scaler {
                    "standard" => pipe.step("scale", StandardScaler::new()),
                    "minmax" => pipe.step("scale", MinMaxScaler::new()),
                    _ => pipe,
                };
                pipe = pipe.estimator("rf", rf);
                let depth_s = depth
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "none".into());
                out.push((format!("{scaler} | rf(trees={n}, depth={depth_s})"), pipe));
            }
        }
        for &k in &[3usize, 5, 9] {
            let mut pipe = Pipeline::new();
            pipe = match scaler {
                "standard" => pipe.step("scale", StandardScaler::new()),
                "minmax" => pipe.step("scale", MinMaxScaler::new()),
                _ => pipe,
            };
            out.push((
                format!("{scaler} | knn(k={k})"),
                pipe.estimator("knn", Knn::k(k)),
            ));
        }
        let mut pipe = Pipeline::new();
        pipe = match scaler {
            "standard" => pipe.step("scale", StandardScaler::new()),
            "minmax" => pipe.step("scale", MinMaxScaler::new()),
            _ => pipe,
        };
        out.push((
            format!("{scaler} | naive_bayes"),
            pipe.estimator("nb", NaiveBayes::new()),
        ));
        let mut pipe = Pipeline::new();
        pipe = match scaler {
            "standard" => pipe.step("scale", StandardScaler::new()),
            "minmax" => pipe.step("scale", MinMaxScaler::new()),
            _ => pipe,
        };
        out.push((
            format!("{scaler} | svc(linear)"),
            pipe.estimator("svc", Svc::linear()),
        ));
        if class_count(dataset.target()) == 2 {
            for &l2 in &[0.0, 0.01, 0.1] {
                let mut pipe = Pipeline::new();
                pipe = match scaler {
                    "standard" => pipe.step("scale", StandardScaler::new()),
                    "minmax" => pipe.step("scale", MinMaxScaler::new()),
                    _ => pipe,
                };
                out.push((
                    format!("{scaler} | logistic(l2={l2})"),
                    pipe.estimator("logistic", LogisticRegression::new().l2(l2)),
                ));
            }
        }
    }

    out
}

/// Preprocessing × LinearRegression candidates.
#[cfg_attr(not(feature = "eda"), allow(unused_variables))]
fn regressor_candidates(dataset: &Dataset) -> Vec<(String, Pipeline)> {
    use crate::transform::{MinMaxScaler, StandardScaler};
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
        let mut pipe = Pipeline::new();
        pipe = match scaler {
            "standard" => pipe.step("scale", StandardScaler::new()),
            "minmax" => pipe.step("scale", MinMaxScaler::new()),
            _ => pipe,
        };
        pipe = pipe.estimator("lr", LinearRegression::new());
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
