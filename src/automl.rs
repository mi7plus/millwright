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

use crate::backends::smartcore::{LinearRegression, RandomForest};
use crate::ensemble::Voting;
use crate::error::{Error, Result};
use crate::frame::{Dataset, Frame};
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

/// An automated model search.
pub struct AutoML {
    task: Task,
    budget: Budget,
    metric: Metric,
    cv: Box<dyn CrossValidator>,
    seed: u64,
    ensemble: bool,
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

        // Auto-ensemble the top-k single pipelines and let it compete.
        let mut ensemble_entry: Option<(String, f64)> = None;
        if self.ensemble && board.len() >= 2 {
            let k = board.len().min(3);
            let mut vote = Voting::soft();
            for (i, (_, _, pipe)) in board.iter().take(k).enumerate() {
                vote = vote.add(format!("c{i}"), pipe.clone());
            }
            let score = cross_val_score(&vote, dataset, self.cv.as_ref(), self.metric)?;
            ensemble_entry = Some((format!("ensemble(top-{k})"), score));
        }

        // The leaderboard is every single candidate plus the ensemble entry.
        let mut leaderboard: Vec<(String, f64)> =
            board.iter().map(|(l, s, _)| (l.clone(), *s)).collect();
        if let Some(e) = &ensemble_entry {
            leaderboard.push(e.clone());
        }
        sort_pairs(&mut leaderboard, self.metric.greater_is_better());

        // Decide and refit the winner on the full dataset.
        let ensemble_wins = ensemble_entry
            .as_ref()
            .map(|(_, s)| leaderboard[0].1 == *s && leaderboard[0].0.starts_with("ensemble"))
            .unwrap_or(false);

        let (winner, label, score) = if ensemble_wins {
            let k = board.len().min(3);
            let mut vote = Voting::soft();
            for (i, (_, _, pipe)) in board.iter().take(k).enumerate() {
                vote = vote.add(format!("c{i}"), pipe.clone());
            }
            vote.fit(dataset)?;
            (
                Winner::Ensemble(Box::new(vote)),
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

    /// The best single pipeline, if a pipeline (not an ensemble) won. Only a
    /// single pipeline is ONNX-exportable.
    pub fn best_pipeline(&self) -> Option<&Pipeline> {
        match &self.winner {
            Winner::Single(p) => Some(p),
            Winner::Ensemble(_) => None,
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

    /// Export the winner to ONNX, if it is a single pipeline.
    #[cfg(feature = "onnx")]
    pub fn export_onnx(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::onnx::ExportOnnx;
        match &self.winner {
            Winner::Single(p) => p.export_onnx(path),
            Winner::Ensemble(_) => Err(Error::Backend(
                "the AutoML winner is an ensemble; not ONNX-exportable".into(),
            )),
        }
    }
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

/// Preprocessing × RandomForest-hyperparameter candidates.
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
