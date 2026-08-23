//! Model selection — cross-validation, scoring, and search over a pipeline.
//!
//! This module adapts [`model-selection-rs`](https://docs.rs/model-selection-rs)
//! (the CV splitters and scorers) into the framework, then layers grid and
//! random search over any [`Model`](crate::traits::Model) — including a whole
//! [`Pipeline`](crate::pipeline::Pipeline), tuned by `"step__param"` path.
//!
//! It is split into [`scoring`] (the [`Metric`] enum), [`cv`] (the
//! [`CrossValidator`] splitters and [`cross_val_score`]), and [`search`]
//! ([`ParamGrid`], [`GridSearch`], [`RandomSearch`], and — behind `hpo` —
//! `BayesSearch`).
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

pub mod cv;
pub mod scoring;
pub mod search;

pub use cv::{cross_val_score, CrossValidator, CrossValidatorClone, KFold, StratifiedKFold};
pub use scoring::Metric;
pub use search::{GridSearch, ParamGrid, RandomSearch, SearchResult};

#[cfg(feature = "hpo")]
pub use search::{BayesSearch, SearchSpace};

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

// Metric and CV-splitter tests that need no backend — they exercise the scoring
// directions and fold partitioning that every search relies on.
#[cfg(test)]
mod metric_cv_tests {
    use super::*;
    use crate::frame::{Dataset, Frame};
    use crate::logistic::LogisticRegression;

    fn labelled() -> Dataset {
        let rows: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        Dataset::new(Frame::from_rows(rows, vec!["x".into()]).unwrap(), y).unwrap()
    }

    #[test]
    fn metric_direction_is_correct() {
        assert!(Metric::Accuracy.greater_is_better());
        assert!(Metric::F1.greater_is_better());
        assert!(Metric::R2.greater_is_better());
        assert!(!Metric::Mae.greater_is_better());
        assert!(!Metric::Mse.greater_is_better());
        assert!(!Metric::Rmse.greater_is_better());
    }

    #[test]
    fn metric_scores_are_sane() {
        let t = [1.0, 2.0, 3.0];
        assert!(Metric::Mae.score(&t, &t).abs() < 1e-9); // perfect fit -> 0 error
        assert!((Metric::Accuracy.score(&[0.0, 1.0, 1.0], &[0.0, 1.0, 1.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kfold_tests_every_row_once() {
        let splits = KFold::new(4).splits(&labelled()).unwrap();
        assert_eq!(splits.len(), 4);
        let mut seen: Vec<usize> = splits.iter().flat_map(|(_, test)| test.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn stratified_keeps_both_classes_per_fold() {
        let ds = labelled();
        for (_, test) in StratifiedKFold::new(2).splits(&ds).unwrap() {
            let classes: std::collections::BTreeSet<i64> =
                test.iter().map(|&i| ds.target()[i] as i64).collect();
            assert_eq!(classes.len(), 2, "fold missing a class: {test:?}");
        }
    }

    #[test]
    fn cross_val_score_runs_a_core_model() {
        // the search engine works with any Model — here the core logistic one,
        // no backend feature required.
        let ds = labelled();
        let score = cross_val_score(
            &LogisticRegression::new(),
            &ds,
            &StratifiedKFold::new(2),
            Metric::Accuracy,
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&score));
    }
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;
    use crate::frame::{Dataset, Frame};
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
        let test = Frame::from_rows(
            vec![vec![0.1, 0.1], vec![9.2, 9.2]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(search.predict(&test).unwrap(), vec![0.0, 1.0]);
    }

    /// Two overlapping clusters with deterministic label noise — learnable but
    /// imperfect, so a shallow-tree fold can predict the majority class and
    /// leave F1 undefined (0/0). That fold used to poison the mean with NaN.
    fn noisy_two_class_dataset() -> Dataset {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let a = i as f64 * 0.1; // 0.0 .. 3.9
            rows.push(vec![a, a]);
            let base = if a > 2.0 { 1.0 } else { 0.0 };
            // flip every 7th label to inject overlap
            y.push(if i % 7 == 0 { 1.0 - base } else { base });
        }
        Dataset::new(
            Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(),
            y,
        )
        .unwrap()
    }

    #[test]
    fn grid_search_f1_stays_finite_on_noisy_data() {
        let ds = noisy_two_class_dataset();
        let pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("rf", RandomForest::new());
        let search = GridSearch::new(pipe, crate::grid! { "rf__max_depth" => [1, 2, 4] })
            .cv(StratifiedKFold::new(5))
            .scoring(Metric::F1)
            .fit(&ds)
            .unwrap();
        let s = search.best_score();
        assert!(s.is_finite(), "F1 best_score must be finite, got {s}");
        assert!(
            s > 0.0,
            "F1 best_score should be > 0 on learnable data, got {s}"
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn search_winner_exports_to_onnx() {
        use crate::backends::smartcore::LinearRegression;
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i % 3) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 1.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();
        // affine preprocessing + a linear estimator => ONNX-exportable
        let pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("lr", LinearRegression::new());
        let search = GridSearch::new(pipe, ParamGrid::new())
            .cv(KFold::new(3))
            .scoring(Metric::R2)
            .fit(&ds)
            .unwrap();
        let path =
            std::env::temp_dir().join(format!("mw_search_export_{}.onnx", std::process::id()));
        search.export_onnx(&path).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
        let _ = std::fs::remove_file(&path);
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

    #[cfg(feature = "hpo")]
    #[test]
    fn bayes_search_tunes_over_a_space() {
        let ds = two_class_dataset();
        let space = SearchSpace::new().int("max_depth", 1, 8);
        let search = BayesSearch::new(RandomForest::new(), space)
            .n_trials(6)
            .seed(0)
            .cv(StratifiedKFold::new(3))
            .scoring(Metric::F1)
            .fit(&ds)
            .unwrap();
        assert_eq!(search.leaderboard().len(), 6);
        assert!(search.best_score() > 0.9, "F1 was {}", search.best_score());
        let test = Frame::from_rows(
            vec![vec![0.1, 0.1], vec![9.2, 9.2]],
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(search.predict(&test).unwrap(), vec![0.0, 1.0]);
    }
}
