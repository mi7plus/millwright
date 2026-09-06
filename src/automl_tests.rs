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
    assert!(result.elapsed_seconds().is_finite());
    assert!(result.attempted_trials() > 0);
    assert_eq!(result.attempted_trials(), result.completed_trials());

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
    if !result.supports_proba() {
        let probe = Frame::from_rows(vec![vec![0.1, 0.1]], vec!["a".into(), "b".into()]).unwrap();
        let error = result.predict_proba(&probe).unwrap_err();
        assert!(error.to_string().contains("probability prediction"));
    }
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
        elapsed_seconds: 0.0,
        attempted_trials: 0,
        completed_trials: 0,
        attempted_ensemble_trials: 0,
        completed_ensemble_trials: 0,
        budget_exhausted: false,
        ensemble_search_skipped_by_budget: false,
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
    assert!(result.supports_proba());
    assert_eq!(result.predict_proba(&probe).unwrap().shape(), (2, 2));
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
fn non_finite_scores_are_not_viable_candidates() {
    let features =
        Frame::from_rows((0..12).map(|i| vec![i as f64]).collect(), vec!["x".into()]).unwrap();
    let dataset = Dataset::new(features, vec![1.0; 12]).unwrap();
    let error = AutoML::regressor()
        .budget(Budget::trials(3))
        .cv(KFold::new(3))
        .no_ensemble()
        .fit(&dataset)
        .err()
        .expect("all-NaN R2 search must fail");
    assert!(error.to_string().contains("non-finite score"));
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
    let bad = Pipeline::new().estimator("logistic", LogisticRegression::new().learning_rate(0.0));
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
