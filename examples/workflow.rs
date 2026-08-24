//! Phase 1 end to end: a tunable, ensemble-ready workflow.
//!
//! Preprocess → balance → search over a pipeline with stratified CV → then
//! combine models with voting and stacking. Run with:
//! `cargo run --example workflow`

use millwright::grid;
use millwright::prelude::*;

fn main() -> Result<()> {
    let (train, probe) = make_data()?;

    // 1 — a preprocessing + model pipeline, with a train-time SMOTE balancer.
    let pipe = Pipeline::new()
        .step("impute", SimpleImputer::median())
        .step("scale", StandardScaler::new())
        .balance(Smote::new().k_neighbors(3).random_state(0))
        .estimator("rf", RandomForest::new());

    // 2 — grid-search the pipeline: tune the forest by path, stratified CV, F1.
    let search = GridSearch::new(pipe, grid! { "rf__max_depth" => [2, 4, 8] })
        .cv(StratifiedKFold::new(4))
        .scoring(Metric::F1)
        .fit(&train)?;

    println!("grid search:");
    for (params, score) in search.leaderboard() {
        println!("  {score:.3}  {params:?}");
    }
    println!("  best F1 = {:.3}", search.best_score());
    println!("  predictions = {:?}", search.predict(&probe)?);

    // 3 — ensembles: probability-averaged soft voting, plus a stacked blend.
    let mut vote = Voting::soft()
        .add("lr", LogisticRegression::new())
        .add("lr_l2", LogisticRegression::new().l2(0.01));
    vote.fit(&train)?;
    println!("soft-vote predictions = {:?}", vote.predict(&probe)?);

    let mut stack = Stacking::meta(RandomForest::new().n_trees(50))
        .base("rf", RandomForest::new().n_trees(30))
        .base("rf2", RandomForest::new().max_depth(3))
        .cv(StratifiedKFold::new(4));
    stack.fit(&train)?;
    println!("stacking predictions = {:?}", stack.predict(&probe)?);

    println!("ok — a real, tunable, ensemble-ready workflow.");
    Ok(())
}

/// Two well-separated classes on two features (with a couple of missing values
/// to exercise the imputer), plus a two-row probe.
fn make_data() -> Result<(Dataset, Frame)> {
    let cols = vec!["a".to_string(), "b".to_string()];
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..24 {
        rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        y.push(0.0);
    }
    for i in 0..8 {
        rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
        y.push(1.0);
    }
    rows[0][0] = f64::NAN; // a missing value for the imputer to fill
    let train = Dataset::new(Frame::from_rows(rows, cols.clone())?, y)?;
    let probe = Frame::from_rows(vec![vec![0.3, 0.2], vec![9.4, 9.3]], cols)?;
    Ok((train, probe))
}
