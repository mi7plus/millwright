//! Phase 7: AutoML — the framework, pointed at itself.
//!
//! Point it at data and a budget; get back a leaderboard and the best
//! *deployable* model — which flows straight into ONNX export. Run with:
//! `cargo run --example automl --features "smartcore-backend automl onnx"`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // Two clearly separable classes on two features.
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..30 {
        rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        y.push(0.0);
        rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
        y.push(1.0);
    }
    let train = Dataset::new(Frame::from_rows(rows, vec!["a".into(), "b".into()])?, y)?;

    // Point AutoML at the data and a budget.
    let result = AutoML::classifier()
        .budget(Budget::trials(20))
        .metric(Metric::F1)
        .cv(StratifiedKFold::new(5))
        .seed(0)
        .fit(&train)?;

    println!("leaderboard (top rows):");
    for line in result.leaderboard().lines().take(6) {
        println!("  {line}");
    }
    println!(
        "winner : {}  (F1 = {:.3})",
        result.best_label(),
        result.best_score()
    );

    let probe = Frame::from_rows(
        vec![vec![0.2, 0.2], vec![9.3, 9.3]],
        vec!["a".into(), "b".into()],
    )?;
    println!("predictions : {:?}", result.predict(&probe)?);

    // The winner is deployable — unlike a TPOT object.
    let path = temp_dir().join("millwright_automl.onnx");
    match result.export_onnx(&path) {
        Ok(()) => println!("exported winner -> {} (deployable ONNX)", path.display()),
        Err(e) => println!("winner is an ensemble, not single-pipeline: {e}"),
    }

    println!("ok — auto-sklearn, but the output actually deploys.");
    Ok(())
}
