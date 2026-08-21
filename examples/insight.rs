//! Phase 3: trust the model, not just run it.
//!
//! Evaluation report, SHAP + permutation importance, OLS regression
//! diagnostics, and report figures rendered to SVG. Run with:
//! `cargo run --example insight --features "smartcore-backend diagnostics explain viz"`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // A classification task where only the first feature carries signal.
    let (train, test) = classification_split();
    let mut rf = RandomForest::new().n_trees(60);
    rf.fit(&train)?;

    // 1 — evaluation report (metrics bundle, task auto-detected)
    println!("=== evaluation ===");
    print!("{}", rf.evaluate(&test)?);

    // 2 — explainability: SHAP values + permutation importance
    println!("=== explainability ===");
    let shap = rf.explain(&Explainer::kernel().nsamples(80), test.features())?;
    println!("SHAP importance        : {:?}", round2(shap.importance()));
    let perm = permutation_importance(&rf, &test, 8, 0)?;
    println!("permutation importance : {:?}", round2(perm));

    // 3 — OLS regression diagnostics on a linear dataset
    println!("=== diagnostics (OLS) ===");
    let reg = regression_data();
    let diag = Diagnostics::of(&reg)?;
    println!("R^2 = {:.4}   max Cook's D = {:.4}", diag.r_squared(), diag.max_cooks_distance());
    println!("VIF = {:?}", round2(diag.vif()));

    // 4 — report figures to SVG
    println!("=== figures ===");
    let scores = rf.predict(test.features())?;
    let roc_path = temp_dir().join("millwright_roc.svg");
    let auc = viz::roc_svg(test.target(), &scores, &roc_path, (520, 420))?;
    println!("wrote {} (AUC = {auc:.3})", roc_path.display());

    let residuals = diag.residuals();
    let y_pred: Vec<f64> = reg
        .target()
        .iter()
        .zip(&residuals)
        .map(|(y, r)| y - r)
        .collect();
    let resid_path = temp_dir().join("millwright_residuals.svg");
    viz::residuals_svg(reg.target(), &y_pred, &resid_path, (520, 420))?;
    println!("wrote {}", resid_path.display());

    println!("ok — trust the model, not just run it.");
    Ok(())
}

fn round2(v: Vec<(String, f64)>) -> Vec<(String, f64)> {
    v.into_iter().map(|(k, x)| (k, (x * 100.0).round() / 100.0)).collect()
}

fn classification_split() -> (Dataset, Dataset) {
    let cols = vec!["signal".to_string(), "noise".to_string()];
    // Build each class contiguously, then split each class 70/30 so both the
    // train and test sets carry both classes.
    let (mut tr_rows, mut tr_y, mut te_rows, mut te_y) = (vec![], vec![], vec![], vec![]);
    for i in 0..30 {
        let class0 = vec![i as f64 * 0.05, (i % 7) as f64];
        let class1 = vec![9.0 + i as f64 * 0.05, (i % 7) as f64];
        if i < 21 {
            tr_rows.push(class0);
            tr_y.push(0.0);
            tr_rows.push(class1);
            tr_y.push(1.0);
        } else {
            te_rows.push(class0);
            te_y.push(0.0);
            te_rows.push(class1);
            te_y.push(1.0);
        }
    }
    let train = Dataset::new(Frame::from_rows(tr_rows, cols.clone()).unwrap(), tr_y).unwrap();
    let test = Dataset::new(Frame::from_rows(te_rows, cols).unwrap(), te_y).unwrap();
    (train, test)
}

fn regression_data() -> Dataset {
    // y = 2*x1 - 1*x2 + 5
    let rows: Vec<Vec<f64>> = (0..25)
        .map(|i| vec![i as f64, (i as f64 * 0.7) % 5.0])
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] - r[1] + 5.0).collect();
    Dataset::new(Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(), y).unwrap()
}
