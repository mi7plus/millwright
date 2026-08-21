//! Ingest & EDA: the front of the lifecycle, now real.
//!
//! Load a typed CSV into a polars-backed `Table`, profile it (stats,
//! missingness, correlations, alerts), render an HTML report, and let the
//! profile *draft* the preprocessing pipeline — which we finish with a model
//! and fit. Run with:
//! `cargo run --example explore --features "eda smartcore-backend"`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // A small typed dataset: a numeric with a missing value, a categorical,
    // a near-constant column, and an integer class label.
    let csv = temp_dir().join("millwright_customers.csv");
    std::fs::write(
        &csv,
        "\
age,city,tenure,plan,churned
25,ny,2,basic,1
41,sf,40,pro,0
33,ny,12,basic,0
52,la,60,pro,0
29,sf,,basic,1
48,ny,55,pro,0
37,la,18,basic,1
60,sf,72,pro,0
",
    )
    .map_err(|e| millwright::Error::Backend(e.to_string()))?;

    // 1 — ingest: strings, ints, nulls and all.
    let table = Table::from_csv(&csv)?;
    println!(
        "loaded {:?}  columns = {:?}",
        table.shape(),
        table.column_names()
    );

    // 2 — profile against the target.
    let profile = Profile::of_with_target(&table, "churned")?;
    print!("{}", profile.summary());

    println!("alerts:");
    for alert in profile.alerts() {
        println!("  {alert}");
    }

    // 3 — a shareable HTML report.
    let report = temp_dir().join("millwright_eda.html");
    profile.to_html(&report)?;
    println!("wrote {}", report.display());

    // 4 — EDA drafts the starting pipeline; we just add the model and fit.
    let train = table.into_dataset("churned")?;
    let mut pipe = profile
        .suggest_pipeline()
        .estimator("rf", RandomForest::new().n_trees(50));
    println!("pipeline (EDA-seeded + model): {:?}", pipe.step_names());
    pipe.fit(&train)?;

    let preds = pipe.predict(train.features())?;
    println!("in-sample predictions: {preds:?}");

    println!("ok — the lifecycle starts where the data does.");
    Ok(())
}
