//! Phase 5: past where scikit-learn stops — registry, drift monitor, server.
//!
//! Registers a model version, tags it, rolls back, watches a prediction stream
//! for PSI drift, and builds a served endpoint with the monitor attached. Run:
//! `cargo run --example operations --features "smartcore-backend onnx registry monitor serve"`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // A simple regression model (linear → ONNX runs in tract).
    let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 5) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let train = Dataset::new(Frame::from_rows(rows, vec!["x1".into(), "x2".into()])?, y)?;
    let mut model = LinearRegression::new();
    model.fit(&train)?;
    let reference = model.predict(train.features())?;

    // 1 — registry: version the artifact, tag it, roll back
    let root = temp_dir().join("mw_ops_models");
    let _ = std::fs::remove_dir_all(&root);
    let reg = Registry::local(&root);

    let v1 = reg.register(
        "demand",
        &model,
        Metadata {
            metrics: vec![("r2".into(), 1.0)],
            reference: reference.clone(),
            note: "baseline".into(),
        },
    )?;
    reg.tag("demand", &v1.id, "prod")?;

    // register a genuinely different second version, promote it, then roll back
    let y2: Vec<f64> = train
        .features()
        .as_rows()
        .iter()
        .map(|r| 2.0 * r[0] + 3.0 * r[1] + 5.0)
        .collect();
    let train2 = Dataset::new(train.features().clone(), y2)?;
    let mut model2 = LinearRegression::new();
    model2.fit(&train2)?;
    let v2 = reg.register("demand", &model2, Metadata::default())?;
    reg.tag("demand", &v2.id, "prod")?;
    println!("registry versions : {:?}", reg.versions("demand")?);
    println!("prod -> {}", reg.resolve("demand", "prod")?);
    let reverted = reg.rollback("demand", "prod")?;
    println!("rolled prod back to {reverted}");

    // 2 — drift monitor over the prediction stream
    let monitor = DriftMonitor::psi(&reference)?;
    let probe = train.features();
    monitor.observe(&model.predict(probe)?); // same distribution → stable
    println!("after in-distribution traffic : {:?}", monitor.report()?);

    let drifted = DriftMonitor::psi(&reference)?;
    drifted.observe(&vec![10_000.0; 200]); // way outside the reference
    println!("after shifted traffic         : {:?}", drifted.report()?);

    // 3 — serve the registry's prod artifact, watching for drift
    let onnx = reg.onnx_path("demand", "prod")?;
    let server = Server::from_onnx(&onnx)?
        .route("/predict")
        .with_monitor(DriftMonitor::psi(&reference)?);
    let _router = server.router(); // POST /predict, GET /metrics
    println!("server ready: POST /predict, GET /metrics  (call .serve(addr).await to bind)");

    println!("ok — past where scikit-learn stops.");
    Ok(())
}
