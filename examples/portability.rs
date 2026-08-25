//! Phase 4: train once; run in Rust, Python, or any ONNX runtime.
//!
//! Exports models and a whole pipeline to ONNX, then loads the artifacts back
//! through tract and checks the predictions match the native ones. Run with:
//! `cargo run --example portability --features "smartcore-backend onnx"`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // A linear pipeline: standardize, then ordinary least squares.
    // y = 2*x1 + 3*x2 + 1
    let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i % 5) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let cols = vec!["x1".to_string(), "x2".to_string()];
    let train = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y)?;

    let mut pipe = Pipeline::new()
        .step("scale", StandardScaler::new())
        .estimator("lr", LinearRegression::new());
    pipe.fit(&train)?;

    let probe = Frame::from_rows(vec![vec![25.0, 1.0], vec![7.0, 3.0]], cols)?;
    let native = pipe.predict(&probe)?;

    // Export the WHOLE pipeline (scaler + model) as one ONNX graph, then run it
    // back through tract — a full round-trip.
    let path = temp_dir().join("millwright_pipeline.onnx");
    pipe.export_onnx(&path)?;
    println!("exported pipeline -> {}", path.display());

    let model = InferenceModel::load(&path)?;
    let via_onnx = model.predict(&probe)?;
    println!("native predictions : {native:?}");
    println!("onnx  predictions  : {via_onnx:?}");
    for (a, b) in native.iter().zip(&via_onnx) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }

    // A random forest exports to a valid ONNX-ML artifact. Millwright runs the
    // tree graph through its native interpreter; external runtimes can run it too.
    let mut rf = RandomForest::new().n_trees(30).max_depth(4);
    let churn = Dataset::new(
        Frame::from_rows(
            (0..20)
                .flat_map(|i| [vec![i as f64 * 0.1, 0.0], vec![9.0 + i as f64 * 0.1, 1.0]])
                .collect(),
            vec!["score".into(), "flag".into()],
        )?,
        (0..20).flat_map(|_| [0.0, 1.0]).collect(),
    )?;
    rf.fit(&churn)?;
    let rf_path = temp_dir().join("millwright_forest.onnx");
    rf.export_onnx(&rf_path)?;
    println!("exported random forest -> {} (ONNX-ML)", rf_path.display());

    println!("ok — train once; run in Rust, Python, or any ONNX runtime.");
    Ok(())
}
