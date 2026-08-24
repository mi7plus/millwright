use super::*;
use crate::backends::smartcore::RandomForest;
use crate::frame::Dataset;
use crate::traits::{Estimator, Predictor};

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("millwright_onnx_{name}.onnx"))
}

fn two_class() -> (Dataset, Frame) {
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        rows.push(vec![i as f64 * 0.05, i as f64 * 0.05]);
        y.push(0.0);
        rows.push(vec![9.0 + i as f64 * 0.05, 9.0 + i as f64 * 0.05]);
        y.push(1.0);
    }
    let cols = vec!["a".to_string(), "b".to_string()];
    let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
    let probe =
        Frame::from_rows(vec![vec![0.3, 0.2], vec![9.2, 9.3], vec![0.1, 0.0]], cols).unwrap();
    (ds, probe)
}

#[test]
fn random_forest_serves_natively_through_onnx() {
    // A forest exports to ONNX-ML tree ops tract can't run — so InferenceModel
    // evaluates them natively. The round-trip must match the pipeline.
    use crate::pipeline::Pipeline;
    use crate::transform::StandardScaler;
    let (ds, probe) = two_class();
    let mut pipe = Pipeline::new()
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new().n_trees(15).max_depth(4));
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();

    let path = scratch("rf_serve");
    pipe.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    let via_onnx = loaded.predict(&probe).unwrap();
    assert_eq!(
        native, via_onnx,
        "ONNX-served forest must match the pipeline"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn random_forest_exports_valid_onnx() {
    // The tree-ensemble export is a valid ONNX-ML artifact for full runtimes
    // (e.g. onnxruntime). tract implements NN ops, not ONNX-ML tree ops, so
    // we validate the export here rather than running it through tract.
    let (ds, _) = two_class();
    let mut rf = RandomForest::new().n_trees(20).max_depth(4);
    rf.fit(&ds).unwrap();
    assert!(rf.to_onnx().is_ok());
    let path = scratch("rf");
    rf.export_onnx(&path).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn linear_regression_round_trips_through_onnx() {
    use crate::backends::smartcore::LinearRegression;
    // y = 2*x1 + 3*x2 + 1
    let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let ds = Dataset::new(
        Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
        y,
    )
    .unwrap();
    let mut lr = LinearRegression::new();
    lr.fit(&ds).unwrap();

    let probe = Frame::from_rows(
        vec![vec![20.0, 1.0], vec![5.0, 2.0]],
        vec!["x1".into(), "x2".into()],
    )
    .unwrap();
    let native = lr.predict(&probe).unwrap();

    let path = scratch("lr");
    lr.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    let via_onnx = loaded.predict(&probe).unwrap();
    for (a, b) in native.iter().zip(&via_onnx) {
        assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn pipeline_scaler_plus_linear_round_trips() {
    use crate::backends::smartcore::LinearRegression;
    use crate::pipeline::Pipeline;
    use crate::transform::StandardScaler;

    let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let ds = Dataset::new(
        Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
        y,
    )
    .unwrap();

    let mut pipe = Pipeline::new()
        .step("scale", StandardScaler::new())
        .estimator("lr", LinearRegression::new());
    pipe.fit(&ds).unwrap();

    let probe = Frame::from_rows(
        vec![vec![20.0, 1.0], vec![5.0, 2.0]],
        vec!["x1".into(), "x2".into()],
    )
    .unwrap();
    let native = pipe.predict(&probe).unwrap();

    // whole pipeline -> one ONNX graph (Sub, Div, Gemm) -> tract
    let path = scratch("pipe");
    pipe.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    let via_onnx = loaded.predict(&probe).unwrap();
    for (a, b) in native.iter().zip(&via_onnx) {
        assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn inference_model_serves_as_a_pipeline_estimator() {
    use crate::backends::smartcore::LinearRegression;
    use crate::pipeline::Pipeline;

    // Train and export an external model...
    let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let cols = vec!["x1".to_string(), "x2".to_string()];
    let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
    let mut lr = LinearRegression::new();
    lr.fit(&ds).unwrap();
    let path = scratch("pipe_estimator");
    lr.export_onnx(&path).unwrap();

    // ...then load it back and drop it in as a pipeline's (frozen) estimator.
    let onnx = InferenceModel::load(&path).unwrap();
    let mut pipe = Pipeline::new().estimator("onnx", onnx);
    pipe.fit(&ds).unwrap(); // no-op fit — the model is already trained
    let probe = Frame::from_rows(vec![vec![20.0, 1.0]], cols).unwrap();

    let via_pipe = pipe.predict(&probe).unwrap();
    let direct = InferenceModel::load(&path)
        .unwrap()
        .predict(&probe)
        .unwrap();
    assert!((via_pipe[0] - direct[0]).abs() < 1e-4);
    let _ = std::fs::remove_file(&path);
}
