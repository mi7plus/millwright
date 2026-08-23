#![cfg(feature = "onnx")]
use millwright::prelude::*;

fn data_with_nans() -> (Dataset, Frame) {
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        let a = if i % 5 == 0 { f64::NAN } else { i as f64 * 0.1 };
        rows.push(vec![a, i as f64 * 0.1]);
        y.push(0.0);
        rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
        y.push(1.0);
    }
    let cols = vec!["a".into(), "b".into()];
    let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
    let probe = Frame::from_rows(vec![vec![f64::NAN, 0.2], vec![9.2, 9.3]], cols).unwrap();
    (ds, probe)
}

#[test]
fn impute_scale_rf_serves_natively() {
    let (ds, probe) = data_with_nans();
    let mut pipe = Pipeline::new()
        .step("impute", SimpleImputer::median())
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new().n_trees(15).max_depth(4));
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();
    let path = std::env::temp_dir().join("mw_impute_rf.onnx");
    pipe.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    assert_eq!(
        native,
        loaded.predict(&probe).unwrap(),
        "impute+scale+rf via ONNX"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn impute_scale_linear_serves_via_tract() {
    let (ds, probe) = data_with_nans();
    let mut pipe = Pipeline::new()
        .step("impute", SimpleImputer::mean())
        .step("scale", StandardScaler::new())
        .estimator("lr", LinearRegression::new());
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();
    let path = std::env::temp_dir().join("mw_impute_lr.onnx");
    pipe.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    let via = loaded.predict(&probe).unwrap();
    for (a, b) in native.iter().zip(&via) {
        assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
    }
    let _ = std::fs::remove_file(&path);
}
