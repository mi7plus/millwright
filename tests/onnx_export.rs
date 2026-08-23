#![cfg(feature = "onnx")]
use millwright::prelude::*;

// A low-cardinality integer column `g` (one-hot inferred) + a numeric `x`.
fn categorical_data() -> (Dataset, Frame) {
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..30 {
        let g = (i % 3) as f64;
        let x = i as f64 * 0.1;
        rows.push(vec![x, g]);
        y.push(f64::from(u8::from(x + g > 2.0)));
    }
    let cols = vec!["x".to_string(), "g".to_string()];
    let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
    let probe =
        Frame::from_rows(vec![vec![0.5, 1.0], vec![2.5, 2.0], vec![1.0, 0.0]], cols).unwrap();
    (ds, probe)
}

#[test]
fn onehot_scale_rf_serves_natively() {
    let (ds, probe) = categorical_data();
    let mut pipe = Pipeline::new()
        .step("encode", OneHotEncoder::infer())
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new().n_trees(15).max_depth(4));
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();
    let path = std::env::temp_dir().join("mw_onehot_rf.onnx");
    pipe.export_onnx(&path).unwrap();
    let loaded = InferenceModel::load(&path).unwrap();
    assert_eq!(
        native,
        loaded.predict(&probe).unwrap(),
        "onehot+scale+rf via ONNX"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn onehot_scale_linear_serves_via_tract() {
    let (ds, probe) = categorical_data();
    let mut pipe = Pipeline::new()
        .step("encode", OneHotEncoder::infer())
        .step("scale", StandardScaler::new())
        .estimator("lr", LinearRegression::new());
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();
    let path = std::env::temp_dir().join("mw_onehot_lr.onnx");
    pipe.export_onnx(&path).unwrap();
    let via = InferenceModel::load(&path)
        .unwrap()
        .predict(&probe)
        .unwrap();
    for (a, b) in native.iter().zip(&via) {
        assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn full_pipeline_impute_encode_scale_rf_round_trips() {
    // The realistic lifecycle pipeline: impute -> one-hot -> scale -> forest.
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..40 {
        let g = if i % 7 == 0 { f64::NAN } else { (i % 3) as f64 };
        let x = i as f64 * 0.1;
        rows.push(vec![x, g]);
        y.push(f64::from(u8::from(x > 2.0)));
    }
    let cols = vec!["x".to_string(), "g".to_string()];
    let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
    let probe = Frame::from_rows(vec![vec![0.5, 1.0], vec![2.5, 2.0]], cols).unwrap();

    let mut pipe = Pipeline::new()
        .step("impute", SimpleImputer::median())
        .step("encode", OneHotEncoder::infer())
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new().n_trees(20).max_depth(4));
    pipe.fit(&ds).unwrap();
    let native = pipe.predict(&probe).unwrap();

    let path = std::env::temp_dir().join("mw_full_pipe.onnx");
    pipe.export_onnx(&path).unwrap();
    let served = InferenceModel::load(&path)
        .unwrap()
        .predict(&probe)
        .unwrap();
    assert_eq!(
        native, served,
        "impute+encode+scale+rf must serve identically"
    );
    let _ = std::fs::remove_file(&path);
}

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
