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

#[test]
fn forest_onnx_maps_class_labels() {
    // A forest with non-0-based labels must serve the labels it predicts, not
    // the raw argmax index.
    for labels in [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 5.0, 9.0]] {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..15 {
            rows.push(vec![i as f64 * 0.05, 0.0]);
            y.push(labels[0]);
            rows.push(vec![5.0 + i as f64 * 0.05, 5.0]);
            y.push(labels[1]);
            rows.push(vec![10.0 + i as f64 * 0.05, 10.0]);
            y.push(labels[2]);
        }
        let cols = vec!["a".to_string(), "b".to_string()];
        let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
        let probe =
            Frame::from_rows(vec![vec![0.1, 0.0], vec![5.1, 5.0], vec![10.1, 10.0]], cols).unwrap();

        let mut rf = RandomForest::new().n_trees(20).max_depth(4);
        rf.fit(&ds).unwrap();
        let native = rf.predict(&probe).unwrap();

        let path = std::env::temp_dir().join(format!("mw_labels_{}.onnx", labels[2] as i64));
        rf.export_onnx(&path).unwrap();
        let served = InferenceModel::load(&path)
            .unwrap()
            .predict(&probe)
            .unwrap();
        assert_eq!(
            native, served,
            "labels {labels:?} must round-trip through ONNX"
        );
        let _ = std::fs::remove_file(&path);
    }
}

// A 3-class dataset with non-contiguous labels `[2, 5, 9]`.
fn tensorize_data(n: usize, p: usize) -> Dataset {
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cls = (i % 3) as f64;
        rows.push(
            (0..p)
                .map(|j| ((i * p + j) as f64).sin() + cls * 2.0)
                .collect(),
        );
        y.push([2.0, 5.0, 9.0][cls as usize]);
    }
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap()
}

// `export_onnx_gpu` re-encodes a forest as tensor ops (Gather/LessOrEqual/MatMul/
// Equal), a graph with no ONNX-ML op, so it round-trips through tract on CPU and
// must predict exactly the same class labels as the native forest. This covers
// the tensorization independently of the `gpu-inference` runtime.
#[test]
fn tensorized_forest_matches_native_via_tract() {
    let ds = tensorize_data(180, 5);
    let mut rf = RandomForest::new().n_trees(10).max_depth(4);
    rf.fit(&ds).unwrap();
    let native = rf.predict(ds.features()).unwrap();

    let path = std::env::temp_dir().join("mw_tensorized_rf.onnx");
    rf.export_onnx_gpu(&path).unwrap();
    let via_tensor = InferenceModel::load(&path)
        .unwrap()
        .predict(ds.features())
        .unwrap();
    assert_eq!(native, via_tensor);
    assert!(native.iter().all(|v| [2.0, 5.0, 9.0].contains(v)));
    let _ = std::fs::remove_file(&path);
}

// A single-tree forest exercises the one-tree (`Identity`) path of the encoding.
#[test]
fn tensorized_single_tree_forest_matches_native() {
    let ds = tensorize_data(120, 4);
    let mut rf = RandomForest::new().n_trees(1).max_depth(4);
    rf.fit(&ds).unwrap();
    let native = rf.predict(ds.features()).unwrap();

    let path = std::env::temp_dir().join("mw_tensorized_rf_single.onnx");
    rf.export_onnx_gpu(&path).unwrap();
    let via_tensor = InferenceModel::load(&path)
        .unwrap()
        .predict(ds.features())
        .unwrap();
    assert_eq!(native, via_tensor);
    let _ = std::fs::remove_file(&path);
}

// A non-tree model has no tree op to re-encode, so the GPU export equals the
// normal one.
#[test]
fn tensorized_export_is_noop_for_linear() {
    let ds = tensorize_data(60, 4);
    let mut model = LinearRegression::new();
    model.fit(&ds).unwrap();

    let plain = std::env::temp_dir().join("mw_lin_plain.onnx");
    let gpu = std::env::temp_dir().join("mw_lin_gpu.onnx");
    model.export_onnx(&plain).unwrap();
    model.export_onnx_gpu(&gpu).unwrap();

    let a = InferenceModel::load(&plain)
        .unwrap()
        .predict(ds.features())
        .unwrap();
    let b = InferenceModel::load(&gpu)
        .unwrap()
        .predict(ds.features())
        .unwrap();
    assert_eq!(a, b);
    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&gpu);
}
