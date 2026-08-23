// Verifies the design brief's hero snippet (steps 1-4) compiles and runs.
#![cfg(all(
    feature = "model-selection",
    feature = "explain",
    feature = "onnx",
    feature = "preprocessing"
))]
use millwright::prelude::*;

#[test]
fn brief_hero_snippet_compiles_and_runs() {
    let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 2) as f64]).collect();
    let y: Vec<f64> = (0..40).map(|i| (i >= 20) as i32 as f64).collect();
    let train = Dataset::new(
        Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(),
        y,
    )
    .unwrap();
    let test = train.clone();

    // 1 — compose
    let pipe = Pipeline::new()
        .step("scale", StandardScaler::new())
        .balance(Smote::default())
        .estimator("rf", RandomForest::new());

    // 2 — search & cross-validate
    let model = GridSearch::new(pipe, millwright::grid! { "rf__max_depth" => [4, 8, 16] })
        .cv(StratifiedKFold::new(5))
        .scoring(Metric::F1)
        .fit(&train)
        .unwrap();

    // 3 — assess & explain
    let _report = model.evaluate(&test).unwrap();
    let _shap = model
        .explain(&Explainer::kernel(), test.features())
        .unwrap();

    // 4 — export one portable ONNX artifact
    let path = std::env::temp_dir().join(format!("mw_hero_{}.onnx", std::process::id()));
    model.export_onnx(&path).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    let _ = std::fs::remove_file(&path);
}
