//! Golden-output tests (Phase 8 · HARDEN → 1.0).
//!
//! These lock the *numeric* behaviour of the framework to known-good values on
//! fixed inputs. Unit tests (in each module) prove a piece is correct; these
//! prove it does not silently *change* — the guard that matters when a dozen
//! young engine crates churn underneath a stable trait contract. If an engine
//! bump moves a number here, the diff makes it impossible to miss, and the
//! change becomes a deliberate decision rather than an accident.
//!
//! Anchors are chosen to be reproducible regardless of RNG jitter: exact for
//! the deterministic paths (OLS, the affine transforms, the metric formulas)
//! and well-separated for the stochastic ones (the forest), so the golden class
//! labels hold across seeds and platforms.

use millwright::prelude::*;

/// Assert two floats are equal to a tight tolerance, with a message that prints
/// the golden and the actual so a drift is legible at a glance.
#[track_caller]
fn close(actual: f64, golden: f64) {
    assert!(
        (actual - golden).abs() < 1e-9,
        "golden drift: expected {golden}, got {actual} (Δ {:.3e})",
        (actual - golden).abs()
    );
}

#[track_caller]
fn close_vec(actual: &[f64], golden: &[f64]) {
    assert_eq!(
        actual.len(),
        golden.len(),
        "length drift: {actual:?} vs {golden:?}"
    );
    for (a, g) in actual.iter().zip(golden) {
        close(*a, *g);
    }
}

// --------------------------------------------------------------------------
// Deterministic transforms — exact golden output.
// --------------------------------------------------------------------------

#[test]
fn golden_standard_scaler() {
    let f = Frame::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]], vec!["x".into()]).unwrap();
    let out = StandardScaler::new().fit_transform(&f).unwrap();
    // mean 2, population std sqrt(2/3); (x-2)/std = ±sqrt(3/2), 0.
    close_vec(
        &out.column(0),
        &[-1.224744871391589, 0.0, 1.224744871391589],
    );
}

#[test]
fn golden_minmax_scaler() {
    let f = Frame::from_rows(
        vec![vec![10.0, 100.0], vec![20.0, 300.0], vec![30.0, 500.0]],
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    let out = MinMaxScaler::new().fit_transform(&f).unwrap();
    close_vec(&out.column(0), &[0.0, 0.5, 1.0]);
    close_vec(&out.column(1), &[0.0, 0.5, 1.0]);
}

#[test]
fn golden_simple_imputer() {
    let nan = f64::NAN;
    let f = Frame::from_rows(
        vec![
            vec![1.0, 1.0],
            vec![nan, 2.0],
            vec![3.0, nan],
            vec![5.0, 4.0],
        ],
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    // a present {1,3,5} -> mean 3; b present {1,2,4} -> median 2.
    let mean = SimpleImputer::mean().fit_transform(&f).unwrap();
    close(mean.get(1, 0), 3.0);
    let median = SimpleImputer::median().fit_transform(&f).unwrap();
    close(median.get(2, 1), 2.0);
}

#[test]
fn golden_one_hot_encoder() {
    let f = Frame::from_rows(
        vec![
            vec![0.0, 5.0],
            vec![2.0, 6.0],
            vec![1.0, 7.0],
            vec![0.0, 8.0],
        ],
        vec!["cat".into(), "num".into()],
    )
    .unwrap();
    let out = OneHotEncoder::columns(["cat"]).fit_transform(&f).unwrap();
    assert_eq!(
        out.columns(),
        &[
            "cat=0".to_string(),
            "cat=1".into(),
            "cat=2".into(),
            "num".into()
        ],
    );
    close_vec(out.row(0), &[1.0, 0.0, 0.0, 5.0]); // cat = 0
    close_vec(out.row(1), &[0.0, 0.0, 1.0, 6.0]); // cat = 2
    close_vec(out.row(2), &[0.0, 1.0, 0.0, 7.0]); // cat = 1
}

// --------------------------------------------------------------------------
// Deterministic estimator — OLS recovers a known plane exactly.
// --------------------------------------------------------------------------

#[test]
fn golden_linear_regression_plane() {
    // y = 3 + 2*x1 - 1*x2, sampled at points that identify the plane.
    let rows = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 3.0],
    ];
    let y: Vec<f64> = rows.iter().map(|r| 3.0 + 2.0 * r[0] - r[1]).collect();
    let train = Dataset::new(
        Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
        y,
    )
    .unwrap();

    let mut lr = LinearRegression::new();
    lr.fit(&train).unwrap();

    let probe = Frame::from_rows(
        vec![vec![4.0, 2.0], vec![-1.0, 5.0]],
        vec!["x1".into(), "x2".into()],
    )
    .unwrap();
    // 3 + 8 - 2 = 9 ; 3 - 2 - 5 = -4
    close_vec(&lr.predict(&probe).unwrap(), &[9.0, -4.0]);
}

// --------------------------------------------------------------------------
// Metric formulas — exact golden over hand-computable inputs.
// --------------------------------------------------------------------------

#[test]
fn golden_regression_report() {
    // Non-integral truth so the task infers as Regression.
    let t = [1.5, 2.5, 3.5, 4.5];
    let p = [1.0, 2.0, 4.0, 5.0];
    let r = Report::new(&t, &p);
    assert_eq!(r.task(), Task::Regression);
    close(r.get("mae").unwrap(), 0.5);
    close(r.get("mse").unwrap(), 0.25);
    close(r.get("rmse").unwrap(), 0.5);
    close(r.get("r2").unwrap(), 0.8);
}

#[test]
fn golden_classification_report() {
    // Half right: macro precision/recall/f1 all 0.5 on this 2x2 pattern.
    let t = [0.0, 0.0, 1.0, 1.0];
    let p = [0.0, 1.0, 0.0, 1.0];
    let r = Report::new(&t, &p);
    assert_eq!(r.task(), Task::Classification);
    close(r.get("accuracy").unwrap(), 0.5);
    close(r.get("precision").unwrap(), 0.5);
    close(r.get("recall").unwrap(), 0.5);
    close(r.get("f1").unwrap(), 0.5);
}

// --------------------------------------------------------------------------
// Stochastic estimators — golden *class labels* on well-separated data,
// stable across seeds. These pin behaviour without pinning the RNG.
// --------------------------------------------------------------------------

/// Two tight, well-separated clusters and a two-row probe straddling them.
fn two_clusters() -> (Dataset, Frame) {
    let cols = vec!["a".to_string(), "b".to_string()];
    let features = Frame::from_rows(
        vec![
            vec![0.0, 0.1],
            vec![0.4, 0.2],
            vec![0.2, 0.5],
            vec![9.0, 9.1],
            vec![9.4, 8.7],
            vec![8.8, 9.5],
        ],
        cols.clone(),
    )
    .unwrap();
    let train = Dataset::new(features, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let probe = Frame::from_rows(vec![vec![0.3, 0.2], vec![9.1, 9.0]], cols).unwrap();
    (train, probe)
}

#[test]
fn golden_random_forest_labels() {
    let (train, probe) = two_clusters();
    let mut rf = RandomForest::new().n_trees(50);
    rf.fit(&train).unwrap();
    close_vec(&rf.predict(&probe).unwrap(), &[0.0, 1.0]);
}

#[test]
fn golden_pipeline_labels() {
    let (train, probe) = two_clusters();
    let mut pipe = Pipeline::new()
        .step("impute", SimpleImputer::median())
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new().n_trees(50));
    pipe.fit(&train).unwrap();
    close_vec(&pipe.predict(&probe).unwrap(), &[0.0, 1.0]);
}

#[test]
fn golden_soft_vote_labels() {
    let (train, probe) = two_clusters();
    let mut vote = Voting::soft()
        .add("rf_shallow", RandomForest::new().max_depth(2))
        .add("rf_deep", RandomForest::new().max_depth(8));
    vote.fit(&train).unwrap();
    close_vec(&vote.predict(&probe).unwrap(), &[0.0, 1.0]);
}

// --------------------------------------------------------------------------
// ONNX round-trip — the exported artifact must reproduce native predictions.
// Only compiled with the `onnx` feature (CI's `full` job); off by default.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Ingest & EDA — the real CSV path: parse a typed table, profile it, lower it.
// Only compiled with the `eda` feature.
// --------------------------------------------------------------------------

#[cfg(feature = "eda")]
#[test]
fn golden_table_and_profile_from_csv() {
    let path = std::env::temp_dir().join("millwright_golden_eda.csv");
    std::fs::write(&path, "age,city,label\n20,ny,0\n30,sf,1\n,ny,0\n40,la,1\n").unwrap();

    let table = Table::from_csv(&path).unwrap();
    assert_eq!(table.shape(), (4, 3));
    assert_eq!(table.null_count("age").unwrap(), 1); // the blank cell

    let profile = Profile::of_with_target(&table, "label").unwrap();
    let o = profile.overview();
    assert_eq!((o.nrows, o.ncols), (4, 3));
    assert_eq!(o.n_numeric, 2); // age, label
    assert_eq!(o.n_categorical, 1); // city
    assert_eq!(o.missing_cells, 1);

    // lower to the numeric world: features + target, nulls -> NaN
    let ds = table.into_dataset("label").unwrap();
    assert_eq!(ds.features().shape(), (4, 2));
    assert_eq!(ds.target(), &[0.0, 1.0, 0.0, 1.0]);
    assert!(ds.features().get(2, 0).is_nan()); // the missing age

    // EDA drafts a real preprocessing pipeline
    assert_eq!(
        profile.suggest_pipeline().step_names(),
        vec!["impute", "encode", "scale"],
    );
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "onnx")]
#[test]
fn golden_onnx_linear_regression_roundtrip() {
    use millwright::onnx::{ExportOnnx, InferenceModel};

    let rows = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 1.0).collect();
    let train = Dataset::new(Frame::from_rows(rows, vec!["x".into()]).unwrap(), y).unwrap();

    let mut lr = LinearRegression::new();
    lr.fit(&train).unwrap();

    let probe = Frame::from_rows(vec![vec![4.0], vec![10.0]], vec!["x".into()]).unwrap();
    let native = lr.predict(&probe).unwrap();
    close_vec(&native, &[9.0, 21.0]);

    // Export to ONNX, run through tract, and require the same numbers back.
    let path = std::env::temp_dir().join("millwright_golden_lr.onnx");
    lr.export_onnx(&path).unwrap();
    let model = InferenceModel::load(&path).unwrap();
    let onnx_out = model.predict(&probe).unwrap();
    close_vec(&onnx_out, &native);
    let _ = std::fs::remove_file(&path);
}
