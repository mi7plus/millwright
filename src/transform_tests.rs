use super::*;

#[test]
fn winsorize_clips_to_band() {
    // 0..=10; 5th/95th percentiles clip the extremes inward.
    let rows: Vec<Vec<f64>> = (0..=10).map(|i| vec![i as f64]).collect();
    let f = Frame::from_rows(rows, vec!["x".into()]).unwrap();
    let out = Winsorize::quantiles(0.1, 0.9).fit_transform(&f).unwrap();
    let col = out.column(0);
    // bounds are the 10th/90th pct = 1 and 9; 0 -> 1, 10 -> 9.
    assert_eq!(col[0], 1.0);
    assert_eq!(col[10], 9.0);
    assert_eq!(col[5], 5.0);
}

#[test]
fn power_transform_reduces_skew() {
    // a right-skewed column
    let rows: Vec<Vec<f64>> = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 20.0]
        .iter()
        .map(|&x| vec![x])
        .collect();
    let f = Frame::from_rows(rows, vec!["x".into()]).unwrap();
    let before = skewness(&f.column(0)).abs();
    let out = PowerTransform::yeo_johnson().fit_transform(&f).unwrap();
    let after = skewness(&out.column(0)).abs();
    assert!(after < before, "skew not reduced: {before} -> {after}");
}

#[test]
fn column_transformer_scales_one_group_passes_rest() {
    let f = Frame::from_rows(
        vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]],
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    // scale only "a"; "b" passes through
    let out = ColumnTransformer::new()
        .add(StandardScaler::new(), ["a"])
        .fit_transform(&f)
        .unwrap();
    assert_eq!(out.columns(), &["a".to_string(), "b".into()]);
    // a standardized (mean 0), b untouched
    assert!((out.column(0).iter().sum::<f64>()).abs() < 1e-9);
    assert_eq!(out.column(1), vec![100.0, 200.0, 300.0]);
}

#[test]
fn onehot_prefers_schema_over_heuristic() {
    // Both columns are low-cardinality integers, so the value heuristic would
    // one-hot *both*. With the schema, only the Categorical one is encoded.
    let f = Frame::from_rows(
        vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]],
        vec!["cat".into(), "code".into()],
    )
    .unwrap()
    .with_dtypes(vec![Dtype::Categorical, Dtype::Numeric])
    .unwrap();

    let out = OneHotEncoder::infer().fit_transform(&f).unwrap();
    assert_eq!(
        out.columns(),
        &[
            "cat=0".to_string(),
            "cat=1".into(),
            "cat=2".into(),
            "code".into()
        ]
    );
}

#[test]
fn scaler_passes_categorical_columns_through() {
    let f = Frame::from_rows(
        vec![vec![0.0, 10.0], vec![1.0, 20.0], vec![2.0, 30.0]],
        vec!["cat".into(), "num".into()],
    )
    .unwrap()
    .with_dtypes(vec![Dtype::Categorical, Dtype::Numeric])
    .unwrap();

    let out = StandardScaler::new().fit_transform(&f).unwrap();
    assert_eq!(out.column(0), vec![0.0, 1.0, 2.0]); // categorical untouched
    assert!(out.column(1).iter().sum::<f64>().abs() < 1e-9); // numeric standardized
    assert_eq!(out.dtype(0), Dtype::Categorical); // dtype preserved
}

#[test]
fn target_encoder_maps_category_to_mean_target() {
    // cat in {0,1}; target mean is 1.0 for cat 0, 0.0 for cat 1
    let x = Frame::from_rows(
        vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]],
        vec!["cat".into()],
    )
    .unwrap();
    let ds = Dataset::new(x.clone(), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    // no smoothing for an exact check
    let out = TargetEncoder::columns(["cat"])
        .smoothing(0.0)
        .fit_transform(&ds)
        .unwrap();
    assert_eq!(out.column(0), vec![1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn standardizes_to_zero_mean_unit_std() {
    let f = Frame::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]], vec!["x".into()]).unwrap();
    let mut s = StandardScaler::new();
    let out = s.fit_transform(&f).unwrap();
    let col = out.column(0);
    let mean: f64 = col.iter().sum::<f64>() / 3.0;
    assert!(mean.abs() < 1e-9);
    // population std of the standardized column is 1
    let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0;
    assert!((var - 1.0).abs() < 1e-9);
}

#[test]
fn transform_before_fit_errors() {
    let f = Frame::from_rows(vec![vec![1.0]], vec!["x".into()]).unwrap();
    assert!(StandardScaler::new().transform(&f).is_err());
}

#[test]
fn minmax_maps_to_unit_interval() {
    let f = Frame::from_rows(vec![vec![10.0], vec![20.0], vec![30.0]], vec!["x".into()]).unwrap();
    let mut s = MinMaxScaler::new();
    let out = s.fit_transform(&f).unwrap();
    assert_eq!(out.column(0), vec![0.0, 0.5, 1.0]);
}

#[test]
fn imputer_fills_mean_and_median() {
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
    // mean of a's present {1,3,5} = 3; median of b's present {1,2,4} = 2
    let mut mean = SimpleImputer::mean();
    let om = mean.fit_transform(&f).unwrap();
    assert_eq!(om.get(1, 0), 3.0);
    let mut med = SimpleImputer::median();
    let od = med.fit_transform(&f).unwrap();
    assert_eq!(od.get(2, 1), 2.0);
}

#[test]
fn onehot_expands_selected_column() {
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
    let mut enc = OneHotEncoder::columns(["cat"]);
    let out = enc.fit_transform(&f).unwrap();
    // cat has categories {0,1,2} -> 3 indicator cols + passthrough num = 4
    assert_eq!(
        out.columns(),
        &[
            "cat=0".to_string(),
            "cat=1".into(),
            "cat=2".into(),
            "num".into()
        ]
    );
    assert_eq!(out.row(0), &[1.0, 0.0, 0.0, 5.0]); // cat=0
    assert_eq!(out.row(1), &[0.0, 0.0, 1.0, 6.0]); // cat=2
}

#[test]
fn onehot_infers_low_cardinality_integer_columns() {
    let f = Frame::from_rows(
        vec![vec![0.0, 1.5], vec![1.0, 2.5], vec![0.0, 3.5]],
        vec!["flag".into(), "cont".into()],
    )
    .unwrap();
    let mut enc = OneHotEncoder::infer();
    let out = enc.fit_transform(&f).unwrap();
    // flag is integral low-cardinality -> expanded; cont is continuous -> passthrough
    assert_eq!(
        out.columns(),
        &["flag=0".to_string(), "flag=1".into(), "cont".into()]
    );
}
