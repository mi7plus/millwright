use super::*;
use polars::prelude::*;

fn sample() -> Table {
    let df = df!(
        "age"   => [Some(20.0_f64), Some(30.0), None, Some(40.0)],
        "city"  => ["ny", "sf", "ny", "la"],
        "const" => [1.0_f64, 1.0, 1.0, 1.0],
        "label" => [0_i64, 1, 0, 1],
    )
    .unwrap();
    Table::from_polars(df)
}

fn numeric<'a>(p: &'a Profile, name: &str) -> &'a NumericProfile {
    p.columns()
        .iter()
        .find_map(|c| match c {
            ColumnProfile::Numeric(n) if n.name == name => Some(n),
            _ => None,
        })
        .expect("numeric column")
}

#[test]
fn overview_counts_kinds_and_missing() {
    let p = Profile::of(&sample()).unwrap();
    let o = p.overview();
    assert_eq!((o.nrows, o.ncols), (4, 4));
    assert_eq!(o.n_numeric, 3); // age, const, label
    assert_eq!(o.n_categorical, 1); // city
    assert_eq!(o.missing_cells, 1);
    assert_eq!(p.missingness().total, 1);
}

#[test]
fn numeric_stats_ignore_nulls() {
    let p = Profile::of(&sample()).unwrap();
    let age = numeric(&p, "age");
    assert_eq!(age.missing, 1);
    assert!((age.mean - 30.0).abs() < 1e-9); // mean of {20,30,40}
    assert_eq!(age.count, 3);
}

#[test]
fn flags_constant_and_categorical() {
    let p = Profile::of(&sample()).unwrap();
    assert!(p
        .alerts()
        .iter()
        .any(|a| a.suggested == "Drop" && a.column.as_deref() == Some("const")));
    assert!(p
        .alerts()
        .iter()
        .any(|a| a.suggested == "OneHotEncoder" && a.column.as_deref() == Some("city")));
}

#[test]
fn suggests_impute_encode_scale() {
    let p = Profile::of(&sample()).unwrap();
    let pipe = p.suggest_pipeline();
    assert_eq!(pipe.step_names(), vec!["impute", "encode", "scale"]);
}

#[test]
fn target_classification_class_balance() {
    let p = Profile::of_with_target(&sample(), "label").unwrap();
    match &p.target().unwrap().kind {
        TargetKind::Classification { classes } => {
            assert_eq!(classes.len(), 2);
            assert!(classes.iter().all(|(_, n)| *n == 2));
        }
        _ => panic!("expected classification target"),
    }
}

#[test]
fn renders_html_report() {
    let p = Profile::of_with_target(&sample(), "label").unwrap();
    let html = p.render_html();
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains("Data profile"));
    assert!(html.contains("Alerts"));
}

#[test]
fn computes_kurtosis_and_z_outliers() {
    let mut xs = vec![1.0_f64; 19];
    xs.push(100.0); // one extreme value
    let p = Profile::of(&Table::from_polars(df!("x" => xs).unwrap())).unwrap();
    let x = numeric(&p, "x");
    assert!(
        x.kurtosis.is_finite() && x.kurtosis > 3.0,
        "kurtosis {}",
        x.kurtosis
    );
    assert_eq!(x.outliers_z, 1, "the extreme value is a z-score outlier");
}

#[test]
fn computes_spearman_correlation() {
    // y = x^2: perfectly monotonic (Spearman = 1) but nonlinear (Pearson < 1).
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();
    let p = Profile::of(&Table::from_polars(df!("x" => x, "y" => y).unwrap())).unwrap();
    let c = p.correlations();
    let (i, j) = (
        c.columns.iter().position(|n| n == "x").unwrap(),
        c.columns.iter().position(|n| n == "y").unwrap(),
    );
    assert!(
        (c.spearman[i][j] - 1.0).abs() < 1e-9,
        "spearman {}",
        c.spearman[i][j]
    );
    assert!(c.matrix[i][j] < 0.99, "pearson {}", c.matrix[i][j]);
}

#[test]
fn flags_co_missing_columns() {
    // `a` and `b` are null on exactly the same rows.
    let a = [Some(1.0_f64), None, Some(3.0), None, Some(5.0)];
    let b = [Some(1.0_f64), None, Some(3.0), None, Some(5.0)];
    let c = [Some(1.0_f64), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
    let p = Profile::of(&Table::from_polars(
        df!("a" => a, "b" => b, "c" => c).unwrap(),
    ))
    .unwrap();
    let cm = &p.missingness().co_missing;
    assert!(
        cm.iter()
            .any(|(x, y, phi)| (x == "a" && y == "b") && *phi > 0.9),
        "co_missing = {cm:?}"
    );
}

#[test]
fn high_cardinality_suggests_target_encoder() {
    let ids: Vec<String> = (0..30).map(|i| format!("id{i}")).collect();
    let p = Profile::of(&Table::from_polars(df!("uid" => ids).unwrap())).unwrap();
    assert!(p
        .alerts()
        .iter()
        .any(|a| a.suggested == "TargetEncoder" && a.column.as_deref() == Some("uid")));
}

#[test]
fn imbalance_raises_a_smote_alert() {
    let mut y = vec![0_i64; 9];
    y.extend([1, 1, 1]); // 9:3 = 3:1 imbalance
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let p = Profile::of_with_target(&Table::from_polars(df!("x" => x, "y" => y).unwrap()), "y")
        .unwrap();
    assert!(p.alerts().iter().any(|a| a.suggested == "Smote"));
    let _ = p.suggest_pipeline(); // builds without panicking
}
