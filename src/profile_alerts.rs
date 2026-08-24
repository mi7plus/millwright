use super::{Alert, ColumnProfile, CorrMatrix, Overview, TargetKind, TargetProfile};

pub(super) fn alerts(
    overview: &Overview,
    columns: &[ColumnProfile],
    corr: &CorrMatrix,
    target: &Option<TargetProfile>,
) -> Vec<Alert> {
    let mut out = Vec::new();
    let n = overview.nrows.max(1) as f64;

    for col in columns {
        match col {
            ColumnProfile::Numeric(np) => {
                if np.missing as f64 / n > 0.2 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("{:.0}% missing", 100.0 * np.missing as f64 / n),
                        suggested: "SimpleImputer",
                    });
                }
                if np.distinct <= 1 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: "constant / zero-variance".into(),
                        suggested: "Drop",
                    });
                }
                if np.skew.is_finite() && np.skew.abs() > 2.0 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("skewed (skew {:.1})", np.skew),
                        suggested: "PowerTransform",
                    });
                }
                if np.outliers as f64 / n > 0.01 {
                    out.push(Alert {
                        column: Some(np.name.clone()),
                        message: format!("{} IQR outliers", np.outliers),
                        suggested: "Winsorize",
                    });
                }
            }
            ColumnProfile::Categorical(cp) => {
                if cp.missing as f64 / n > 0.2 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("{:.0}% missing", 100.0 * cp.missing as f64 / n),
                        suggested: "SimpleImputer",
                    });
                }
                if cp.distinct > 20 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("high cardinality ({} levels)", cp.distinct),
                        suggested: "TargetEncoder",
                    });
                } else if cp.distinct >= 2 {
                    out.push(Alert {
                        column: Some(cp.name.clone()),
                        message: format!("categorical ({} levels)", cp.distinct),
                        suggested: "OneHotEncoder",
                    });
                }
            }
        }
    }

    for (a, b, r) in &corr.high_pairs {
        out.push(Alert {
            column: Some(format!("{a} ~ {b}")),
            message: format!("correlated |r| = {:.2}", r.abs()),
            suggested: "drop one",
        });
    }

    if let Some(TargetProfile {
        kind: TargetKind::Classification { classes },
        ..
    }) = target
    {
        if let (Some((_, max)), Some((_, min))) = (classes.first(), classes.last()) {
            if *min > 0 && *max as f64 / *min as f64 >= 3.0 {
                out.push(Alert {
                    column: None,
                    message: format!("class imbalance {}:{}", max, min),
                    suggested: "Smote",
                });
            }
        }
    }

    out
}
