use super::{
    Alert, CategoricalProfile, ColumnProfile, CorrMatrix, NumericProfile, Overview, TargetKind,
    TargetProfile,
};

pub(super) fn alerts(
    overview: &Overview,
    columns: &[ColumnProfile],
    corr: &CorrMatrix,
    target: &Option<TargetProfile>,
) -> Vec<Alert> {
    let mut out = Vec::new();
    let n = overview.nrows.max(1) as f64;

    for column in columns {
        out.extend(column_alerts(column, n));
    }
    out.extend(correlation_alerts(corr));
    out.extend(target_alerts(target));

    out
}

fn column_alerts(column: &ColumnProfile, nrows: f64) -> Vec<Alert> {
    match column {
        ColumnProfile::Numeric(profile) => numeric_alerts(profile, nrows),
        ColumnProfile::Categorical(profile) => categorical_alerts(profile, nrows),
    }
}

fn missing_alert(name: &str, missing: usize, nrows: f64) -> Option<Alert> {
    (missing as f64 / nrows > 0.2).then(|| Alert {
        column: Some(name.to_string()),
        message: format!("{:.0}% missing", 100.0 * missing as f64 / nrows),
        suggested: "SimpleImputer",
    })
}

fn numeric_alerts(profile: &NumericProfile, nrows: f64) -> Vec<Alert> {
    let mut out = missing_alert(&profile.name, profile.missing, nrows)
        .into_iter()
        .collect::<Vec<_>>();
    if profile.distinct <= 1 {
        out.push(Alert {
            column: Some(profile.name.clone()),
            message: "constant / zero-variance".into(),
            suggested: "Drop",
        });
    }
    if profile.skew.is_finite() && profile.skew.abs() > 2.0 {
        out.push(Alert {
            column: Some(profile.name.clone()),
            message: format!("skewed (skew {:.1})", profile.skew),
            suggested: "PowerTransform",
        });
    }
    if profile.outliers as f64 / nrows > 0.01 {
        out.push(Alert {
            column: Some(profile.name.clone()),
            message: format!("{} IQR outliers", profile.outliers),
            suggested: "Winsorize",
        });
    }
    out
}

fn categorical_alerts(profile: &CategoricalProfile, nrows: f64) -> Vec<Alert> {
    let mut out = missing_alert(&profile.name, profile.missing, nrows)
        .into_iter()
        .collect::<Vec<_>>();
    let cardinality = if profile.distinct > 20 {
        Some((
            format!("high cardinality ({} levels)", profile.distinct),
            "TargetEncoder",
        ))
    } else if profile.distinct >= 2 {
        Some((
            format!("categorical ({} levels)", profile.distinct),
            "OneHotEncoder",
        ))
    } else {
        None
    };
    if let Some((message, suggested)) = cardinality {
        out.push(Alert {
            column: Some(profile.name.clone()),
            message,
            suggested,
        });
    }
    out
}

fn correlation_alerts(corr: &CorrMatrix) -> Vec<Alert> {
    corr.high_pairs
        .iter()
        .map(|(a, b, r)| Alert {
            column: Some(format!("{a} ~ {b}")),
            message: format!("correlated |r| = {:.2}", r.abs()),
            suggested: "drop one",
        })
        .collect()
}

fn target_alerts(target: &Option<TargetProfile>) -> Option<Alert> {
    let Some(TargetProfile {
        kind: TargetKind::Classification { classes },
        ..
    }) = target
    else {
        return None;
    };
    let (Some((_, max)), Some((_, min))) = (classes.first(), classes.last()) else {
        return None;
    };
    (*min > 0 && *max as f64 / *min as f64 >= 3.0).then(|| Alert {
        column: None,
        message: format!("class imbalance {}:{}", max, min),
        suggested: "Smote",
    })
}
