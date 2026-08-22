//! Trust the probabilities and spot the weird rows.
//!
//! Calibrated probabilities (from a real `ProbaPredictor`), a reliability check,
//! unsupervised outlier detection, and the robustness transformers applied
//! per-column. Run with:
//! `cargo run --example trust --features "calibration anomaly"`

use millwright::prelude::*;

fn main() -> Result<()> {
    let (train, calib, probe) = make_data()?;

    // 1 — a probability-capable model, then *calibrate* its probabilities.
    let mut model = LogisticRegression::new();
    model.fit(&train)?;

    let calibrated = CalibratedClassifier::isotonic(model).fit(&calib)?;
    let proba = calibrated.predict_proba(&probe)?;
    println!("=== calibrated probabilities ===");
    println!("predictions        : {:?}", calibrated.predict(&probe)?);
    println!("P(class 1)         : {:?}", round(proba.column(1)));

    // reliability on the calibration set: predicted vs. observed per bin.
    let scores = calibrated.predict_proba(calib.features())?.column(1);
    println!("=== reliability ===");
    for b in reliability_curve(&scores, calib.target(), 4) {
        println!(
            "  predicted {:.2}  observed {:.2}  (n={})",
            b.mean_predicted, b.fraction_positive, b.count
        );
    }

    // 2 — unsupervised outlier detection over the same features.
    println!("=== anomaly ===");
    let mut maha = Mahalanobis::new();
    maha.fit(train.features())?;
    println!("mahalanobis scores : {:?}", round(maha.score(&probe)?));
    println!("flagged (> 3.0)    : {:?}", maha.is_outlier(&probe, 3.0)?);

    let mut knn = KnnScore::new(3);
    knn.fit(train.features())?;
    println!("knn (k=3) scores   : {:?}", round(knn.score(&probe)?));

    // 3 — robustness transformers, one per column, via a ColumnTransformer.
    println!("=== shaping ===");
    let mut pre = ColumnTransformer::new()
        .add(PowerTransform::yeo_johnson(), ["skewed"]) // de-skew
        .add(Winsorize::new(), ["spiky"]); // clip outliers
    let shaped = pre.fit_transform(train.features())?;
    println!("shaped columns     : {:?}", shaped.columns());

    println!("ok — trust the probabilities, spot the weird rows.");
    Ok(())
}

fn round(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|x| (x * 100.0).round() / 100.0).collect()
}

/// A two-feature binary problem: `skewed` is right-skewed, `spiky` has a couple
/// of outliers. Returns train, a held-out calibration set, and a probe.
fn make_data() -> Result<(Dataset, Dataset, Frame)> {
    let cols = vec!["skewed".to_string(), "spiky".to_string()];
    let (mut rows, mut y) = (Vec::new(), Vec::new());
    for i in 0..40 {
        let cls = i % 2;
        // class 0 sits low, class 1 sits high, on a skewed first feature
        let skewed = if cls == 0 {
            (i as f64 * 0.05).powi(2)
        } else {
            5.0 + i as f64 * 0.1
        };
        let spiky = if i == 7 {
            40.0
        } else {
            cls as f64 + (i % 3) as f64 * 0.1
        };
        rows.push(vec![skewed, spiky]);
        y.push(cls as f64);
    }
    // split 30/10 train/calibration
    let (tr_rows, ca_rows) = rows.split_at(30);
    let (tr_y, ca_y) = y.split_at(30);
    let train = Dataset::new(
        Frame::from_rows(tr_rows.to_vec(), cols.clone())?,
        tr_y.to_vec(),
    )?;
    let calib = Dataset::new(
        Frame::from_rows(ca_rows.to_vec(), cols.clone())?,
        ca_y.to_vec(),
    )?;
    let probe = Frame::from_rows(vec![vec![0.1, 0.0], vec![9.0, 40.0]], cols)?;
    Ok((train, calib, probe))
}
