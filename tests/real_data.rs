//! End-to-end validation on a real, named dataset — not synthetic.
//!
//! Quinlan's "PlayTennis" (the canonical decision-tree teaching set): 14 days,
//! four *categorical* features (outlook / temperature / humidity / windy) and a
//! categorical target (play). It exercises the whole user flow on real data —
//! CSV ingest, typed profiling, the dtype-aware lowering and encoding, a
//! suggested pipeline, fit, and scoring.

#![cfg(all(feature = "eda", feature = "smartcore-backend"))]

use millwright::prelude::*;

const PLAY_TENNIS: &str = "\
outlook,temperature,humidity,windy,play
sunny,hot,high,false,no
sunny,hot,high,true,no
overcast,hot,high,false,yes
rainy,mild,high,false,yes
rainy,cool,normal,false,yes
rainy,cool,normal,true,no
overcast,cool,normal,true,yes
sunny,mild,high,false,no
sunny,cool,normal,false,yes
rainy,mild,normal,false,yes
sunny,mild,normal,true,yes
overcast,mild,high,true,yes
overcast,hot,normal,false,yes
rainy,mild,high,true,no
";

fn write_csv() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("mw_playtennis_{}.csv", std::process::id()));
    std::fs::write(&path, PLAY_TENNIS).unwrap();
    path
}

#[test]
fn play_tennis_end_to_end() {
    let path = write_csv();
    let table = Table::from_csv(&path).unwrap();

    // 1 — ingest: real typed data — text columns categorical, "windy" inferred
    // as boolean.
    assert_eq!(table.shape(), (14, 5));
    for col in ["outlook", "temperature", "humidity", "play"] {
        assert_eq!(table.kind(col).unwrap(), ColKind::Categorical, "{col}");
    }
    assert_eq!(table.kind("windy").unwrap(), ColKind::Boolean);

    // 2 — profile: the target reads as a 2-class problem (9 yes / 5 no).
    let profile = Profile::of_with_target(&table, "play").unwrap();
    match &profile.target().unwrap().kind {
        TargetKind::Classification { classes } => {
            assert_eq!(classes.len(), 2);
            assert_eq!(classes.iter().map(|(_, n)| n).sum::<usize>(), 14);
        }
        _ => panic!("expected a classification target"),
    }

    // 3 — the lowered features carry their categorical dtype (the three text
    //     columns; "windy" lowers to a 0/1 numeric).
    let train = table.into_dataset("play").unwrap();
    assert_eq!(train.features().shape(), (14, 4));
    assert_eq!(train.features().categorical_columns().len(), 3);

    // 4 — …so the EDA-suggested pipeline one-hots them (no numeric scale step),
    //     and a forest on top learns this tiny, learnable set.
    let mut pipe = profile
        .suggest_pipeline()
        .estimator("lr", LogisticRegression::new().epochs(2000));
    assert!(pipe.step_names().contains(&"encode"));
    assert!(!pipe.step_names().contains(&"scale")); // nothing numeric to scale
    pipe.fit(&train).unwrap();

    let preds = pipe.predict(train.features()).unwrap();
    let correct = preds
        .iter()
        .zip(train.target())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let accuracy = correct as f64 / preds.len() as f64;
    assert!(
        accuracy >= 0.85,
        "train accuracy on PlayTennis was {accuracy}"
    );

    let _ = std::fs::remove_file(&path);
}
