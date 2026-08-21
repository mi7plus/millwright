//! Phase 0 end to end: compose a pipeline, fit it, predict — proving
//! `fit · transform · predict · Pipeline` across the smartcore backend.
//!
//! Run with: `cargo run --example spine`

use millwright::prelude::*;

fn main() -> Result<()> {
    // Two well-separated classes on two features.
    let features = Frame::from_rows(
        vec![
            vec![0.0, 0.1],
            vec![0.4, 0.2],
            vec![0.2, 0.5],
            vec![9.0, 9.1],
            vec![9.4, 8.7],
            vec![8.8, 9.5],
        ],
        vec!["a".into(), "b".into()],
    )?;
    let train = Dataset::new(features.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;

    // A pipeline: standardize, then a random forest — one object.
    let mut pipe = Pipeline::new()
        .step("scale", StandardScaler::new())
        .estimator("rf", RandomForest::new());

    // Tune a parameter deep in the chain by path, the sklearn way.
    pipe.set_param("rf__n_trees", ParamValue::Int(50))?;
    pipe.set_param("rf__max_depth", ParamValue::Int(4))?;

    // fit → transform (inside) → fit estimator.
    pipe.fit(&train)?;
    println!("pipeline steps: {:?}", pipe.step_names());

    // predict replays the fitted transforms, then the model.
    let test = Frame::from_rows(
        vec![vec![0.3, 0.2], vec![9.1, 9.0]],
        vec!["a".into(), "b".into()],
    )?;
    let preds = pipe.predict(&test)?;
    println!("predictions: {preds:?}"); // expect [0.0, 1.0]

    assert_eq!(preds, vec![0.0, 1.0]);
    println!("ok — the spine holds.");
    Ok(())
}
