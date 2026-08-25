use super::*;
use crate::backends::smartcore::RandomForest;

#[derive(Clone)]
struct WrongLength;

impl Estimator for WrongLength {
    fn name(&self) -> &'static str {
        "WrongLength"
    }

    fn fit(&mut self, _dataset: &Dataset) -> Result<()> {
        Ok(())
    }
}

impl Predictor for WrongLength {
    fn predict(&self, _frame: &Frame) -> Result<Vec<f64>> {
        Ok(vec![0.0])
    }
}

fn two_class() -> Dataset {
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..15 {
        rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        y.push(0.0);
        rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
        y.push(1.0);
    }
    Dataset::new(
        Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(),
        y,
    )
    .unwrap()
}

fn probe() -> Frame {
    Frame::from_rows(
        vec![vec![0.2, 0.2], vec![9.3, 9.3]],
        vec!["a".into(), "b".into()],
    )
    .unwrap()
}

#[test]
fn hard_voting_predicts_clusters() {
    let mut v = Voting::hard()
        .add("rf1", RandomForest::new().n_trees(10))
        .add("rf2", RandomForest::new().n_trees(20));
    v.fit(&two_class()).unwrap();
    assert_eq!(v.predict(&probe()).unwrap(), vec![0.0, 1.0]);
}

#[test]
fn soft_voting_exposes_probabilities() {
    let mut v = Voting::soft()
        .add("lr1", crate::logistic::LogisticRegression::new())
        .add("lr2", crate::logistic::LogisticRegression::new().l2(0.01));
    v.fit(&two_class()).unwrap();
    let proba = v.predict_proba(&probe()).unwrap();
    assert_eq!(proba.shape(), (2, 2)); // 2 rows, 2 classes
    assert!(proba.get(0, 0) > 0.5);
    assert!(proba.get(1, 1) > 0.5);
    assert_eq!(v.predict(&probe()).unwrap(), vec![0.0, 1.0]);
}

#[test]
fn bagging_predicts_clusters() {
    let mut b = Bagging::of(RandomForest::new().n_trees(10))
        .n_estimators(5)
        .seed(1);
    b.fit(&two_class()).unwrap();
    assert_eq!(b.predict(&probe()).unwrap(), vec![0.0, 1.0]);
}

#[test]
fn explicit_classification_rejects_fractional_labels() {
    let features = Frame::from_rows(vec![vec![0.0], vec![1.0]], vec!["x".into()]).unwrap();
    let dataset = Dataset::new(features, vec![0.25, 0.75]).unwrap();
    let mut voting = Voting::hard()
        .add("rf", RandomForest::new())
        .task(EnsembleTask::Classification);
    assert!(voting.fit(&dataset).is_err());
}

#[test]
fn malformed_member_prediction_length_is_an_error() {
    let mut voting = Voting::hard().add("bad", WrongLength);
    voting.fit(&two_class()).unwrap();
    assert!(matches!(voting.predict(&probe()), Err(Error::Shape(_))));
}

#[test]
fn explicit_regression_preserves_integer_valued_targets() {
    use crate::backends::smartcore::LinearRegression;
    use crate::ensemble::EnsembleTask;

    let frame = Frame::from_rows(
        vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ],
        vec!["x".into()],
    )
    .unwrap();
    let dataset = Dataset::new(frame.clone(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();
    let mut model = Bagging::of(LinearRegression::new())
        .n_estimators(5)
        .seed(7)
        .task(EnsembleTask::Regression);
    model.fit(&dataset).unwrap();
    let prediction = model.predict(&frame).unwrap();
    assert!(
        prediction[3] > 4.0,
        "integer regression was treated as voting: {prediction:?}"
    );
}

#[test]
fn boosting_predicts_clusters() {
    // Boost depth-1 stumps; the alpha-weighted vote should separate.
    let mut b = Boosting::of(RandomForest::new().n_trees(1).max_depth(1))
        .n_estimators(15)
        .seed(1);
    b.fit(&two_class()).unwrap();
    assert_eq!(b.predict(&probe()).unwrap(), vec![0.0, 1.0]);
}

#[test]
fn boosting_is_seed_reproducible() {
    let run = || {
        let mut b = Boosting::of(RandomForest::new().n_trees(1).max_depth(1))
            .n_estimators(10)
            .seed(7);
        b.fit(&two_class()).unwrap();
        b.predict(&probe()).unwrap()
    };
    assert_eq!(run(), run());
}

#[test]
fn boosting_rejects_invalid_learning_rates() {
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut model = Boosting::of(RandomForest::new()).learning_rate(rate);
        assert!(model.fit(&two_class()).is_err(), "accepted {rate}");
    }
}

#[cfg(feature = "model-selection")]
#[test]
fn stacking_predicts_clusters() {
    use crate::selection::StratifiedKFold;
    let mut s = Stacking::meta(RandomForest::new().n_trees(10))
        .base("rf", RandomForest::new().n_trees(10))
        .base("rf2", RandomForest::new().n_trees(15))
        .cv(StratifiedKFold::new(3));
    s.fit(&two_class()).unwrap();
    assert_eq!(s.predict(&probe()).unwrap(), vec![0.0, 1.0]);
}

#[cfg(all(feature = "onnx", feature = "model-selection"))]
#[test]
fn every_ensemble_family_round_trips_through_onnx() {
    use crate::logistic::LogisticRegression;
    use crate::onnx::InferenceModel;
    use crate::selection::StratifiedKFold;

    let dataset = two_class();
    let probe = probe();
    let mut models: Vec<Box<dyn Model>> = vec![
        Box::new(
            Voting::soft()
                .add("lr", LogisticRegression::new())
                .add("lr_l2", LogisticRegression::new().l2(0.01)),
        ),
        Box::new(
            Bagging::of(LogisticRegression::new())
                .n_estimators(3)
                .seed(4),
        ),
        Box::new(
            Boosting::of(RandomForest::new().n_trees(1).max_depth(1))
                .n_estimators(4)
                .seed(4),
        ),
        Box::new(
            Stacking::meta(LogisticRegression::new())
                .base("lr", LogisticRegression::new())
                .base("lr_l2", LogisticRegression::new().l2(0.01))
                .cv(StratifiedKFold::new(3)),
        ),
    ];

    for (index, model) in models.iter_mut().enumerate() {
        model.fit(&dataset).unwrap();
        let expected = model.predict(&probe).unwrap();
        let proto = model.to_onnx_proto().unwrap();
        if index == 2 {
            assert!(
                proto
                    .opset_import
                    .iter()
                    .any(|import| import.domain == "ai.onnx.ml"),
                "tree ensemble composition must preserve the ONNX-ML opset import"
            );
        }
        let path = std::env::temp_dir().join(format!(
            "millwright-ensemble-{}-{index}.onnx",
            std::process::id()
        ));
        onnx_export_rs::graph_builder::save_to_file(&proto, &path).unwrap();
        if index == 2 {
            if let Ok(python) = std::env::var("MILLWRIGHT_ONNX_CHECKER") {
                let script = "import onnx,sys; onnx.checker.check_model(onnx.load(sys.argv[1]))";
                let status = std::process::Command::new(python)
                    .args(["-c", script, path.to_str().unwrap()])
                    .status()
                    .unwrap();
                assert!(
                    status.success(),
                    "official ONNX checker rejected tree ensemble"
                );
            }
        }
        let loaded = InferenceModel::load(&path).unwrap();
        assert_eq!(loaded.predict(&probe).unwrap(), expected);
        std::fs::remove_file(path).ok();
    }
}
