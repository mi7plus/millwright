use super::*;
use crate::backends::smartcore::RandomForest;

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
    let mut v = Voting::soft().add("rf", RandomForest::new().n_trees(10));
    v.fit(&two_class()).unwrap();
    let proba = v.predict_proba(&probe()).unwrap();
    assert_eq!(proba.shape(), (2, 2)); // 2 rows, 2 classes
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
