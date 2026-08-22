// Verifies the design brief's ensemble spotlight compiles & runs against the API.
#![cfg(all(feature = "ensemble", feature = "model-selection"))]
use millwright::prelude::*;

#[test]
fn brief_ensemble_snippet_compiles_and_runs() {
    // Two well-separated, class-balanced clusters — learnable by every member,
    // including boosted depth-1 stumps.
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..12 {
        rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        y.push(0.0);
        rows.push(vec![9.0 + i as f64 * 0.1, 9.0 + i as f64 * 0.1]);
        y.push(1.0);
    }
    let x = Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap();
    let train = Dataset::new(x.clone(), y).unwrap();

    let vote = Voting::soft()
        .add("lr", LogisticRegression::new())
        .add("rf", RandomForest::new())
        .add("svc", Svc::rbf());

    let stack = Stacking::meta(LogisticRegression::new())
        .base("rf", RandomForest::new())
        .base("knn", Knn::k(5))
        .cv(StratifiedKFold::new(3));

    let bag = Bagging::of(Svc::rbf()).n_estimators(10);
    let boost = Boosting::of(RandomForest::new().n_trees(1).max_depth(1)).n_estimators(20);

    for mut m in [
        Box::new(vote) as Box<dyn Model>,
        Box::new(stack),
        Box::new(bag),
        Box::new(boost),
    ] {
        m.fit(&train).unwrap();
        assert_eq!(m.predict(&x).unwrap().len(), 24);
    }
}
