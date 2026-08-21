//! Phase 2: two backends, one contract — plus Bayesian search.
//!
//! Shows the linfa backend (clustering + PCA) proving the Frame boundary
//! conversion, and TPE hyperparameter search over the smartcore backend behind
//! the same search API as grid/random search. Run with:
//! `cargo run --example backends --features "smartcore-backend linfa-backend hpo"`

use millwright::prelude::*;

fn main() -> Result<()> {
    // Two well-separated blobs on two features.
    let x = blobs();

    // --- linfa backend: unsupervised clustering + dimensionality reduction ---
    let mut km = KMeans::new(2);
    km.fit(&x)?;
    println!("k-means labels : {:?}", km.predict(&x)?);

    let dbscan = Dbscan::new(3).tolerance(1.0);
    println!("dbscan labels  : {:?}", dbscan.fit_predict(&x)?);

    let mut pca = Pca::new(1);
    let reduced = pca.fit_transform(&x)?;
    println!("pca -> {} cols ({} rows)", reduced.ncols(), reduced.nrows());

    // --- smartcore backend: Bayesian (TPE) search over a space ---
    let train = Dataset::new(x.clone(), labels())?;
    let space = SearchSpace::new().int("max_depth", 1, 12);
    let search = BayesSearch::new(RandomForest::new(), space)
        .n_trials(12)
        .seed(0)
        .cv(StratifiedKFold::new(4))
        .scoring(Metric::F1)
        .fit(&train)?;

    println!(
        "TPE search     : best F1 = {:.3} at {:?}",
        search.best_score(),
        search.best_params()
    );
    let probe = Frame::from_rows(
        vec![vec![0.2, 0.2], vec![9.3, 9.3]],
        vec!["a".into(), "b".into()],
    )?;
    println!("predictions    : {:?}", search.predict(&probe)?);

    println!("ok — two backends, one contract.");
    Ok(())
}

fn blobs() -> Frame {
    let mut rows = Vec::new();
    for i in 0..15 {
        rows.push(vec![i as f64 * 0.05, i as f64 * 0.05]);
        rows.push(vec![9.0 + i as f64 * 0.05, 9.0 + i as f64 * 0.05]);
    }
    Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap()
}

fn labels() -> Vec<f64> {
    (0..15).flat_map(|_| [0.0, 1.0]).collect()
}
