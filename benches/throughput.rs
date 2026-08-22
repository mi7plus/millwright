//! Throughput benchmarks — the numbers behind the "Rust speed" claim.
//!
//! Covers the boundary conversion the whole design rests on (`Frame ->
//! DenseMatrix`) and fit/predict for the core models. Run with:
//! `cargo bench --features smartcore-backend`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use millwright::backends::smartcore::as_dense;
use millwright::prelude::*;

/// A synthetic, mildly-separable binary dataset of `n` rows × `p` features.
fn make_dataset(n: usize, p: usize) -> Dataset {
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cls = (i % 2) as f64;
        let row = (0..p)
            .map(|j| ((i * p + j) as f64).sin() + cls * 3.0)
            .collect();
        rows.push(row);
        y.push(cls);
    }
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap()
}

fn bench_boundary(c: &mut Criterion) {
    let ds = make_dataset(1000, 10);
    c.bench_function("frame_to_dense_1000x10", |b| {
        b.iter(|| as_dense(black_box(ds.features())).unwrap())
    });

    c.bench_function("frame_from_rows_1000x10", |b| {
        let rows = ds.features().as_rows();
        let cols = ds.features().columns().to_vec();
        b.iter(|| Frame::from_rows(black_box(rows.clone()), cols.clone()).unwrap())
    });
}

fn bench_models(c: &mut Criterion) {
    let ds = make_dataset(500, 8);

    c.bench_function("random_forest_fit_500x8", |b| {
        b.iter(|| {
            let mut rf = RandomForest::new().n_trees(20);
            rf.fit(black_box(&ds)).unwrap();
        })
    });

    let mut rf = RandomForest::new().n_trees(20);
    rf.fit(&ds).unwrap();
    c.bench_function("random_forest_predict_500x8", |b| {
        b.iter(|| rf.predict(black_box(ds.features())).unwrap())
    });

    c.bench_function("logistic_fit_500x8_100epochs", |b| {
        b.iter(|| {
            let mut lr = LogisticRegression::new().epochs(100);
            lr.fit(black_box(&ds)).unwrap();
        })
    });

    c.bench_function("standard_scaler_fit_transform_500x8", |b| {
        b.iter(|| {
            let mut s = StandardScaler::new();
            s.fit_transform(black_box(ds.features())).unwrap()
        })
    });
}

criterion_group!(benches, bench_boundary, bench_models);
criterion_main!(benches);
