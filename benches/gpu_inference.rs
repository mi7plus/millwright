//! GPU vs CPU inference throughput — the numbers behind the `gpu-inference`
//! feature.
//!
//! Exports a linear model (a matmul-heavy graph, the shape onnxruntime's GPU
//! providers actually accelerate) and times batched prediction through the
//! onnxruntime CPU provider against `Device::Auto` (the OS-native GPU provider
//! with CPU fallback). On a machine with no usable GPU the two converge — that
//! is the honest baseline. Run with:
//! `cargo bench --features gpu-inference --bench gpu_inference`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use millwright::prelude::*;
use tempfile::tempdir;

/// A synthetic regression dataset of `n` rows × `p` features with a linear
/// target — exports to a Gemm-shaped ONNX graph.
fn make_dataset(n: usize, p: usize) -> Dataset {
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<f64> = (0..p)
            .map(|j| ((i * p + j) as f64).sin() + (i as f64) * 0.01)
            .collect();
        let target = row
            .iter()
            .enumerate()
            .map(|(j, v)| (j as f64 + 1.0) * v)
            .sum();
        rows.push(row);
        y.push(target);
    }
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap()
}

fn bench_inference(c: &mut Criterion) {
    let p = 16;
    let mut model = LinearRegression::new();
    model.fit(&make_dataset(500, p)).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("model.onnx");
    model.export_onnx(&path).unwrap();

    for &rows in &[1_000usize, 10_000, 100_000] {
        let batch = make_dataset(rows, p);

        let cpu = InferenceModel::load_on(&path, Device::Cpu).unwrap();
        c.bench_function(&format!("ort_cpu_predict_{rows}x{p}"), |b| {
            b.iter(|| cpu.predict(black_box(batch.features())).unwrap())
        });

        let auto = InferenceModel::load_on(&path, Device::Auto).unwrap();
        c.bench_function(&format!("ort_auto_predict_{rows}x{p}"), |b| {
            b.iter(|| auto.predict(black_box(batch.features())).unwrap())
        });
    }
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
