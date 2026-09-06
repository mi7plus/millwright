//! GPU vs CPU for the classic-ML compute primitives — the numbers behind the
//! `gpu-compute` feature.
//!
//! Times [`millwright::gpu::gemm`] and [`millwright::gpu::pairwise_sqdist`]
//! against straightforward CPU implementations. GPU wins grow with problem size;
//! at small sizes the dispatch + transfer overhead dominates. Run with:
//! `cargo bench --features gpu-compute --bench gpu_compute`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use millwright::gpu;

fn cpu_gemm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn cpu_sqdist(x: &[f32], n: usize, y: &[f32], m: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0f32;
            for l in 0..d {
                let diff = x[i * d + l] - y[j * d + l];
                s += diff * diff;
            }
            out[i * m + j] = s;
        }
    }
    out
}

fn bench_compute(c: &mut Criterion) {
    if !gpu::is_available() {
        eprintln!("no GPU adapter; skipping gpu_compute benchmarks");
        return;
    }

    // GEMM: square matrices of increasing size.
    for &size in &[128usize, 256, 512] {
        let a: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.002).cos()).collect();
        c.bench_function(&format!("gemm_cpu_{size}"), |bn| {
            bn.iter(|| cpu_gemm(black_box(&a), size, size, black_box(&b), size))
        });
        c.bench_function(&format!("gemm_gpu_{size}"), |bn| {
            bn.iter(|| gpu::gemm(black_box(&a), size, size, black_box(&b), size).unwrap())
        });
    }

    // Pairwise squared distances: n×m over dimension d.
    for &(n, m, d) in &[(512usize, 512usize, 16usize), (1024, 1024, 32)] {
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let y: Vec<f32> = (0..m * d).map(|i| (i as f32 * 0.02).cos()).collect();
        c.bench_function(&format!("sqdist_cpu_{n}x{m}x{d}"), |bn| {
            bn.iter(|| cpu_sqdist(black_box(&x), n, black_box(&y), m, d))
        });
        c.bench_function(&format!("sqdist_gpu_{n}x{m}x{d}"), |bn| {
            bn.iter(|| gpu::pairwise_sqdist(black_box(&x), n, black_box(&y), m, d).unwrap())
        });
    }
}

criterion_group!(benches, bench_compute);
criterion_main!(benches);
