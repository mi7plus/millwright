//! `gpu-compute` parity — the GPU primitives must match a CPU reference within
//! floating-point tolerance (both run in f32; accumulation order differs). The
//! tests skip when no GPU adapter is available, and run serially so parallel
//! test threads don't contend on one device.
#![cfg(feature = "gpu-compute")]

use millwright::gpu;
use serial_test::serial;

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

fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let scale = 1.0 + c.abs();
        assert!(
            (g - c).abs() / scale <= tol,
            "index {i}: gpu {g} vs cpu {c} exceeds tol {tol}"
        );
    }
}

#[test]
#[serial(gpu)]
fn gemm_matches_cpu() {
    if !gpu::is_available() {
        eprintln!("no GPU adapter; skipping gemm_matches_cpu");
        return;
    }
    let (m, k, n) = (17usize, 23usize, 13usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.021).cos()).collect();

    let gpu = gpu::gemm(&a, m, k, &b, n).unwrap();
    let cpu = cpu_gemm(&a, m, k, &b, n);
    assert_close(&gpu, &cpu, 1e-3);
}

#[test]
#[serial(gpu)]
fn pairwise_sqdist_matches_cpu() {
    if !gpu::is_available() {
        eprintln!("no GPU adapter; skipping pairwise_sqdist_matches_cpu");
        return;
    }
    let (n, m, d) = (21usize, 15usize, 7usize);
    let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.05).sin()).collect();
    let y: Vec<f32> = (0..m * d).map(|i| (i as f32 * 0.03).cos()).collect();

    let gpu = gpu::pairwise_sqdist(&x, n, &y, m, d).unwrap();
    let cpu = cpu_sqdist(&x, n, &y, m, d);
    assert_close(&gpu, &cpu, 1e-3);
}

#[test]
fn dimension_mismatch_errors() {
    // Length checks happen before any GPU call, so this runs without a device.
    assert!(gpu::gemm(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 2.0, 3.0, 4.0], 2).is_err());
    assert!(gpu::pairwise_sqdist(&[1.0, 2.0], 1, &[1.0], 1, 2).is_err());
}

// A well-conditioned blob (rows 0..n-1) plus one far outlier (last row).
#[cfg(feature = "anomaly")]
fn anomaly_frame(n: usize, p: usize) -> millwright::prelude::Frame {
    use millwright::prelude::Frame;
    let mut rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..p)
                .map(|j| ((i as f64 + 1.0) * (j as f64 + 1.3)).sin() + 0.01 * i as f64)
                .collect()
        })
        .collect();
    rows.push(vec![9.0; p]); // the outlier
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    Frame::from_rows(rows, cols).unwrap()
}

fn rel_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() / (1.0 + x.abs()) <= tol,
            "row {i}: {x} vs {y} exceeds rel tol {tol}"
        );
    }
}

// KnnScore's GPU distance path must match the CPU path within f32 tolerance.
#[cfg(feature = "anomaly")]
#[test]
#[serial(gpu)]
fn knn_score_gpu_matches_cpu() {
    use millwright::prelude::KnnScore;
    if !gpu::is_available() {
        eprintln!("no GPU adapter; skipping knn_score_gpu_matches_cpu");
        return;
    }
    let f = anomaly_frame(60, 5);
    let mut cpu = KnnScore::new(3);
    cpu.fit(&f).unwrap();
    let mut gpu_scorer = KnnScore::new(3).on_gpu();
    gpu_scorer.fit(&f).unwrap();

    let sc = cpu.score(&f).unwrap();
    let sg = gpu_scorer.score(&f).unwrap();
    rel_close(&sc, &sg, 1e-3);
    // The outlier (last row) is the most anomalous on both paths.
    let argmax = |v: &[f64]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    assert_eq!(argmax(&sc), argmax(&sg));
}

// Mahalanobis's GPU covariance (a gemm) must match the CPU path closely. The
// covariance is inverted, so tolerance is looser than the raw gemm.
#[cfg(feature = "anomaly")]
#[test]
#[serial(gpu)]
fn mahalanobis_gpu_matches_cpu() {
    use millwright::prelude::Mahalanobis;
    if !gpu::is_available() {
        eprintln!("no GPU adapter; skipping mahalanobis_gpu_matches_cpu");
        return;
    }
    let f = anomaly_frame(80, 4);
    let mut cpu = Mahalanobis::new();
    cpu.fit(&f).unwrap();
    let mut gpu_scorer = Mahalanobis::new().on_gpu();
    gpu_scorer.fit(&f).unwrap();

    let sc = cpu.score(&f).unwrap();
    let sg = gpu_scorer.score(&f).unwrap();
    rel_close(&sc, &sg, 2e-2);
    let argmax = |v: &[f64]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    assert_eq!(argmax(&sc), argmax(&sg));
}
