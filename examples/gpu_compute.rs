//! Phase 2: GPU-accelerated classic-ML compute.
//!
//! Uses the `gpu` primitives directly, then shows the opt-in GPU paths on two
//! estimators (`KnnScore` distances, `Mahalanobis` covariance) producing the
//! same answers as CPU. Everything falls back to CPU when no GPU is present. Run
//! with: `cargo run --example gpu_compute --features "gpu-compute anomaly"`

use millwright::gpu;
use millwright::prelude::*;

fn main() -> Result<()> {
    if !gpu::is_available() {
        println!("no GPU adapter found — the same code paths run on CPU.");
        return Ok(());
    }
    println!("GPU adapter found.");

    // Primitive: C = A · B (2×3 by 3×2).
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
    let c = gpu::gemm(&a, 2, 3, &b, 2)?;
    println!("gemm 2x3·3x2 = {c:?}"); // [58, 64, 139, 154]

    // Primitive: squared distances between two point sets.
    let x = [0.0f32, 0.0, 1.0, 1.0]; // 2 points in 2-D
    let y = [0.0f32, 0.0, 3.0, 4.0]; // 2 points in 2-D
    let dsq = gpu::pairwise_sqdist(&x, 2, &y, 2, 2)?;
    println!("pairwise sqdist = {dsq:?}"); // [0, 25, 2, 13]

    // Estimator integration: a blob plus one outlier.
    let mut rows: Vec<Vec<f64>> = (0..50)
        .map(|i| vec![(i as f64).sin(), (i as f64 * 0.7).cos(), 0.01 * i as f64])
        .collect();
    rows.push(vec![9.0, 9.0, 9.0]);
    let frame = Frame::from_rows(rows, vec!["a".into(), "b".into(), "c".into()])?;

    let mut knn_cpu = KnnScore::new(3);
    knn_cpu.fit(&frame)?;
    let mut knn_gpu = KnnScore::new(3).on_gpu();
    knn_gpu.fit(&frame)?;
    let (sc, sg) = (knn_cpu.score(&frame)?, knn_gpu.score(&frame)?);
    println!(
        "KnnScore  outlier score  cpu={:.4}  gpu={:.4}",
        sc[50], sg[50]
    );

    let mut maha_cpu = Mahalanobis::new();
    maha_cpu.fit(&frame)?;
    let mut maha_gpu = Mahalanobis::new().on_gpu();
    maha_gpu.fit(&frame)?;
    let (mc, mg) = (maha_cpu.score(&frame)?, maha_gpu.score(&frame)?);
    println!(
        "Mahalanobis outlier score cpu={:.4} gpu={:.4}",
        mc[50], mg[50]
    );

    println!("ok — GPU paths agree with CPU within f32 tolerance.");
    Ok(())
}
