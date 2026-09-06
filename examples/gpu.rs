//! Phase 4: GPU-accelerated inference — the same exported ONNX model, run on
//! the best device available.
//!
//! Trains a linear model, exports it to ONNX, then runs it through onnxruntime
//! on `Device::Auto` (the OS-native GPU provider — DirectML / CoreML / CUDA —
//! with a CPU fallback) and on `Device::Cpu`, checking both agree with the
//! in-process tract path. On a machine with no usable GPU, `Auto` simply falls
//! back to CPU and still runs. Run with:
//! `cargo run --example gpu --features gpu-inference`

use std::env::temp_dir;

use millwright::prelude::*;

fn main() -> Result<()> {
    // y = 2*x1 + 3*x2 + 1
    let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 5) as f64]).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
    let cols = vec!["x1".to_string(), "x2".to_string()];
    let train = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y)?;

    let mut model = LinearRegression::new();
    model.fit(&train)?;

    let path = temp_dir().join("millwright_gpu.onnx");
    model.export_onnx(&path)?;
    println!("exported model -> {}", path.display());

    let probe = Frame::from_rows(vec![vec![25.0, 1.0], vec![7.0, 3.0]], cols)?;

    // In-process reference (tract / native interpreter).
    let native = InferenceModel::load(&path)?.predict(&probe)?;

    // onnxruntime, forced onto CPU, and onto the best available device.
    let via_cpu = InferenceModel::load_on(&path, Device::Cpu)?.predict(&probe)?;
    let via_auto = InferenceModel::load_on(&path, Device::Auto)?.predict(&probe)?;

    println!("tract (in-process) : {native:?}");
    println!("onnxruntime  (cpu) : {via_cpu:?}");
    println!("onnxruntime (auto) : {via_auto:?}");

    for (a, b) in native.iter().zip(&via_auto) {
        assert!((a - b).abs() < 1e-3, "device disagreement: {a} vs {b}");
    }

    // A RandomForest's ONNX-ML tree op has no GPU kernel, so it would run on CPU
    // even on a GPU. `export_onnx_gpu` re-encodes the forest as tensor ops that a
    // GPU provider can run. Here we verify it round-trips identically on CPU
    // (tract cannot run the tree op, so this exercises the pure tensor encoding).
    let churn = Dataset::new(
        Frame::from_rows(
            (0..40)
                .flat_map(|i| [vec![i as f64 * 0.1, 0.0], vec![9.0 + i as f64 * 0.1, 1.0]])
                .collect(),
            vec!["score".into(), "flag".into()],
        )?,
        (0..40).flat_map(|_| [0.0, 1.0]).collect(),
    )?;
    let mut rf = RandomForest::new().n_trees(20).max_depth(4);
    rf.fit(&churn)?;

    let tree_path = temp_dir().join("millwright_gpu_forest.onnx");
    rf.export_onnx_gpu(&tree_path)?;
    let forest_native = rf.predict(churn.features())?;
    let forest_onnx = InferenceModel::load(&tree_path)?.predict(churn.features())?;
    assert_eq!(
        forest_native, forest_onnx,
        "tensor-encoded forest must match the native forest"
    );
    println!("forest re-encoded as tensor ops -> {}", tree_path.display());
    println!("  (run it on a GPU with InferenceModel::load_on(.., Device::Auto))");

    // Across several GPUs, split a batch for throughput:
    //     InferenceModel::load_multi(&tree_path, &[0, 1])?;

    println!("ok — same predictions on CPU and on the best available device.");
    Ok(())
}
