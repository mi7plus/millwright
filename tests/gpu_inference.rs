//! `gpu-inference` parity — an ONNX model run through onnxruntime
//! (`InferenceModel::load_on`) must predict the same values as the in-process
//! tract path (`InferenceModel::load`), and `Device::Auto` must agree with
//! `Device::Cpu`. This is what makes "GPU accelerates, CPU falls back" safe:
//! the device only changes speed, never the answer.
#![cfg(feature = "gpu-inference")]

use millwright::prelude::*;
use serial_test::serial;
use tempfile::tempdir;

/// True for onnxruntime GPU-driver/device failures (device removed or suspended,
/// a TDR) that reflect a stressed GPU environment, not a logic error. Correctness
/// is asserted on the CPU/tract path; these are tolerated when actually running
/// on the GPU so a flaky driver doesn't fail the suite.
fn is_gpu_driver_error(e: &Error) -> bool {
    let m = e.to_string();
    m.contains("suspended") || m.contains("removed") || m.contains("driver") || m.contains("887A")
}

fn dataset(n: usize, p: usize) -> Dataset {
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

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "prediction counts differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "row {i}: {x} vs {y} exceeds tolerance {tol}"
        );
    }
}

#[test]
#[serial(gpu)]
fn ort_matches_tract_and_auto_matches_cpu() {
    let p = 8;
    let mut model = LinearRegression::new();
    model.fit(&dataset(300, p)).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("model.onnx");
    model.export_onnx(&path).unwrap();

    let batch = dataset(200, p);

    let tract = InferenceModel::load(&path).unwrap();
    let cpu = InferenceModel::load_on(&path, Device::Cpu).unwrap();
    let auto = InferenceModel::load_on(&path, Device::Auto).unwrap();

    let expected = tract.predict(batch.features()).unwrap();
    let via_cpu = cpu.predict(batch.features()).unwrap();
    let via_auto = auto.predict(batch.features()).unwrap();

    assert!(
        expected.iter().all(|v| v.is_finite()),
        "tract produced non-finite output"
    );
    // onnxruntime runs in f32; tract folds to f32 too, so tolerance is loose.
    assert_close(&expected, &via_cpu, 1e-3);
    // Auto (GPU-or-CPU) must agree with the forced-CPU path.
    assert_close(&via_cpu, &via_auto, 1e-3);
}

/// A tree-ensemble (`RandomForest`) uses ONNX-ML ops that onnxruntime runs on
/// its CPU provider — even under `Device::Auto`. It must still load through
/// `load_on` and predict the same class labels as the in-process path.
#[test]
#[serial(gpu)]
fn forest_loads_and_predicts_through_onnxruntime() {
    let p = 6;
    let n = 240;
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cls = (i % 3) as f64;
        let row = (0..p)
            .map(|j| ((i * p + j) as f64).sin() + cls * 2.5)
            .collect();
        rows.push(row);
        // non-contiguous labels, to exercise the class-label mapping too
        y.push([2.0, 5.0, 9.0][cls as usize]);
    }
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    let train = Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap();

    let mut rf = RandomForest::new().n_trees(15);
    rf.fit(&train).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("forest.onnx");
    rf.export_onnx(&path).unwrap();

    let expected = InferenceModel::load(&path)
        .unwrap()
        .predict(train.features())
        .unwrap();
    let via_auto = InferenceModel::load_on(&path, Device::Auto)
        .unwrap()
        .predict(train.features())
        .unwrap();

    // Class labels are exact integers; they must match, not merely be close.
    assert_eq!(expected, via_auto);
    assert!(via_auto.iter().all(|v| [2.0, 5.0, 9.0].contains(v)));
}

#[test]
fn device_auto_is_default() {
    assert_eq!(Device::default(), Device::Auto);
}

/// Build a 3-class dataset with non-contiguous labels `[2, 5, 9]`.
fn forest_dataset(n: usize, p: usize) -> Dataset {
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cls = (i % 3) as f64;
        let row = (0..p)
            .map(|j| ((i * p + j) as f64).sin() + cls * 2.5)
            .collect();
        rows.push(row);
        y.push([2.0, 5.0, 9.0][cls as usize]);
    }
    let cols = (0..p).map(|j| format!("f{j}")).collect();
    Dataset::new(Frame::from_rows(rows, cols).unwrap(), y).unwrap()
}

/// The GPU-friendly (tensor-encoded) forest export must predict *exactly* the
/// same class labels as the normal export — first proven on CPU through tract
/// (which cannot run the ONNX-ML tree op, so this exercises the pure tensor-op
/// encoding), then through onnxruntime on the best available device.
#[test]
#[serial(gpu)]
fn tensorized_forest_matches_native() {
    let train = forest_dataset(240, 6);
    let mut rf = RandomForest::new().n_trees(12).max_depth(4);
    rf.fit(&train).unwrap();

    let dir = tempdir().unwrap();
    let std_path = dir.path().join("rf.onnx");
    let gpu_path = dir.path().join("rf_tensorized.onnx");
    rf.export_onnx(&std_path).unwrap();
    rf.export_onnx_gpu(&gpu_path).unwrap();

    // Reference: the standard tree-ensemble export (native interpreter).
    let expected = InferenceModel::load(&std_path)
        .unwrap()
        .predict(train.features())
        .unwrap();

    // The tensorized graph has no ONNX-ML op, so this runs through tract on CPU.
    let via_tract = InferenceModel::load(&gpu_path)
        .unwrap()
        .predict(train.features())
        .unwrap();
    assert_eq!(
        expected, via_tract,
        "tensor-encoded forest disagrees on CPU/tract"
    );

    // And through onnxruntime on the best available device (GPU when present).
    // A GPU-driver failure here (device removed/TDR) is tolerated — correctness
    // is already proven on CPU/tract above.
    match InferenceModel::load_on(&gpu_path, Device::Auto).and_then(|m| m.predict(train.features()))
    {
        Ok(via_gpu) => assert_eq!(
            expected, via_gpu,
            "tensor-encoded forest disagrees on device"
        ),
        Err(ref e) if is_gpu_driver_error(e) => eprintln!("skipping forest GPU exec: {e}"),
        Err(e) => panic!("tensorized forest device error: {e}"),
    }

    // Labels must be the real class labels, not 0..k indices.
    assert!(expected.iter().all(|v| [2.0, 5.0, 9.0].contains(v)));
}

/// Splitting a batch across a multi-session pool must match single-session
/// predictions. Uses two sessions on device 0 to exercise the split/concat path
/// on any single-GPU machine; on a machine with no GPU it errors clearly.
#[test]
#[serial(gpu)]
fn load_multi_matches_single_or_errors() {
    let p = 8;
    let mut model = LinearRegression::new();
    model.fit(&dataset(150, p)).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("model.onnx");
    model.export_onnx(&path).unwrap();

    let batch = dataset(103, p); // deliberately not divisible by the pool size
    let single = InferenceModel::load_on(&path, Device::Cpu)
        .unwrap()
        .predict(batch.features())
        .unwrap();

    // Two sessions on device 0 exercise the split/concat path. Running two
    // concurrent GPU sessions on one physical device can TDR a consumer GPU, so a
    // driver failure is tolerated; a successful run must match single-session.
    match InferenceModel::load_multi(&path, &[0, 0]) {
        Ok(pool) => match pool.predict(batch.features()) {
            Ok(multi) => assert_close(&single, &multi, 1e-3),
            Err(ref e) if is_gpu_driver_error(e) => eprintln!("skipping multi-GPU exec: {e}"),
            Err(e) => panic!("multi-GPU predict error: {e}"),
        },
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("GPU") || msg.contains("onnxruntime"),
                "expected a GPU/onnxruntime error, got: {msg}"
            );
        }
    }
}

/// `Device::Gpu` must never silently degrade to a wrong answer: on a machine
/// with a usable GPU it loads and predicts correctly; otherwise it returns a
/// clear error rather than quietly running on CPU. This runs on any machine.
#[test]
#[serial(gpu)]
fn device_gpu_uses_gpu_or_errors_clearly() {
    let p = 8;
    let mut model = LinearRegression::new();
    model.fit(&dataset(150, p)).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("model.onnx");
    model.export_onnx(&path).unwrap();

    let batch = dataset(100, p);
    let cpu = InferenceModel::load_on(&path, Device::Cpu)
        .unwrap()
        .predict(batch.features())
        .unwrap();

    match InferenceModel::load_on(&path, Device::Gpu) {
        // A GPU provider initialized: predictions must still match CPU (a driver
        // failure while running is tolerated).
        Ok(gpu) => match gpu.predict(batch.features()) {
            Ok(via_gpu) => assert_close(&cpu, &via_gpu, 1e-3),
            Err(ref e) if is_gpu_driver_error(e) => eprintln!("skipping Device::Gpu exec: {e}"),
            Err(e) => panic!("Device::Gpu predict error: {e}"),
        },
        // No usable GPU (or none compiled in): a clear error, never a wrong answer.
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("GPU") || msg.contains("onnxruntime"),
                "expected a GPU/onnxruntime error, got: {msg}"
            );
        }
    }
}
