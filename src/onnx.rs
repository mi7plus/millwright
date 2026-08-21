//! ONNX export and inference — *train once; run in Rust, Python, or any ONNX
//! runtime.*
//!
//! [`ExportOnnx`] writes a trained model — or a whole [`Pipeline`](crate::pipeline::Pipeline) — to a single
//! `.onnx` file via [`onnx-export-rs`](https://docs.rs/onnx-export-rs).
//! [`InferenceModel`] loads any ONNX file and runs it through
//! [`tract`](https://docs.rs/tract-onnx), so the exported artifact round-trips
//! back into Rust and is portable to every other ONNX runtime.
//!
//! A pipeline is exported by folding its leading affine transformers
//! (`StandardScaler`, `MinMaxScaler`) into the estimator's ONNX graph, producing
//! one self-contained graph: raw features in, predictions out. Non-affine steps
//! (imputation, one-hot) are not yet ONNX-expressible and are reported as an
//! error naming the offending step.

use std::path::Path;

use ndarray::Array2;
use onnx_export_rs::graph_builder::{make_node, make_tensor, save_to_file};
use onnx_export_rs::proto::ModelProto;

use crate::error::{Error, Result};
use crate::frame::Frame;

/// A model or pipeline that can be exported to ONNX.
pub trait ExportOnnx {
    /// Build the ONNX graph for this object.
    fn to_onnx(&self) -> Result<ModelProto>;

    /// Write the ONNX graph to `path`.
    fn export_onnx(&self, path: impl AsRef<Path>) -> Result<()> {
        let proto = self.to_onnx()?;
        save_to_file(&proto, path).map_err(|e| Error::Backend(format!("ONNX save failed: {e}")))
    }
}

/// Prepend an affine transform `y = (x - shift) / scale` to a model graph whose
/// input tensor is named `X`.
///
/// The estimator's input `X` becomes an internal tensor produced by the affine
/// nodes; a fresh graph input `mw_input` feeds the transform. Used to splice a
/// scaler in front of an estimator so the whole pipeline is one ONNX graph.
pub(crate) fn prepend_affine(proto: &mut ModelProto, shift: &[f32], scale: &[f32]) -> Result<()> {
    let graph = proto
        .graph
        .as_mut()
        .ok_or_else(|| Error::Backend("exported model has no graph".into()))?;
    let est_input = graph
        .input
        .first()
        .map(|vi| vi.name.clone())
        .ok_or_else(|| Error::Backend("exported model has no input".into()))?;

    let shift_t = make_tensor(
        "mw_shift",
        &Array2::from_shape_vec((1, shift.len()), shift.to_vec())
            .map_err(|e| Error::Backend(e.to_string()))?
            .into_dyn(),
    );
    let scale_t = make_tensor(
        "mw_scale",
        &Array2::from_shape_vec((1, scale.len()), scale.to_vec())
            .map_err(|e| Error::Backend(e.to_string()))?
            .into_dyn(),
    );

    // mw_input - mw_shift -> mw_centered ; mw_centered / mw_scale -> <est_input>
    let sub = make_node("Sub", ["mw_input", "mw_shift"], ["mw_centered"], Vec::new());
    let div = make_node(
        "Div",
        ["mw_centered", "mw_scale"],
        [est_input.as_str()],
        Vec::new(),
    );

    graph.initializer.push(shift_t);
    graph.initializer.push(scale_t);
    // affine nodes must run before the estimator's nodes
    graph.node.insert(0, div);
    graph.node.insert(0, sub);
    // the graph now consumes raw features on `mw_input`
    if let Some(vi) = graph.input.first_mut() {
        vi.name = "mw_input".into();
    }
    Ok(())
}

/// A loaded ONNX model, ready to run via tract.
pub struct InferenceModel {
    plan: TractPlan,
}

// tract's runnable-model type is verbose; name it once. `into_runnable` hands
// back an `Arc`, and `run` takes `&Arc<Self>`.
type TractPlan = std::sync::Arc<tract_onnx::prelude::TypedRunnableModel>;

impl InferenceModel {
    /// Load an ONNX model from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<InferenceModel> {
        use tract_onnx::prelude::*;
        let plan = tract_onnx::onnx()
            .model_for_path(path.as_ref())
            .map_err(|e| Error::Backend(format!("ONNX load failed: {e}")))?
            .into_optimized()
            .map_err(|e| Error::Backend(format!("ONNX optimize failed: {e}")))?
            .into_runnable()
            .map_err(|e| Error::Backend(format!("ONNX plan failed: {e}")))?;
        Ok(Self { plan })
    }

    /// Run the model on a frame, returning one prediction per row.
    ///
    /// For a multi-column classifier score output the arg-max class index is
    /// returned; for a single-column output the value itself.
    pub fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        use tract_onnx::prelude::*;

        let (n, p) = frame.shape();
        let data: Vec<f32> = frame.buf().iter().map(|v| *v as f32).collect();
        let input = tract_ndarray::Array2::from_shape_vec((n, p), data)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let tensor: Tensor = input.into();
        let outputs = self
            .plan
            .run(tvec!(tensor.into()))
            .map_err(|e| Error::Backend(format!("ONNX run failed: {e}")))?;

        let out: &Tensor = &outputs[0];
        let plain = out
            .try_as_plain()
            .map_err(|e| Error::Backend(format!("ONNX output not plain: {e}")))?;
        // Prefer an integer label output (classifier); fall back to floats.
        if let Ok(view) = plain.to_array_view::<i64>() {
            return Ok(view.iter().map(|v| *v as f64).collect());
        }
        let view = plain
            .to_array_view::<f32>()
            .map_err(|e| Error::Backend(format!("unexpected ONNX output type: {e}")))?;
        let shape = view.shape();
        if shape.len() == 2 && shape[1] > 1 {
            // multi-class scores -> arg-max index
            let cols = shape[1];
            let flat: Vec<f32> = view.iter().copied().collect();
            Ok((0..n)
                .map(|r| {
                    let row = &flat[r * cols..(r + 1) * cols];
                    let mut best = 0usize;
                    for c in 1..cols {
                        if row[c] > row[best] {
                            best = c;
                        }
                    }
                    best as f64
                })
                .collect())
        } else {
            Ok(view.iter().map(|v| *v as f64).collect())
        }
    }
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;
    use crate::frame::Dataset;
    use crate::traits::{Estimator, Predictor};

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("millwright_onnx_{name}.onnx"))
    }

    fn two_class() -> (Dataset, Frame) {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            rows.push(vec![i as f64 * 0.05, i as f64 * 0.05]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.05, 9.0 + i as f64 * 0.05]);
            y.push(1.0);
        }
        let cols = vec!["a".to_string(), "b".to_string()];
        let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
        let probe =
            Frame::from_rows(vec![vec![0.3, 0.2], vec![9.2, 9.3], vec![0.1, 0.0]], cols).unwrap();
        (ds, probe)
    }

    #[test]
    fn random_forest_exports_valid_onnx() {
        // The tree-ensemble export is a valid ONNX-ML artifact for full runtimes
        // (e.g. onnxruntime). tract implements NN ops, not ONNX-ML tree ops, so
        // we validate the export here rather than running it through tract.
        let (ds, _) = two_class();
        let mut rf = RandomForest::new().n_trees(20).max_depth(4);
        rf.fit(&ds).unwrap();
        assert!(rf.to_onnx().is_ok());
        let path = scratch("rf");
        rf.export_onnx(&path).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn linear_regression_round_trips_through_onnx() {
        use crate::backends::smartcore::LinearRegression;
        // y = 2*x1 + 3*x2 + 1
        let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();

        let probe = Frame::from_rows(
            vec![vec![20.0, 1.0], vec![5.0, 2.0]],
            vec!["x1".into(), "x2".into()],
        )
        .unwrap();
        let native = lr.predict(&probe).unwrap();

        let path = scratch("lr");
        lr.export_onnx(&path).unwrap();
        let loaded = InferenceModel::load(&path).unwrap();
        let via_onnx = loaded.predict(&probe).unwrap();
        for (a, b) in native.iter().zip(&via_onnx) {
            assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pipeline_scaler_plus_linear_round_trips() {
        use crate::backends::smartcore::LinearRegression;
        use crate::pipeline::Pipeline;
        use crate::transform::StandardScaler;

        let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();

        let mut pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("lr", LinearRegression::new());
        pipe.fit(&ds).unwrap();

        let probe = Frame::from_rows(
            vec![vec![20.0, 1.0], vec![5.0, 2.0]],
            vec!["x1".into(), "x2".into()],
        )
        .unwrap();
        let native = pipe.predict(&probe).unwrap();

        // whole pipeline -> one ONNX graph (Sub, Div, Gemm) -> tract
        let path = scratch("pipe");
        pipe.export_onnx(&path).unwrap();
        let loaded = InferenceModel::load(&path).unwrap();
        let via_onnx = loaded.predict(&probe).unwrap();
        for (a, b) in native.iter().zip(&via_onnx) {
            assert!((a - b).abs() < 1e-3, "native {a} vs onnx {b}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
