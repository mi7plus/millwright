//! ONNX export and inference — *train once; run in Rust, Python, or any ONNX
//! runtime.*
//!
//! [`ExportOnnx`] writes a trained model — or a whole [`Pipeline`](crate::pipeline::Pipeline) — to a single
//! `.onnx` file via [`onnx-export-rs`](https://docs.rs/onnx-export-rs).
//! [`InferenceModel`] loads any ONNX file and runs it: linear / NN graphs
//! through [`tract`](https://docs.rs/tract-onnx), and the ONNX-ML tree-ensemble
//! ops tract doesn't implement (from an exported forest) through a small native
//! interpreter. So the exported artifact always round-trips back into Rust — a
//! `RandomForest` included — and stays portable to every other ONNX runtime.
//!
//! A pipeline is exported by splicing each leading transformer that is
//! ONNX-expressible in front of the estimator's graph, in order — scalers as an
//! affine `(x - shift) / scale`, imputers as `Where(IsNaN(x), fill, x)` — so the
//! result is one self-contained graph: raw features in, predictions out. A step
//! that changes feature width (one-hot encoding) is not yet expressible and is
//! reported as an error naming the offending step.

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

/// A preprocessing step expressible as ONNX graph nodes, prepended in front of
/// an estimator so a whole pipeline becomes one graph. Each maps a same-width
/// feature tensor to another.
pub enum Prefix {
    /// `y = (x - shift) / scale`, elementwise per column (scalers).
    Affine { shift: Vec<f64>, scale: Vec<f64> },
    /// Replace missing (`NaN`) values with a per-column constant (imputers).
    Impute { fill: Vec<f64> },
}

fn row_tensor(name: &str, vals: &[f64]) -> Result<onnx_export_rs::proto::TensorProto> {
    let f: Vec<f32> = vals.iter().map(|v| *v as f32).collect();
    Ok(make_tensor(
        name,
        &Array2::from_shape_vec((1, f.len()), f)
            .map_err(|e| Error::Backend(e.to_string()))?
            .into_dyn(),
    ))
}

/// Prepend a chain of preprocessing [`Prefix`] steps to a model graph, so the
/// graph consumes raw features on a fresh `mw_input` and threads them through
/// the prefixes into the estimator's original input.
pub(crate) fn prepend_prefixes(proto: &mut ModelProto, prefixes: &[Prefix]) -> Result<()> {
    let graph = proto
        .graph
        .as_mut()
        .ok_or_else(|| Error::Backend("exported model has no graph".into()))?;
    let est_input = graph
        .input
        .first()
        .map(|vi| vi.name.clone())
        .ok_or_else(|| Error::Backend("exported model has no input".into()))?;

    let mut nodes = Vec::new();
    let mut inits = Vec::new();
    let mut cur = "mw_input".to_string();
    for (i, prefix) in prefixes.iter().enumerate() {
        // the last prefix feeds the estimator's original input
        let out = if i + 1 == prefixes.len() {
            est_input.clone()
        } else {
            format!("mw_pre{i}")
        };
        match prefix {
            Prefix::Impute { fill } => {
                let mask = format!("mw_isnan{i}");
                let fill_name = format!("mw_fill{i}");
                nodes.push(make_node(
                    "IsNaN",
                    [cur.as_str()],
                    [mask.as_str()],
                    Vec::new(),
                ));
                // Where(mask, fill, x): pick the fill where x is NaN, else x.
                nodes.push(make_node(
                    "Where",
                    [mask.as_str(), fill_name.as_str(), cur.as_str()],
                    [out.as_str()],
                    Vec::new(),
                ));
                inits.push(row_tensor(&fill_name, fill)?);
            }
            Prefix::Affine { shift, scale } => {
                let centered = format!("mw_cent{i}");
                let shift_name = format!("mw_shift{i}");
                let scale_name = format!("mw_scale{i}");
                nodes.push(make_node(
                    "Sub",
                    [cur.as_str(), shift_name.as_str()],
                    [centered.as_str()],
                    Vec::new(),
                ));
                nodes.push(make_node(
                    "Div",
                    [centered.as_str(), scale_name.as_str()],
                    [out.as_str()],
                    Vec::new(),
                ));
                inits.push(row_tensor(&shift_name, shift)?);
                inits.push(row_tensor(&scale_name, scale)?);
            }
        }
        cur = out;
    }

    for init in inits {
        graph.initializer.push(init);
    }
    // prefix nodes must run before the estimator's nodes, in order
    nodes.append(&mut graph.node);
    graph.node = nodes;
    if let Some(vi) = graph.input.first_mut() {
        vi.name = "mw_input".into();
    }
    Ok(())
}

/// A loaded ONNX model, ready to run.
///
/// tract runs NN / linear graphs; ONNX-ML ops it does not implement (tree
/// ensembles, from an exported forest) are evaluated by a small native
/// interpreter over the ops Millwright's own exporter emits. So a model exported
/// here always round-trips back in — a `RandomForest` included.
#[derive(Clone)]
pub struct InferenceModel {
    backend: std::sync::Arc<Backend>,
}

enum Backend {
    Tract(TractPlan),
    Native(native::NativeGraph),
}

// tract's runnable-model type is verbose; name it once. `into_runnable` hands
// back an `Arc`, and `run` takes `&Arc<Self>`.
type TractPlan = std::sync::Arc<tract_onnx::prelude::TypedRunnableModel>;

impl InferenceModel {
    /// Load an ONNX model from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<InferenceModel> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Backend(format!("ONNX read failed: {e}")))?;

        // Decode to inspect the ops. ONNX-ML ops (tree ensembles) tract cannot
        // run are handled by the native interpreter; everything else via tract.
        use prost::Message;
        let proto = ModelProto::decode(&bytes[..])
            .map_err(|e| Error::Backend(format!("ONNX decode failed: {e}")))?;
        if native::needs_native(&proto) {
            let graph = native::NativeGraph::from_proto(&proto)?;
            return Ok(Self {
                backend: std::sync::Arc::new(Backend::Native(graph)),
            });
        }

        use tract_onnx::prelude::*;
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| Error::Backend(format!("ONNX load failed: {e}")))?
            .into_optimized()
            .map_err(|e| Error::Backend(format!("ONNX optimize failed: {e}")))?
            .into_runnable()
            .map_err(|e| Error::Backend(format!("ONNX plan failed: {e}")))?;
        Ok(Self {
            backend: std::sync::Arc::new(Backend::Tract(plan)),
        })
    }

    /// Run the model on a frame, returning one prediction per row.
    ///
    /// For a multi-column classifier score output the arg-max class index is
    /// returned; for a single-column output the value itself.
    pub fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        match &*self.backend {
            Backend::Native(g) => g.run(frame),
            Backend::Tract(plan) => Self::tract_predict(plan, frame),
        }
    }

    fn tract_predict(plan: &TractPlan, frame: &Frame) -> Result<Vec<f64>> {
        use tract_onnx::prelude::*;

        let (n, p) = frame.shape();
        let data: Vec<f32> = frame.buf().iter().map(|v| *v as f32).collect();
        let input = tract_ndarray::Array2::from_shape_vec((n, p), data)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let tensor: Tensor = input.into();
        let outputs = plan
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

impl crate::traits::Estimator for InferenceModel {
    fn name(&self) -> &'static str {
        "InferenceModel"
    }

    /// No-op: the model arrives already trained.
    fn fit(&mut self, _dataset: &crate::frame::Dataset) -> Result<()> {
        Ok(())
    }
}

impl crate::traits::Predictor for InferenceModel {
    fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        InferenceModel::predict(self, frame)
    }
}

/// A tiny native interpreter for the ONNX-ML ops tract does not implement.
///
/// It only handles the ops Millwright's own exporter emits — a leading affine
/// map (`Sub`/`Div`, from a folded scaler), a `TreeEnsembleRegressor` (a forest,
/// aggregating leaf weights per class), and a final `ArgMax`. That is enough to
/// round-trip an exported `RandomForest` back into `InferenceModel`.
mod native {
    use std::collections::HashMap;

    use onnx_export_rs::proto::{ModelProto, NodeProto, TensorProto};

    use crate::error::{Error, Result};
    use crate::frame::Frame;

    /// Does this graph use an ONNX-ML op tract cannot run?
    pub fn needs_native(proto: &ModelProto) -> bool {
        proto
            .graph
            .as_ref()
            .is_some_and(|g| g.node.iter().any(|n| n.op_type.starts_with("TreeEnsemble")))
    }

    /// A dense row-major `f32` matrix.
    #[derive(Clone)]
    struct Mat {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    }

    enum Op {
        Sub {
            a: String,
            b: String,
            out: String,
        },
        Div {
            a: String,
            b: String,
            out: String,
        },
        Tree {
            input: String,
            out: String,
            ens: TreeEnsemble,
        },
        ArgMax {
            input: String,
            out: String,
        },
        IsNaN {
            input: String,
            out: String,
        },
        Where {
            cond: String,
            a: String,
            b: String,
            out: String,
        },
    }

    pub struct NativeGraph {
        input: String,
        output: String,
        inits: HashMap<String, Mat>,
        ops: Vec<Op>,
    }

    impl NativeGraph {
        pub fn from_proto(proto: &ModelProto) -> Result<NativeGraph> {
            let g = proto
                .graph
                .as_ref()
                .ok_or_else(|| Error::Backend("native ONNX: no graph".into()))?;
            let input = g
                .input
                .first()
                .map(|v| v.name.clone())
                .ok_or_else(|| Error::Backend("native ONNX: no input".into()))?;
            let output = g
                .output
                .first()
                .map(|v| v.name.clone())
                .ok_or_else(|| Error::Backend("native ONNX: no output".into()))?;

            let inits = g
                .initializer
                .iter()
                .map(|t| (t.name.clone(), tensor_to_mat(t)))
                .collect();

            let mut ops = Vec::with_capacity(g.node.len());
            for n in &g.node {
                let out = |i: usize| n.output.get(i).cloned().unwrap_or_default();
                let inp = |i: usize| n.input.get(i).cloned().unwrap_or_default();
                match n.op_type.as_str() {
                    "Sub" => ops.push(Op::Sub {
                        a: inp(0),
                        b: inp(1),
                        out: out(0),
                    }),
                    "Div" => ops.push(Op::Div {
                        a: inp(0),
                        b: inp(1),
                        out: out(0),
                    }),
                    "TreeEnsembleRegressor" => ops.push(Op::Tree {
                        input: inp(0),
                        out: out(0),
                        ens: TreeEnsemble::from_node(n)?,
                    }),
                    "ArgMax" => ops.push(Op::ArgMax {
                        input: inp(0),
                        out: out(0),
                    }),
                    "IsNaN" => ops.push(Op::IsNaN {
                        input: inp(0),
                        out: out(0),
                    }),
                    "Where" => ops.push(Op::Where {
                        cond: inp(0),
                        a: inp(1),
                        b: inp(2),
                        out: out(0),
                    }),
                    other => {
                        return Err(Error::Backend(format!(
                            "native ONNX eval: unsupported op '{other}'"
                        )))
                    }
                }
            }
            Ok(NativeGraph {
                input,
                output,
                inits,
                ops,
            })
        }

        pub fn run(&self, frame: &Frame) -> Result<Vec<f64>> {
            let (n, p) = frame.shape();
            let mut env: HashMap<&str, Mat> = HashMap::new();
            for (k, v) in &self.inits {
                env.insert(k.as_str(), v.clone());
            }
            env.insert(
                self.input.as_str(),
                Mat {
                    rows: n,
                    cols: p,
                    data: frame.buf().iter().map(|v| *v as f32).collect(),
                },
            );

            let get = |env: &HashMap<&str, Mat>, name: &str| -> Result<Mat> {
                env.get(name)
                    .cloned()
                    .ok_or_else(|| Error::Backend(format!("native ONNX eval: missing '{name}'")))
            };

            for op in &self.ops {
                match op {
                    Op::Sub { a, b, out } => {
                        let m = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x - y);
                        env.insert(out.as_str(), m);
                    }
                    Op::Div { a, b, out } => {
                        let m = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x / y);
                        env.insert(out.as_str(), m);
                    }
                    Op::Tree { input, out, ens } => {
                        let x = get(&env, input)?;
                        let mut data = Vec::with_capacity(x.rows * ens.n_targets);
                        for r in 0..x.rows {
                            data.extend(ens.eval(&x.data[r * x.cols..(r + 1) * x.cols]));
                        }
                        env.insert(
                            out.as_str(),
                            Mat {
                                rows: x.rows,
                                cols: ens.n_targets,
                                data,
                            },
                        );
                    }
                    Op::ArgMax { input, out } => {
                        let x = get(&env, input)?;
                        let data = (0..x.rows)
                            .map(|r| {
                                let row = &x.data[r * x.cols..(r + 1) * x.cols];
                                let mut best = 0usize;
                                for c in 1..x.cols {
                                    if row[c] > row[best] {
                                        best = c;
                                    }
                                }
                                best as f32
                            })
                            .collect();
                        env.insert(
                            out.as_str(),
                            Mat {
                                rows: x.rows,
                                cols: 1,
                                data,
                            },
                        );
                    }
                    Op::IsNaN { input, out } => {
                        let x = get(&env, input)?;
                        let data = x
                            .data
                            .iter()
                            .map(|v| if v.is_nan() { 1.0 } else { 0.0 })
                            .collect();
                        env.insert(
                            out.as_str(),
                            Mat {
                                rows: x.rows,
                                cols: x.cols,
                                data,
                            },
                        );
                    }
                    Op::Where { cond, a, b, out } => {
                        // out = cond != 0 ? a : b, with `a` broadcast per column.
                        let (cond, a, b) = (get(&env, cond)?, get(&env, a)?, get(&env, b)?);
                        let cols = b.cols;
                        let mut data = Vec::with_capacity(b.rows * cols);
                        for r in 0..b.rows {
                            for c in 0..cols {
                                let av = if a.rows == 1 {
                                    a.data[c]
                                } else {
                                    a.data[r * a.cols + c]
                                };
                                data.push(if cond.data[r * cond.cols + c] != 0.0 {
                                    av
                                } else {
                                    b.data[r * cols + c]
                                });
                            }
                        }
                        env.insert(
                            out.as_str(),
                            Mat {
                                rows: b.rows,
                                cols,
                                data,
                            },
                        );
                    }
                }
            }

            let out = env
                .get(self.output.as_str())
                .ok_or_else(|| Error::Backend("native ONNX eval: output not produced".into()))?;
            Ok(out.data.iter().map(|v| *v as f64).collect())
        }
    }

    /// A parsed `TreeEnsembleRegressor` (one score per target, aggregated by sum).
    struct TreeEnsemble {
        n_targets: usize,
        trees: Vec<Tree>,
    }

    #[derive(Default, Clone)]
    struct TNode {
        is_leaf: bool,
        feature: usize,
        threshold: f32,
        true_child: usize,
        false_child: usize,
        // (target, weight) contributions if this node is a leaf
        leaf: Vec<(usize, f32)>,
    }

    #[derive(Default)]
    struct Tree {
        nodes: Vec<TNode>, // indexed by node id (sequential within the tree)
    }

    impl TreeEnsemble {
        fn from_node(n: &NodeProto) -> Result<TreeEnsemble> {
            let n_targets = int_attr(n, "n_targets").max(1) as usize;
            let tree_ids = ints_attr(n, "nodes_treeids");
            let node_ids = ints_attr(n, "nodes_nodeids");
            let feat_ids = ints_attr(n, "nodes_featureids");
            let values = floats_attr(n, "nodes_values");
            let true_ids = ints_attr(n, "nodes_truenodeids");
            let false_ids = ints_attr(n, "nodes_falsenodeids");
            let modes = strings_attr(n, "nodes_modes");

            let n_trees = tree_ids.iter().copied().max().map_or(0, |m| m as usize + 1);
            let mut trees: Vec<Tree> = (0..n_trees).map(|_| Tree::default()).collect();
            for i in 0..node_ids.len() {
                let t = tree_ids[i] as usize;
                let nid = node_ids[i] as usize;
                let tree = &mut trees[t];
                if tree.nodes.len() <= nid {
                    tree.nodes.resize(nid + 1, TNode::default());
                }
                tree.nodes[nid] = TNode {
                    is_leaf: modes.get(i).map(|m| m.as_slice()) == Some(b"LEAF"),
                    feature: feat_ids[i] as usize,
                    threshold: values[i],
                    true_child: true_ids[i] as usize,
                    false_child: false_ids[i] as usize,
                    leaf: Vec::new(),
                };
            }

            // leaf weights, keyed by (tree, node)
            let t_tree = ints_attr(n, "target_treeids");
            let t_node = ints_attr(n, "target_nodeids");
            let t_id = ints_attr(n, "target_ids");
            let t_w = floats_attr(n, "target_weights");
            for j in 0..t_id.len() {
                let (t, nid) = (t_tree[j] as usize, t_node[j] as usize);
                trees[t].nodes[nid].leaf.push((t_id[j] as usize, t_w[j]));
            }

            Ok(TreeEnsemble { n_targets, trees })
        }

        fn eval(&self, x: &[f32]) -> Vec<f32> {
            let mut scores = vec![0.0f32; self.n_targets];
            for tree in &self.trees {
                let mut nid = 0usize;
                // depth guard against a malformed graph
                for _ in 0..(tree.nodes.len() + 1) {
                    let node = &tree.nodes[nid];
                    if node.is_leaf {
                        for &(tgt, w) in &node.leaf {
                            scores[tgt] += w;
                        }
                        break;
                    }
                    // BRANCH_LEQ: x[feature] <= threshold -> true child
                    nid = if x[node.feature] <= node.threshold {
                        node.true_child
                    } else {
                        node.false_child
                    };
                }
            }
            scores
        }
    }

    // ---- proto helpers ----

    fn tensor_to_mat(t: &TensorProto) -> Mat {
        let dims: Vec<usize> = t.dims.iter().map(|d| *d as usize).collect();
        let (rows, cols) = match dims.as_slice() {
            [] => (1, 1),
            [c] => (1, *c),
            [r, c, ..] => (*r, *c),
        };
        Mat {
            rows,
            cols,
            data: read_floats(t),
        }
    }

    fn read_floats(t: &TensorProto) -> Vec<f32> {
        if !t.float_data.is_empty() {
            t.float_data.clone()
        } else {
            t.raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
    }

    fn int_attr(n: &NodeProto, name: &str) -> i64 {
        n.attribute
            .iter()
            .find(|a| a.name == name)
            .map_or(0, |a| a.i)
    }
    fn ints_attr(n: &NodeProto, name: &str) -> Vec<i64> {
        n.attribute
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    }
    fn floats_attr(n: &NodeProto, name: &str) -> Vec<f32> {
        n.attribute
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.floats.clone())
            .unwrap_or_default()
    }
    fn strings_attr(n: &NodeProto, name: &str) -> Vec<Vec<u8>> {
        n.attribute
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.strings.clone())
            .unwrap_or_default()
    }

    fn broadcast(a: &Mat, b: &Mat, f: impl Fn(f32, f32) -> f32) -> Mat {
        let cols = a.cols;
        let mut data = Vec::with_capacity(a.rows * cols);
        for r in 0..a.rows {
            for c in 0..cols {
                let bv = if b.rows == 1 {
                    b.data[c]
                } else {
                    b.data[r * b.cols + c]
                };
                data.push(f(a.data[r * cols + c], bv));
            }
        }
        Mat {
            rows: a.rows,
            cols,
            data,
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
    fn random_forest_serves_natively_through_onnx() {
        // A forest exports to ONNX-ML tree ops tract can't run — so InferenceModel
        // evaluates them natively. The round-trip must match the pipeline.
        use crate::pipeline::Pipeline;
        use crate::transform::StandardScaler;
        let (ds, probe) = two_class();
        let mut pipe = Pipeline::new()
            .step("scale", StandardScaler::new())
            .estimator("rf", RandomForest::new().n_trees(15).max_depth(4));
        pipe.fit(&ds).unwrap();
        let native = pipe.predict(&probe).unwrap();

        let path = scratch("rf_serve");
        pipe.export_onnx(&path).unwrap();
        let loaded = InferenceModel::load(&path).unwrap();
        let via_onnx = loaded.predict(&probe).unwrap();
        assert_eq!(
            native, via_onnx,
            "ONNX-served forest must match the pipeline"
        );
        let _ = std::fs::remove_file(&path);
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

    #[test]
    fn inference_model_serves_as_a_pipeline_estimator() {
        use crate::backends::smartcore::LinearRegression;
        use crate::pipeline::Pipeline;

        // Train and export an external model...
        let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
        let cols = vec!["x1".to_string(), "x2".to_string()];
        let ds = Dataset::new(Frame::from_rows(rows, cols.clone()).unwrap(), y).unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();
        let path = scratch("pipe_estimator");
        lr.export_onnx(&path).unwrap();

        // ...then load it back and drop it in as a pipeline's (frozen) estimator.
        let onnx = InferenceModel::load(&path).unwrap();
        let mut pipe = Pipeline::new().estimator("onnx", onnx);
        pipe.fit(&ds).unwrap(); // no-op fit — the model is already trained
        let probe = Frame::from_rows(vec![vec![20.0, 1.0]], cols).unwrap();

        let via_pipe = pipe.predict(&probe).unwrap();
        let direct = InferenceModel::load(&path)
            .unwrap()
            .predict(&probe)
            .unwrap();
        assert!((via_pipe[0] - direct[0]).abs() < 1e-4);
        let _ = std::fs::remove_file(&path);
    }
}
