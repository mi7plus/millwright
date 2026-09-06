use std::collections::HashMap;

use onnx_export_rs::proto::{ModelProto, NodeProto, TensorProto};

use crate::error::{Error, Result};
use crate::frame::Frame;

/// Re-encode every `TreeEnsembleRegressor` in the graph as plain tensor ops
/// (`Gather` / `LessOrEqual` / `MatMul` / `Equal`), which GPU execution
/// providers can run — the ONNX-ML tree op cannot run on a GPU. See
/// [`crate::onnx::ExportOnnx::to_onnx_gpu`]. Other graphs are left untouched.
///
/// Each tree becomes, per row: gather the tested feature at every internal node,
/// compare to the thresholds to get the left/right decisions `p`, multiply by a
/// node→leaf path matrix `C` and match the per-leaf left-count `D` to pick the
/// active leaf, then read that leaf's values from `V`. Trees are summed. The
/// result is numerically identical to the tree op (integer path counts are exact
/// in f32), but every op is one a GPU provider supports.
///
/// This uses one small matrix set per tree, so it stays modest for wide/shallow
/// forests but grows with tree depth; it is opt-in for exactly that reason.
pub(super) fn tensorize_tree_ensembles(proto: &mut ModelProto) -> Result<()> {
    use onnx_export_rs::graph_builder::{make_node, make_value_info, Dimension};

    let graph = proto
        .graph
        .as_mut()
        .ok_or_else(|| Error::Backend("tensorize: exported model has no graph".into()))?;

    let mut tensorized = false;
    // Rewrite each tree-ensemble node in place (there is normally one).
    while let Some(pos) = graph
        .node
        .iter()
        .position(|n| n.op_type == "TreeEnsembleRegressor")
    {
        tensorized = true;
        let node = graph.node[pos].clone();
        let input = node
            .input
            .first()
            .cloned()
            .ok_or_else(|| Error::Backend("tensorize: tree node has no input".into()))?;
        let output = node
            .output
            .first()
            .cloned()
            .ok_or_else(|| Error::Backend("tensorize: tree node has no output".into()))?;
        let ens = TreeEnsemble::from_node(&node)?;

        let mut nodes = Vec::new();
        let mut inits = Vec::new();
        let mut scores = Vec::new();
        for (ti, tree) in ens.trees.iter().enumerate() {
            scores.push(build_tree_gemm(
                tree,
                ens.n_targets,
                ti,
                &input,
                &mut nodes,
                &mut inits,
            )?);
        }
        if scores.is_empty() {
            return Err(Error::Backend(
                "tensorize: tree ensemble has no trees".into(),
            ));
        }
        // Sum the per-tree score tensors into the tree node's original output.
        if scores.len() == 1 {
            nodes.push(make_node(
                "Identity",
                [scores[0].as_str()],
                [output.as_str()],
                vec![],
            ));
        } else {
            let mut cur = scores[0].clone();
            for (k, next) in scores.iter().enumerate().skip(1) {
                let out = if k + 1 == scores.len() {
                    output.clone()
                } else {
                    format!("mw_gemm_sum{k}")
                };
                nodes.push(make_node(
                    "Add",
                    [cur.as_str(), next.as_str()],
                    [out.as_str()],
                    vec![],
                ));
                cur = out;
            }
        }
        graph.node.splice(pos..=pos, nodes);
        graph.initializer.extend(inits);
    }

    // The encoding gathers features by index, so the exact feature width no
    // longer needs to be fixed in the graph — and the tree export can
    // under-declare it (it uses only the width the splits reference, which is
    // smaller when a trailing feature is never split on). Relax the input to
    // dynamic dims so a strict runtime (tract) accepts the real batch and width.
    if tensorized {
        if let Some(vi) = graph.input.first_mut() {
            let name = vi.name.clone();
            *vi = make_value_info(
                &name,
                &[
                    Dimension::Symbolic("batch".into()),
                    Dimension::Symbolic("features".into()),
                ],
            );
        }
    }
    Ok(())
}

/// Emit the tensor-op encoding of one tree, appending its nodes/initializers and
/// returning the name of its `(rows × n_targets)` score tensor.
fn build_tree_gemm(
    tree: &Tree,
    n_targets: usize,
    ti: usize,
    input: &str,
    nodes: &mut Vec<NodeProto>,
    inits: &mut Vec<TensorProto>,
) -> Result<String> {
    use ndarray::Array2;
    use onnx_export_rs::graph_builder::{
        int_attribute, make_i64_tensor, make_node, make_tensor, FLOAT,
    };

    if tree.nodes.is_empty() {
        return Err(Error::Backend("tensorize: empty tree".into()));
    }

    // Depth-first walk: number internal nodes (columns of the decision matrix)
    // and collect, per leaf, the path taken (+1 left / -1 right at each node).
    let mut feats: Vec<i64> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    let mut leaf_paths: Vec<Vec<(usize, f32)>> = Vec::new();
    let mut leaf_weights: Vec<Vec<f32>> = Vec::new();
    let mut stack: Vec<(usize, Vec<(usize, f32)>)> = vec![(0, Vec::new())];
    while let Some((nid, path)) = stack.pop() {
        let node = &tree.nodes[nid];
        if node.is_leaf {
            let mut w = vec![0.0f32; n_targets];
            for &(tgt, weight) in &node.leaf {
                if tgt < n_targets {
                    w[tgt] += weight;
                }
            }
            leaf_weights.push(w);
            leaf_paths.push(path);
        } else {
            let col = feats.len();
            feats.push(node.feature as i64);
            thresholds.push(node.threshold);
            let mut left = path.clone();
            left.push((col, 1.0));
            stack.push((node.true_child, left));
            let mut right = path;
            right.push((col, -1.0));
            stack.push((node.false_child, right));
        }
    }

    let li = feats.len();
    let ll = leaf_weights.len();
    let t = n_targets;
    let score = format!("mw_gemm_t{ti}_score");

    let arr = |rows: usize, cols: usize, data: Vec<f32>| -> Result<_> {
        Array2::from_shape_vec((rows, cols), data)
            .map(|a| a.into_dyn())
            .map_err(|e| Error::Backend(format!("tensorize: bad matrix shape: {e}")))
    };

    // A leaf-only tree contributes a constant score row (broadcast over the batch).
    if li == 0 {
        inits.push(make_tensor(&score, &arr(1, t, leaf_weights[0].clone())?));
        return Ok(score);
    }

    // C: internal-node → leaf path matrix (I × L). D: per-leaf left-count (1 × L).
    let mut c = vec![0.0f32; li * ll];
    let mut d = vec![0.0f32; ll];
    for (leaf, path) in leaf_paths.iter().enumerate() {
        for &(col, dir) in path {
            c[col * ll + leaf] = dir;
            if dir > 0.0 {
                d[leaf] += 1.0;
            }
        }
    }
    // V: per-leaf target weights (L × T).
    let mut v = vec![0.0f32; ll * t];
    for (leaf, w) in leaf_weights.iter().enumerate() {
        v[leaf * t..leaf * t + t].copy_from_slice(w);
    }

    let feat_name = format!("mw_gemm_t{ti}_feat");
    let thr_name = format!("mw_gemm_t{ti}_thr");
    let c_name = format!("mw_gemm_t{ti}_c");
    let d_name = format!("mw_gemm_t{ti}_d");
    let v_name = format!("mw_gemm_t{ti}_v");
    inits.push(make_i64_tensor(&feat_name, &[li], feats));
    inits.push(make_tensor(&thr_name, &arr(1, li, thresholds)?));
    inits.push(make_tensor(&c_name, &arr(li, ll, c)?));
    inits.push(make_tensor(&d_name, &arr(1, ll, d)?));
    inits.push(make_tensor(&v_name, &arr(ll, t, v)?));

    let xa = format!("mw_gemm_t{ti}_xa");
    let pb = format!("mw_gemm_t{ti}_pb");
    let p = format!("mw_gemm_t{ti}_p");
    let s = format!("mw_gemm_t{ti}_s");
    let eb = format!("mw_gemm_t{ti}_eb");
    let e = format!("mw_gemm_t{ti}_e");
    // gather x[feature] at each node -> (rows × I)
    nodes.push(make_node(
        "Gather",
        [input, feat_name.as_str()],
        [xa.as_str()],
        vec![int_attribute("axis", 1)],
    ));
    // BRANCH_LEQ: left taken when x <= threshold
    nodes.push(make_node(
        "LessOrEqual",
        [xa.as_str(), thr_name.as_str()],
        [pb.as_str()],
        vec![],
    ));
    nodes.push(make_node(
        "Cast",
        [pb.as_str()],
        [p.as_str()],
        vec![int_attribute("to", FLOAT as i64)],
    ));
    nodes.push(make_node(
        "MatMul",
        [p.as_str(), c_name.as_str()],
        [s.as_str()],
        vec![],
    ));
    // the active leaf is the one whose left-count matches exactly
    nodes.push(make_node(
        "Equal",
        [s.as_str(), d_name.as_str()],
        [eb.as_str()],
        vec![],
    ));
    nodes.push(make_node(
        "Cast",
        [eb.as_str()],
        [e.as_str()],
        vec![int_attribute("to", FLOAT as i64)],
    ));
    nodes.push(make_node(
        "MatMul",
        [e.as_str(), v_name.as_str()],
        [score.as_str()],
        vec![],
    ));
    Ok(score)
}

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
    Add {
        a: String,
        b: String,
        out: String,
    },
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
    Mul {
        a: String,
        b: String,
        out: String,
    },
    MatMul {
        a: String,
        b: String,
        out: String,
    },
    Sigmoid {
        input: String,
        out: String,
    },
    GreaterOrEqual {
        input: String,
        value: f32,
        out: String,
    },
    Identity {
        input: String,
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
    Gather {
        input: String,
        cols: Vec<usize>,
        out: String,
    },
    // Gather with a constant data table and dynamic indices (e.g. class-label
    // lookup after ArgMax): out[r] = table[indices[r]].
    GatherData {
        table: Vec<f32>,
        indices: String,
        out: String,
    },
    Round {
        input: String,
        out: String,
    },
    Equal {
        input: String,
        value: f32,
        out: String,
    },
    // Cast is a no-op here: every tensor is already f32.
    Cast {
        input: String,
        out: String,
    },
    Concat {
        inputs: Vec<String>,
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

        let inits: HashMap<String, Mat> = g
            .initializer
            .iter()
            .map(|t| (t.name.clone(), tensor_to_mat(t)))
            .collect();
        // Raw initializers, to bake Gather indices / Equal constants at parse
        // time (so the interpreter's runtime tensors stay uniformly f32).
        let raw: HashMap<&str, &TensorProto> =
            g.initializer.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut ops = Vec::with_capacity(g.node.len());
        for n in &g.node {
            let out = |i: usize| n.output.get(i).cloned().unwrap_or_default();
            let inp = |i: usize| n.input.get(i).cloned().unwrap_or_default();
            match n.op_type.as_str() {
                "Add" => ops.push(Op::Add {
                    a: inp(0),
                    b: inp(1),
                    out: out(0),
                }),
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
                "Mul" => ops.push(Op::Mul {
                    a: inp(0),
                    b: inp(1),
                    out: out(0),
                }),
                "MatMul" => ops.push(Op::MatMul {
                    a: inp(0),
                    b: inp(1),
                    out: out(0),
                }),
                "Sigmoid" => ops.push(Op::Sigmoid {
                    input: inp(0),
                    out: out(0),
                }),
                "GreaterOrEqual" => {
                    let value = raw
                        .get(n.input[1].as_str())
                        .and_then(|tensor| read_floats(tensor).first().copied())
                        .unwrap_or(f32::NAN);
                    ops.push(Op::GreaterOrEqual {
                        input: inp(0),
                        value,
                        out: out(0),
                    });
                }
                "Identity" => ops.push(Op::Identity {
                    input: inp(0),
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
                "Gather" => {
                    if let Some(t) = raw.get(n.input[1].as_str()) {
                        // constant indices -> select those columns (one-hot)
                        let cols = read_i64s(t).iter().map(|v| *v as usize).collect();
                        ops.push(Op::Gather {
                            input: inp(0),
                            cols,
                            out: out(0),
                        });
                    } else if let Some(t) = raw.get(n.input[0].as_str()) {
                        // constant data table, dynamic indices -> table lookup
                        // (class-label mapping after ArgMax)
                        let table = read_i64s(t).iter().map(|v| *v as f32).collect();
                        ops.push(Op::GatherData {
                            table,
                            indices: inp(1),
                            out: out(0),
                        });
                    } else {
                        return Err(Error::Backend(
                            "native ONNX eval: Gather needs a constant operand".into(),
                        ));
                    }
                }
                "Round" => ops.push(Op::Round {
                    input: inp(0),
                    out: out(0),
                }),
                "Equal" => {
                    // the compared constant lives in the second input
                    let value = raw
                        .get(n.input[1].as_str())
                        .and_then(|t| read_floats(t).first().copied())
                        .unwrap_or(f32::NAN);
                    ops.push(Op::Equal {
                        input: inp(0),
                        value,
                        out: out(0),
                    });
                }
                "Cast" => ops.push(Op::Cast {
                    input: inp(0),
                    out: out(0),
                }),
                "Concat" => ops.push(Op::Concat {
                    inputs: n.input.clone(),
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
                Op::Add { a, b, out } => {
                    let value = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x + y);
                    env.insert(out.as_str(), value);
                }
                Op::Sub { a, b, out } => {
                    let m = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x - y);
                    env.insert(out.as_str(), m);
                }
                Op::Div { a, b, out } => {
                    let m = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x / y);
                    env.insert(out.as_str(), m);
                }
                Op::Mul { a, b, out } => {
                    let value = broadcast(&get(&env, a)?, &get(&env, b)?, |x, y| x * y);
                    env.insert(out.as_str(), value);
                }
                Op::MatMul { a, b, out } => {
                    let left = get(&env, a)?;
                    let right = get(&env, b)?;
                    if left.cols != right.rows {
                        return Err(Error::Shape(format!(
                            "native ONNX MatMul mismatch: {}x{} by {}x{}",
                            left.rows, left.cols, right.rows, right.cols
                        )));
                    }
                    let mut data = vec![0.0; left.rows * right.cols];
                    for row in 0..left.rows {
                        for col in 0..right.cols {
                            for inner in 0..left.cols {
                                data[row * right.cols + col] += left.data[row * left.cols + inner]
                                    * right.data[inner * right.cols + col];
                            }
                        }
                    }
                    env.insert(
                        out.as_str(),
                        Mat {
                            rows: left.rows,
                            cols: right.cols,
                            data,
                        },
                    );
                }
                Op::Sigmoid { input, out } => {
                    let mut value = get(&env, input)?;
                    for item in &mut value.data {
                        *item = 1.0 / (1.0 + (-*item).exp());
                    }
                    env.insert(out.as_str(), value);
                }
                Op::GreaterOrEqual { input, value, out } => {
                    let mut matrix = get(&env, input)?;
                    for item in &mut matrix.data {
                        *item = (*item >= *value) as u8 as f32;
                    }
                    env.insert(out.as_str(), matrix);
                }
                Op::Identity { input, out } => {
                    env.insert(out.as_str(), get(&env, input)?);
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
                Op::Gather { input, cols, out } => {
                    // select `cols` columns (axis 1) from the input
                    let x = get(&env, input)?;
                    let mut data = Vec::with_capacity(x.rows * cols.len());
                    for r in 0..x.rows {
                        for &c in cols {
                            data.push(x.data[r * x.cols + c]);
                        }
                    }
                    env.insert(
                        out.as_str(),
                        Mat {
                            rows: x.rows,
                            cols: cols.len(),
                            data,
                        },
                    );
                }
                Op::GatherData {
                    table,
                    indices,
                    out,
                } => {
                    // out[r] = table[indices[r]] (class-label lookup)
                    let idx = get(&env, indices)?;
                    let data = idx
                        .data
                        .iter()
                        .map(|v| table.get(*v as usize).copied().unwrap_or(f32::NAN))
                        .collect();
                    env.insert(
                        out.as_str(),
                        Mat {
                            rows: idx.rows,
                            cols: idx.cols,
                            data,
                        },
                    );
                }
                Op::Round { input, out } => {
                    let x = get(&env, input)?;
                    let data = x.data.iter().map(|v| v.round()).collect();
                    env.insert(
                        out.as_str(),
                        Mat {
                            rows: x.rows,
                            cols: x.cols,
                            data,
                        },
                    );
                }
                Op::Equal { input, value, out } => {
                    let x = get(&env, input)?;
                    let data = x
                        .data
                        .iter()
                        .map(|v| if v == value { 1.0 } else { 0.0 })
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
                Op::Cast { input, out } => {
                    let x = get(&env, input)?;
                    env.insert(out.as_str(), x);
                }
                Op::Concat { inputs, out } => {
                    // horizontally stack the pieces (all share row count)
                    let pieces: Vec<Mat> = inputs
                        .iter()
                        .map(|nm| get(&env, nm))
                        .collect::<Result<_>>()?;
                    let rows = pieces.first().map_or(0, |m| m.rows);
                    let cols: usize = pieces.iter().map(|m| m.cols).sum();
                    let mut data = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        for m in &pieces {
                            data.extend_from_slice(&m.data[r * m.cols..(r + 1) * m.cols]);
                        }
                    }
                    env.insert(out.as_str(), Mat { rows, cols, data });
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

fn read_i64s(t: &TensorProto) -> Vec<i64> {
    if !t.int64_data.is_empty() {
        t.int64_data.clone()
    } else {
        t.raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
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
