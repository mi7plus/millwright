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
    Gather {
        input: String,
        cols: Vec<usize>,
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
                "Gather" => {
                    // indices live in the second input (an int64 initializer)
                    let cols = raw
                        .get(n.input[1].as_str())
                        .map(|t| read_i64s(t).iter().map(|v| *v as usize).collect())
                        .unwrap_or_default();
                    ops.push(Op::Gather {
                        input: inp(0),
                        cols,
                        out: out(0),
                    });
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
