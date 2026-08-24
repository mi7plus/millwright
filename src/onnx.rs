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
//! A pipeline is exported by splicing each leading transformer in front of the
//! estimator's graph, in order — scalers as an affine `(x - shift) / scale`,
//! imputers as `Where(IsNaN(x), fill, x)`, one-hot encoders as
//! `Concat(Cast(Equal(Round(Gather(x)), cat)))` — so the result is one
//! self-contained graph: raw features in, predictions out. A step with no ONNX
//! form is reported as an error naming it.

use std::path::Path;

use ndarray::Array2;
use onnx_export_rs::graph_builder::{
    int_attribute, make_i64_tensor, make_node, make_tensor, make_value_info, save_to_file,
    Dimension, FLOAT,
};
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
    /// One-hot encode: `columns[c]` is the categories input column `c` expands to
    /// (an empty list passes the column through). Changes the feature width.
    OneHot { columns: Vec<Vec<i64>> },
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
            Prefix::OneHot { columns } => {
                // For each input column: Gather it out, and either pass it
                // through or expand it to Cast(Equal(Round(col), cat)) indicators.
                // Concat every piece (in order) into the wider encoded tensor.
                let mut pieces: Vec<String> = Vec::new();
                for (c, cats) in columns.iter().enumerate() {
                    let idx_name = format!("mw_idx{i}_{c}");
                    inits.push(make_i64_tensor(&idx_name, &[1], vec![c as i64]));
                    let col = format!("mw_col{i}_{c}");
                    nodes.push(make_node(
                        "Gather",
                        [cur.as_str(), idx_name.as_str()],
                        [col.as_str()],
                        vec![int_attribute("axis", 1)],
                    ));
                    if cats.is_empty() {
                        pieces.push(col);
                        continue;
                    }
                    let rounded = format!("mw_round{i}_{c}");
                    nodes.push(make_node(
                        "Round",
                        [col.as_str()],
                        [rounded.as_str()],
                        Vec::new(),
                    ));
                    for (j, cat) in cats.iter().enumerate() {
                        let cat_name = format!("mw_cat{i}_{c}_{j}");
                        inits.push(row_tensor(&cat_name, &[*cat as f64])?);
                        let eq = format!("mw_eq{i}_{c}_{j}");
                        nodes.push(make_node(
                            "Equal",
                            [rounded.as_str(), cat_name.as_str()],
                            [eq.as_str()],
                            Vec::new(),
                        ));
                        let ind = format!("mw_ind{i}_{c}_{j}");
                        nodes.push(make_node(
                            "Cast",
                            [eq.as_str()],
                            [ind.as_str()],
                            vec![int_attribute("to", FLOAT as i64)],
                        ));
                        pieces.push(ind);
                    }
                }
                nodes.push(make_node(
                    "Concat",
                    pieces,
                    [out.as_str()],
                    vec![int_attribute("axis", 1)],
                ));
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

    // The graph now consumes raw features on `mw_input`. Declare it with the raw
    // feature width (the width the first prefix consumes) — which differs from
    // the estimator's input width when a prefix changes width (one-hot).
    let raw_width = match &prefixes[0] {
        Prefix::Affine { shift, .. } => shift.len(),
        Prefix::Impute { fill } => fill.len(),
        Prefix::OneHot { columns } => columns.len(),
    };
    if let Some(vi) = graph.input.first_mut() {
        *vi = make_value_info(
            "mw_input",
            &[
                Dimension::Symbolic("batch".into()),
                Dimension::Fixed(raw_width),
            ],
        );
    }
    Ok(())
}

#[cfg(feature = "ensemble")]
#[derive(Clone, Copy)]
pub(crate) enum EnsembleAggregation<'a> {
    Mean,
    HardVote {
        classes: &'a [i64],
        weights: &'a [f64],
    },
    SoftVote {
        classes: &'a [i64],
        weights: &'a [f64],
    },
}

#[cfg(feature = "ensemble")]
fn namespace_graph(
    proto: ModelProto,
    prefix: &str,
    replacement_input: &str,
) -> Result<(
    Vec<onnx_export_rs::proto::NodeProto>,
    Vec<onnx_export_rs::proto::TensorProto>,
    String,
)> {
    let graph = proto
        .graph
        .ok_or_else(|| Error::Backend("ensemble member ONNX model has no graph".into()))?;
    let input = graph
        .input
        .first()
        .map(|value| value.name.clone())
        .ok_or_else(|| Error::Backend("ensemble member ONNX model has no input".into()))?;
    let output = graph
        .output
        .first()
        .map(|value| value.name.clone())
        .ok_or_else(|| Error::Backend("ensemble member ONNX model has no output".into()))?;
    let rename = |name: &str| {
        if name == input {
            replacement_input.to_string()
        } else {
            format!("{prefix}{name}")
        }
    };
    let mut nodes = graph.node;
    for node in &mut nodes {
        node.input = node.input.iter().map(|name| rename(name)).collect();
        node.output = node.output.iter().map(|name| rename(name)).collect();
        if !node.name.is_empty() {
            node.name = format!("{prefix}{}", node.name);
        }
    }
    let mut initializers = graph.initializer;
    for initializer in &mut initializers {
        initializer.name = rename(&initializer.name);
    }
    Ok((nodes, initializers, rename(&output)))
}

#[cfg(feature = "ensemble")]
pub(crate) fn combine_onnx(
    protos: Vec<ModelProto>,
    aggregation: EnsembleAggregation<'_>,
) -> Result<ModelProto> {
    use ndarray::Array1;
    use onnx_export_rs::graph_builder::{assemble_model, make_node};
    use onnx_export_rs::proto::GraphProto;

    if protos.is_empty() {
        return Err(Error::Backend("cannot export an empty ensemble".into()));
    }
    let input_info = protos[0]
        .graph
        .as_ref()
        .and_then(|graph| graph.input.first())
        .cloned()
        .ok_or_else(|| Error::Backend("ensemble member ONNX model has no input".into()))?;
    let opset = protos
        .iter()
        .flat_map(|proto| proto.opset_import.iter())
        .filter(|opset| opset.domain.is_empty())
        .map(|opset| opset.version)
        .max()
        .unwrap_or(13)
        .max(13);
    let ir = protos
        .iter()
        .map(|proto| proto.ir_version)
        .max()
        .unwrap_or(8);
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut outputs = Vec::new();
    for (index, proto) in protos.into_iter().enumerate() {
        let (mut member_nodes, mut member_initializers, output) =
            namespace_graph(proto, &format!("mw_m{index}_"), "mw_input")?;
        nodes.append(&mut member_nodes);
        initializers.append(&mut member_initializers);
        outputs.push(output);
    }

    let scalar =
        |name: &str, value: f64| make_tensor(name, &Array1::from(vec![value as f32]).into_dyn());
    fn add_chain(
        nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
        terms: Vec<String>,
        stem: &str,
    ) -> Result<String> {
        let mut iter = terms.into_iter();
        let mut current = iter
            .next()
            .ok_or_else(|| Error::Backend("ensemble aggregation has no terms".into()))?;
        for (index, term) in iter.enumerate() {
            let output = format!("mw_{stem}_sum{index}");
            nodes.push(make_node(
                "Add",
                [current.as_str(), term.as_str()],
                [output.as_str()],
                vec![],
            ));
            current = output;
        }
        Ok(current)
    }

    let final_output = match aggregation {
        EnsembleAggregation::Mean => {
            let count = outputs.len();
            let sum = add_chain(&mut nodes, outputs, "mean")?;
            if count == 1 {
                sum
            } else {
                initializers.push(scalar("mw_divisor", count as f64));
                nodes.push(make_node(
                    "Div",
                    [sum.as_str(), "mw_divisor"],
                    ["mw_output"],
                    vec![],
                ));
                "mw_output".into()
            }
        }
        EnsembleAggregation::HardVote { classes, weights } => {
            if outputs.len() != weights.len() || classes.is_empty() {
                return Err(Error::Backend(
                    "invalid hard-voting ONNX aggregation".into(),
                ));
            }
            let mut class_scores = Vec::new();
            for (class_index, class) in classes.iter().enumerate() {
                let class_name = format!("mw_class_{class_index}");
                initializers.push(scalar(&class_name, *class as f64));
                let mut terms = Vec::new();
                for (member_index, output) in outputs.iter().enumerate() {
                    let equal = format!("mw_eq_{class_index}_{member_index}");
                    let cast = format!("mw_cast_{class_index}_{member_index}");
                    let weighted = format!("mw_weighted_{class_index}_{member_index}");
                    let weight = format!("mw_weight_{member_index}");
                    if class_index == 0 {
                        initializers.push(scalar(&weight, weights[member_index]));
                    }
                    nodes.push(make_node(
                        "Equal",
                        [output.as_str(), class_name.as_str()],
                        [equal.as_str()],
                        vec![],
                    ));
                    nodes.push(make_node(
                        "Cast",
                        [equal.as_str()],
                        [cast.as_str()],
                        vec![int_attribute("to", FLOAT as i64)],
                    ));
                    nodes.push(make_node(
                        "Mul",
                        [cast.as_str(), weight.as_str()],
                        [weighted.as_str()],
                        vec![],
                    ));
                    terms.push(weighted);
                }
                class_scores.push(add_chain(
                    &mut nodes,
                    terms,
                    &format!("class{class_index}"),
                )?);
            }
            nodes.push(make_node(
                "Concat",
                class_scores,
                ["mw_scores"],
                vec![int_attribute("axis", 1)],
            ));
            nodes.push(make_node(
                "ArgMax",
                ["mw_scores"],
                ["mw_index"],
                vec![int_attribute("axis", 1), int_attribute("keepdims", 1)],
            ));
            nodes.push(make_node(
                "Cast",
                ["mw_index"],
                ["mw_index_f"],
                vec![int_attribute("to", FLOAT as i64)],
            ));
            map_class_index(&mut nodes, &mut initializers, classes, "mw_index_f")
        }
        EnsembleAggregation::SoftVote { classes, weights } => {
            if outputs.len() != weights.len() || classes.is_empty() {
                return Err(Error::Backend(
                    "invalid soft-voting ONNX aggregation".into(),
                ));
            }
            let mut terms = Vec::new();
            for (index, output) in outputs.iter().enumerate() {
                let weight = format!("mw_weight_{index}");
                let weighted = format!("mw_weighted_{index}");
                initializers.push(scalar(&weight, weights[index]));
                nodes.push(make_node(
                    "Mul",
                    [output.as_str(), weight.as_str()],
                    [weighted.as_str()],
                    vec![],
                ));
                terms.push(weighted);
            }
            let scores = add_chain(&mut nodes, terms, "soft")?;
            nodes.push(make_node(
                "ArgMax",
                [scores.as_str()],
                ["mw_index"],
                vec![int_attribute("axis", 1), int_attribute("keepdims", 1)],
            ));
            nodes.push(make_node(
                "Cast",
                ["mw_index"],
                ["mw_index_f"],
                vec![int_attribute("to", FLOAT as i64)],
            ));
            map_class_index(&mut nodes, &mut initializers, classes, "mw_index_f")
        }
    };
    let mut input = input_info;
    input.name = "mw_input".into();
    let output_info = make_value_info(
        final_output.clone(),
        &[Dimension::Symbolic("batch".into()), Dimension::Fixed(1)],
    );
    Ok(assemble_model(
        GraphProto {
            node: nodes,
            name: "millwright_ensemble".into(),
            initializer: initializers,
            doc_string: String::new(),
            input: vec![input],
            output: vec![output_info],
            value_info: vec![],
        },
        opset,
        ir,
    ))
}

#[cfg(feature = "ensemble")]
fn map_class_index(
    nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
    initializers: &mut Vec<onnx_export_rs::proto::TensorProto>,
    classes: &[i64],
    index: &str,
) -> String {
    use ndarray::Array1;
    let mut terms = Vec::new();
    for (position, class) in classes.iter().enumerate() {
        let position_name = format!("mw_position_{position}");
        let class_name = format!("mw_label_{position}");
        let equal = format!("mw_index_eq_{position}");
        let cast = format!("mw_index_cast_{position}");
        let term = format!("mw_label_term_{position}");
        initializers.push(make_tensor(
            &position_name,
            &Array1::from(vec![position as f32]).into_dyn(),
        ));
        initializers.push(make_tensor(
            &class_name,
            &Array1::from(vec![*class as f32]).into_dyn(),
        ));
        nodes.push(make_node(
            "Equal",
            [index, position_name.as_str()],
            [equal.as_str()],
            vec![],
        ));
        nodes.push(make_node(
            "Cast",
            [equal.as_str()],
            [cast.as_str()],
            vec![int_attribute("to", FLOAT as i64)],
        ));
        nodes.push(make_node(
            "Mul",
            [cast.as_str(), class_name.as_str()],
            [term.as_str()],
            vec![],
        ));
        terms.push(term);
    }
    let mut current = terms[0].clone();
    for (i, term) in terms.iter().skip(1).enumerate() {
        let output = if i + 2 == terms.len() {
            "mw_output".into()
        } else {
            format!("mw_label_sum{i}")
        };
        nodes.push(make_node(
            "Add",
            [current.as_str(), term.as_str()],
            [output.as_str()],
            vec![],
        ));
        current = output;
    }
    if terms.len() == 1 {
        nodes.push(make_node(
            "Identity",
            [current.as_str()],
            ["mw_output"],
            vec![],
        ));
        "mw_output".into()
    } else {
        current
    }
}

#[cfg(feature = "ensemble")]
pub(crate) fn stack_onnx(bases: Vec<ModelProto>, meta: ModelProto) -> Result<ModelProto> {
    use onnx_export_rs::graph_builder::{assemble_model, make_node};
    use onnx_export_rs::proto::GraphProto;

    if bases.is_empty() {
        return Err(Error::Backend(
            "cannot export stacking without base models".into(),
        ));
    }
    let mut input = bases[0]
        .graph
        .as_ref()
        .and_then(|graph| graph.input.first())
        .cloned()
        .ok_or_else(|| Error::Backend("stacking base ONNX model has no input".into()))?;
    input.name = "mw_input".into();
    let opset = bases
        .iter()
        .chain(std::iter::once(&meta))
        .flat_map(|proto| proto.opset_import.iter())
        .filter(|opset| opset.domain.is_empty())
        .map(|opset| opset.version)
        .max()
        .unwrap_or(13)
        .max(13);
    let ir = bases
        .iter()
        .chain(std::iter::once(&meta))
        .map(|proto| proto.ir_version)
        .max()
        .unwrap_or(8);
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut outputs = Vec::new();
    for (index, proto) in bases.into_iter().enumerate() {
        let (mut member_nodes, mut member_initializers, output) =
            namespace_graph(proto, &format!("mw_b{index}_"), "mw_input")?;
        nodes.append(&mut member_nodes);
        initializers.append(&mut member_initializers);
        outputs.push(output);
    }
    nodes.push(make_node(
        "Concat",
        outputs,
        ["mw_meta_input"],
        vec![int_attribute("axis", 1)],
    ));
    let (mut meta_nodes, mut meta_initializers, meta_output) =
        namespace_graph(meta, "mw_meta_", "mw_meta_input")?;
    nodes.append(&mut meta_nodes);
    initializers.append(&mut meta_initializers);
    Ok(assemble_model(
        GraphProto {
            node: nodes,
            name: "millwright_stacking".into(),
            initializer: initializers,
            doc_string: String::new(),
            input: vec![input],
            output: vec![make_value_info(
                meta_output,
                &[Dimension::Symbolic("batch".into()), Dimension::Fixed(1)],
            )],
            value_info: vec![],
        },
        opset,
        ir,
    ))
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
#[path = "onnx_native.rs"]
mod native;

#[cfg(all(test, feature = "smartcore-backend"))]
#[path = "onnx_tests.rs"]
mod tests;
