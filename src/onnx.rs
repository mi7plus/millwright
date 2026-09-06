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
//! With the `gpu-inference` feature, [`InferenceModel::load_on`] runs a model
//! through onnxruntime instead, so it can use a GPU: [`Device::Auto`] picks the
//! available execution provider (DirectML on Windows, CoreML on macOS, or CUDA
//! with the `gpu-cuda` feature) and falls back to CPU, so it runs on any system.
//! GPU acceleration helps linear / large-batch graphs; tree-ensemble ops run on
//! the CPU provider regardless.
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

    /// Build the ONNX graph, but re-encode any tree-ensemble op as plain tensor
    /// operations (`Gather` / `LessOrEqual` / `MatMul` / …) so it can run on a
    /// GPU execution provider via [`InferenceModel::load_on`] — onnxruntime has
    /// no GPU kernel for the ONNX-ML tree op, so a normally-exported forest would
    /// fall back to CPU. Non-tree graphs (e.g. linear models) are returned
    /// unchanged. Predictions are identical to [`to_onnx`](Self::to_onnx).
    ///
    /// The encoding uses a small matrix set per tree, so it is well-suited to
    /// wide / shallow forests and large batches; the graph grows with tree
    /// depth, which is why it is a separate, opt-in export.
    fn to_onnx_gpu(&self) -> Result<ModelProto> {
        let mut proto = self.to_onnx()?;
        native::tensorize_tree_ensembles(&mut proto)?;
        Ok(proto)
    }

    /// Write the GPU-friendly ONNX graph (see [`to_onnx_gpu`](Self::to_onnx_gpu))
    /// to `path`.
    fn export_onnx_gpu(&self, path: impl AsRef<Path>) -> Result<()> {
        let proto = self.to_onnx_gpu()?;
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
fn merged_opset_imports<'a>(
    protos: impl IntoIterator<Item = &'a ModelProto>,
    minimum_default: i64,
) -> Vec<onnx_export_rs::proto::OperatorSetIdProto> {
    use std::collections::BTreeMap;

    let mut versions = BTreeMap::<String, i64>::new();
    versions.insert(String::new(), minimum_default);
    for proto in protos {
        for import in &proto.opset_import {
            versions
                .entry(import.domain.clone())
                .and_modify(|version| *version = (*version).max(import.version))
                .or_insert(import.version);
        }
    }
    versions
        .into_iter()
        .map(|(domain, version)| onnx_export_rs::proto::OperatorSetIdProto { domain, version })
        .collect()
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
    use onnx_export_rs::graph_builder::assemble_model;
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
    let opset_imports = merged_opset_imports(&protos, 13);
    let opset = opset_imports
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map_or(13, |opset| opset.version);
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

    let final_output = aggregate_ensemble(&mut nodes, &mut initializers, outputs, aggregation)?;
    let mut input = input_info;
    input.name = "mw_input".into();
    let output_info = make_value_info(
        final_output.clone(),
        &[Dimension::Symbolic("batch".into()), Dimension::Fixed(1)],
    );
    let mut model = assemble_model(
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
    );
    model.opset_import = opset_imports;
    Ok(model)
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
    let opset_imports = merged_opset_imports(bases.iter().chain(std::iter::once(&meta)), 13);
    let opset = opset_imports
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map_or(13, |opset| opset.version);
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
    let mut model = assemble_model(
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
    );
    model.opset_import = opset_imports;
    Ok(model)
}

#[cfg(feature = "ensemble")]
fn scalar_initializer(name: &str, value: f64) -> onnx_export_rs::proto::TensorProto {
    use ndarray::Array1;
    make_tensor(name, &Array1::from(vec![value as f32]).into_dyn())
}

#[cfg(feature = "ensemble")]
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

#[cfg(feature = "ensemble")]
fn aggregate_ensemble(
    nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
    initializers: &mut Vec<onnx_export_rs::proto::TensorProto>,
    outputs: Vec<String>,
    aggregation: EnsembleAggregation<'_>,
) -> Result<String> {
    match aggregation {
        EnsembleAggregation::Mean => aggregate_mean(nodes, initializers, outputs),
        EnsembleAggregation::HardVote { classes, weights } => {
            aggregate_hard_vote(nodes, initializers, &outputs, classes, weights)
        }
        EnsembleAggregation::SoftVote { classes, weights } => {
            aggregate_soft_vote(nodes, initializers, &outputs, classes, weights)
        }
    }
}

#[cfg(feature = "ensemble")]
fn aggregate_mean(
    nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
    initializers: &mut Vec<onnx_export_rs::proto::TensorProto>,
    outputs: Vec<String>,
) -> Result<String> {
    let count = outputs.len();
    let sum = add_chain(nodes, outputs, "mean")?;
    if count == 1 {
        return Ok(sum);
    }
    initializers.push(scalar_initializer("mw_divisor", count as f64));
    nodes.push(make_node(
        "Div",
        [sum.as_str(), "mw_divisor"],
        ["mw_output"],
        vec![],
    ));
    Ok("mw_output".into())
}

#[cfg(feature = "ensemble")]
fn aggregate_hard_vote(
    nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
    initializers: &mut Vec<onnx_export_rs::proto::TensorProto>,
    outputs: &[String],
    classes: &[i64],
    weights: &[f64],
) -> Result<String> {
    if outputs.len() != weights.len() || classes.is_empty() {
        return Err(Error::Backend(
            "invalid hard-voting ONNX aggregation".into(),
        ));
    }
    let mut class_scores = Vec::new();
    for (class_index, class) in classes.iter().enumerate() {
        let class_name = format!("mw_class_{class_index}");
        initializers.push(scalar_initializer(&class_name, *class as f64));
        let mut terms = Vec::new();
        for (member_index, output) in outputs.iter().enumerate() {
            let equal = format!("mw_eq_{class_index}_{member_index}");
            let cast = format!("mw_cast_{class_index}_{member_index}");
            let weighted = format!("mw_weighted_{class_index}_{member_index}");
            let weight = format!("mw_weight_{member_index}");
            if class_index == 0 {
                initializers.push(scalar_initializer(&weight, weights[member_index]));
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
        class_scores.push(add_chain(nodes, terms, &format!("class{class_index}"))?);
    }
    nodes.push(make_node(
        "Concat",
        class_scores,
        ["mw_scores"],
        vec![int_attribute("axis", 1)],
    ));
    append_argmax(nodes, "mw_scores");
    Ok(map_class_index(nodes, initializers, classes, "mw_index_f"))
}

#[cfg(feature = "ensemble")]
fn aggregate_soft_vote(
    nodes: &mut Vec<onnx_export_rs::proto::NodeProto>,
    initializers: &mut Vec<onnx_export_rs::proto::TensorProto>,
    outputs: &[String],
    classes: &[i64],
    weights: &[f64],
) -> Result<String> {
    if outputs.len() != weights.len() || classes.is_empty() {
        return Err(Error::Backend(
            "invalid soft-voting ONNX aggregation".into(),
        ));
    }
    let mut terms = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        let weight = format!("mw_weight_{index}");
        let weighted = format!("mw_weighted_{index}");
        initializers.push(scalar_initializer(&weight, weights[index]));
        nodes.push(make_node(
            "Mul",
            [output.as_str(), weight.as_str()],
            [weighted.as_str()],
            vec![],
        ));
        terms.push(weighted);
    }
    let scores = add_chain(nodes, terms, "soft")?;
    append_argmax(nodes, &scores);
    Ok(map_class_index(nodes, initializers, classes, "mw_index_f"))
}

#[cfg(feature = "ensemble")]
fn append_argmax(nodes: &mut Vec<onnx_export_rs::proto::NodeProto>, scores: &str) {
    nodes.push(make_node(
        "ArgMax",
        [scores],
        ["mw_index"],
        vec![int_attribute("axis", 1), int_attribute("keepdims", 1)],
    ));
    nodes.push(make_node(
        "Cast",
        ["mw_index"],
        ["mw_index_f"],
        vec![int_attribute("to", FLOAT as i64)],
    ));
}

/// Append a class-label lookup to a classifier graph whose output is a 0-based
/// class *index* (e.g. the `ArgMax` ending a tree-ensemble export). Splices a
/// `Gather(labels, index)` so the graph emits the original class labels instead
/// of `0..k`. A no-op mapping (`labels == 0..k`) is skipped.
pub(crate) fn append_label_map(proto: &mut ModelProto, labels: &[i64]) -> Result<()> {
    // identity mapping needs no gather
    if labels.iter().copied().eq(0..labels.len() as i64) {
        return Ok(());
    }
    let graph = proto
        .graph
        .as_mut()
        .ok_or_else(|| Error::Backend("exported model has no graph".into()))?;
    let index_out = graph
        .output
        .first()
        .map(|o| o.name.clone())
        .ok_or_else(|| Error::Backend("exported model has no output".into()))?;

    graph.initializer.push(make_i64_tensor(
        "mw_labels",
        &[labels.len()],
        labels.to_vec(),
    ));
    // Gather(data = labels, indices = class index, axis = 0) -> class label
    graph.node.push(make_node(
        "Gather",
        ["mw_labels", index_out.as_str()],
        ["mw_label"],
        vec![int_attribute("axis", 0)],
    ));
    if let Some(o) = graph.output.first_mut() {
        o.name = "mw_label".into();
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
    /// onnxruntime session (GPU or CPU execution providers). `run` needs
    /// `&mut`, so the session is held behind a `Mutex` to keep `predict(&self)`.
    #[cfg(feature = "gpu-inference")]
    Ort(std::sync::Mutex<ort::session::Session>),
    /// One onnxruntime session per GPU (data-parallel): a batch is split across
    /// them and run concurrently. See [`InferenceModel::load_multi`].
    #[cfg(feature = "gpu-inference")]
    OrtPool(Vec<std::sync::Mutex<ort::session::Session>>),
}

/// Which device an [`InferenceModel`] should run on when loaded via
/// [`InferenceModel::load_on`] (requires the `gpu-inference` feature).
///
/// The model runs through onnxruntime's execution providers. Which one actually
/// carries a given op is decided *per node* by onnxruntime, not by the relative
/// power of the CPU and GPU: a GPU-supported op (a `Gemm` / matmul, most
/// elementwise preprocessing) runs on the GPU when a GPU provider is active, and
/// on a strong-GPU / weak-CPU machine that pays off even at small batch sizes.
///
/// Note: onnxruntime has **no GPU kernel** for the ONNX-ML tree-ensemble ops (an
/// exported `RandomForest`), so those run on the CPU provider no matter how
/// powerful the GPU is — a coverage gap, not a speed tradeoff. Accelerating a
/// forest on the GPU needs a different engine, not a different [`Device`].
#[cfg(feature = "gpu-inference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// Try the available GPU provider first — DirectML on Windows, CoreML on
    /// macOS (both automatic), or CUDA if built with the `gpu-cuda` feature —
    /// and **silently fall back to CPU** if none initializes. The portable
    /// default: always loads, on any system.
    #[default]
    Auto,
    /// Require a GPU provider: like [`Auto`](Self::Auto), but **error instead of
    /// falling back to CPU** if no GPU provider can be initialized. Use this on a
    /// machine whose GPU should carry the work, so a missing / broken GPU is
    /// reported loudly rather than silently degrading to slow CPU inference.
    /// (Ops with no GPU kernel — tree ensembles — still run on CPU within the
    /// session; this guarantees a GPU provider is *active*, not that every node
    /// runs on it.) Errors at load time if the build has no GPU provider compiled
    /// in (a non-macOS, non-Windows build without the `gpu-cuda` feature).
    Gpu,
    /// Force the onnxruntime CPU provider (a baseline for benchmarking, and a
    /// guaranteed path where a GPU provider misbehaves).
    Cpu,
}

/// Whether this build has any GPU execution provider compiled in. DirectML and
/// CoreML come in automatically on their OS; CUDA needs the `gpu-cuda` feature.
#[cfg(feature = "gpu-inference")]
const HAS_GPU_PROVIDER: bool =
    cfg!(feature = "gpu-cuda") || cfg!(target_os = "windows") || cfg!(target_os = "macos");

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

    /// Load an ONNX model to run through onnxruntime on `device` (requires the
    /// `gpu-inference` feature).
    ///
    /// Unlike [`load`](Self::load) — which runs on the in-process tract / native
    /// interpreter — this runs the model through an onnxruntime session, so it
    /// can use a GPU execution provider. [`Device::Auto`] uses a GPU provider
    /// with a silent CPU fallback; [`Device::Gpu`] requires a GPU (errors if
    /// none); [`Device::Cpu`] forces CPU. Predictions match the CPU path within
    /// floating-point tolerance.
    ///
    /// ```no_run
    /// use millwright::prelude::*;
    ///
    /// # fn main() -> millwright::Result<()> {
    /// // Load a previously exported model onto the best available device.
    /// let model = InferenceModel::load_on("model.onnx", Device::Auto)?;
    /// let x = Frame::from_rows(vec![vec![1.0, 2.0, 3.0]], vec!["a".into(), "b".into(), "c".into()])?;
    /// let preds = model.predict(&x)?;
    /// # let _ = preds;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gpu-inference")]
    pub fn load_on(path: impl AsRef<Path>, device: Device) -> Result<InferenceModel> {
        let path = path.as_ref();

        match device {
            // Try a GPU provider first, then fall back to CPU. onnxruntime does
            // not fall back on its own once a provider is *registered* but then
            // fails at session creation (a driverless / device-less GPU provider
            // can error rather than yield), so we fall back explicitly: on any
            // failure of the GPU attempt, rebuild a CPU-only session. That is
            // what lets a model load on any system regardless of GPU state.
            Device::Auto => {
                if let Ok(session) = build_session(path, gpu_execution_providers(false, None)) {
                    return Ok(Self::wrap_ort(session));
                }
                Ok(Self::wrap_ort(build_session(
                    path,
                    cpu_execution_providers(),
                )?))
            }
            // Require a GPU: mark the GPU provider(s) `error_on_failure`, so a GPU
            // that can't initialize surfaces as an error instead of a silent CPU
            // downgrade. No compiled-in GPU provider at all is an error too.
            Device::Gpu => {
                if !HAS_GPU_PROVIDER {
                    return Err(Error::Backend(
                        "Device::Gpu requires a GPU execution provider, but none is compiled in \
                         (build on Windows/macOS, or enable the `gpu-cuda` feature)"
                            .into(),
                    ));
                }
                Ok(Self::wrap_ort(build_session(
                    path,
                    gpu_execution_providers(true, None),
                )?))
            }
            Device::Cpu => Ok(Self::wrap_ort(build_session(
                path,
                cpu_execution_providers(),
            )?)),
        }
    }

    /// Load an ONNX model across several GPUs, splitting each batch across them
    /// (requires the `gpu-inference` feature).
    ///
    /// Builds one onnxruntime session per entry in `device_ids`, each pinned to
    /// that GPU (the CUDA / DirectML device index), then splits a batch's rows
    /// evenly across the sessions and runs them concurrently — so throughput
    /// scales with the number of GPUs. This is data parallelism: it raises
    /// rows-per-second, it does not lower the latency of a single row.
    ///
    /// A GPU provider is required (feature `gpu-cuda`, or Windows/macOS); a
    /// device id that can't initialize is an error rather than a silent CPU
    /// fallback. Predictions match the single-device path.
    ///
    /// ```no_run
    /// use millwright::prelude::*;
    /// # fn main() -> millwright::Result<()> {
    /// // Split inference across GPU 0 and GPU 1.
    /// let model = InferenceModel::load_multi("model.onnx", &[0, 1])?;
    /// # let _ = model;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gpu-inference")]
    pub fn load_multi(path: impl AsRef<Path>, device_ids: &[i32]) -> Result<InferenceModel> {
        if !HAS_GPU_PROVIDER {
            return Err(Error::Backend(
                "load_multi requires a GPU execution provider, but none is compiled in \
                 (build on Windows/macOS, or enable the `gpu-cuda` feature)"
                    .into(),
            ));
        }
        if device_ids.is_empty() {
            return Err(Error::Backend(
                "load_multi needs at least one device id".into(),
            ));
        }
        let path = path.as_ref();
        let mut sessions = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            let session = build_session(path, gpu_execution_providers(true, Some(id)))?;
            sessions.push(std::sync::Mutex::new(session));
        }
        Ok(Self {
            backend: std::sync::Arc::new(Backend::OrtPool(sessions)),
        })
    }

    #[cfg(feature = "gpu-inference")]
    fn wrap_ort(session: ort::session::Session) -> InferenceModel {
        Self {
            backend: std::sync::Arc::new(Backend::Ort(std::sync::Mutex::new(session))),
        }
    }

    /// Run the model on a frame, returning one prediction per row.
    ///
    /// For a multi-column classifier score output the arg-max class index is
    /// returned; for a single-column output the value itself.
    pub fn predict(&self, frame: &Frame) -> Result<Vec<f64>> {
        match &*self.backend {
            Backend::Native(g) => g.run(frame),
            Backend::Tract(plan) => Self::tract_predict(plan, frame),
            #[cfg(feature = "gpu-inference")]
            Backend::Ort(session) => Self::ort_predict(session, frame),
            #[cfg(feature = "gpu-inference")]
            Backend::OrtPool(sessions) => Self::ort_pool_predict(sessions, frame),
        }
    }

    /// Run one batch through the onnxruntime session and reduce its output the
    /// same way [`tract_predict`](Self::tract_predict) does: an integer label
    /// output passes through; a multi-column score output becomes the arg-max
    /// class index; a single-column output is returned as-is.
    #[cfg(feature = "gpu-inference")]
    fn ort_predict(
        session: &std::sync::Mutex<ort::session::Session>,
        frame: &Frame,
    ) -> Result<Vec<f64>> {
        use ort::value::Tensor;

        let (n, p) = frame.shape();
        let data: Vec<f32> = frame.buf().iter().map(|v| *v as f32).collect();
        let input = Tensor::from_array(([n, p], data.into_boxed_slice()))
            .map_err(|e| Error::Backend(format!("onnxruntime input build failed: {e}")))?;
        let mut session = session
            .lock()
            .map_err(|_| Error::Backend("onnxruntime session mutex poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| Error::Backend(format!("onnxruntime run failed: {e}")))?;
        let value = &outputs[0];

        // Prefer an integer label output (classifier); fall back to floats.
        if let Ok(view) = value.try_extract_array::<i64>() {
            return Ok(view.iter().map(|v| *v as f64).collect());
        }
        let view = value
            .try_extract_array::<f32>()
            .map_err(|e| Error::Backend(format!("unexpected onnxruntime output type: {e}")))?;
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

    /// Split a batch's rows across the pooled per-GPU sessions, run them
    /// concurrently, and concatenate the predictions back in row order.
    #[cfg(feature = "gpu-inference")]
    fn ort_pool_predict(
        sessions: &[std::sync::Mutex<ort::session::Session>],
        frame: &Frame,
    ) -> Result<Vec<f64>> {
        if sessions.len() == 1 {
            return Self::ort_predict(&sessions[0], frame);
        }
        let (n, _) = frame.shape();
        let chunk = n.div_ceil(sessions.len()).max(1);
        let rows = frame.as_rows();
        let cols = frame.columns().to_vec();
        // One sub-frame per contiguous row-chunk (there are <= sessions.len()).
        let subframes: Vec<Frame> = rows
            .chunks(chunk)
            .map(|rs| Frame::from_rows(rs.to_vec(), cols.clone()))
            .collect::<Result<_>>()?;

        // Run each sub-frame on its own session on a separate thread.
        let mut parts: Vec<Result<Vec<f64>>> = Vec::with_capacity(subframes.len());
        std::thread::scope(|scope| {
            let handles: Vec<_> = subframes
                .iter()
                .zip(sessions.iter())
                .map(|(sf, sess)| scope.spawn(move || Self::ort_predict(sess, sf)))
                .collect();
            for handle in handles {
                parts.push(handle.join().unwrap_or_else(|_| {
                    Err(Error::Backend("onnxruntime worker thread panicked".into()))
                }));
            }
        });

        let mut out = Vec::with_capacity(n);
        for part in parts {
            out.extend(part?);
        }
        Ok(out)
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

/// Build an onnxruntime session from a model file and an ordered provider list.
#[cfg(feature = "gpu-inference")]
fn build_session(
    path: &Path,
    providers: Vec<ort::ep::ExecutionProviderDispatch>,
) -> Result<ort::session::Session> {
    use ort::session::Session;

    let mut builder = Session::builder()
        .map_err(|e| Error::Backend(format!("onnxruntime builder failed: {e}")))?;
    // Batch size (row count) varies from call to call, so the memory-pattern
    // optimizer (which assumes fixed input shapes) is a pessimization; disabling
    // it is also what onnxruntime's DirectML provider wants.
    builder = builder
        .with_memory_pattern(false)
        .map_err(|e| Error::Backend(format!("onnxruntime session option failed: {e}")))?;
    builder = builder
        .with_execution_providers(providers)
        .map_err(|e| Error::Backend(format!("onnxruntime provider setup failed: {e}")))?;
    builder
        .commit_from_file(path)
        .map_err(|e| Error::Backend(format!("onnxruntime load failed: {e}")))
}

/// The available GPU provider(s) first, then CPU as an in-session fallback for
/// nodes the GPU provider can't run. CUDA (feature `gpu-cuda`, opt-in) is
/// preferred when compiled in; DirectML (Windows) and CoreML (macOS) are
/// compiled in automatically on their OS. On a build with no GPU provider this
/// is just the CPU provider.
///
/// When `strict`, each GPU provider is marked `error_on_failure`, so a GPU that
/// can't initialize aborts session creation instead of silently yielding to CPU.
/// `device_id` pins the GPU provider to a specific device (for multi-GPU); `None`
/// uses the provider's default device. CoreML selects its own device.
#[cfg(feature = "gpu-inference")]
// The providers are selected by cfg, so a `vec![]` literal can't express this
// uniformly across builds.
#[allow(clippy::vec_init_then_push)]
fn gpu_execution_providers(
    strict: bool,
    device_id: Option<i32>,
) -> Vec<ort::ep::ExecutionProviderDispatch> {
    let mut providers = Vec::new();
    // Only compiled when a GPU provider actually exists for this build, so the
    // `strict` / `device_id` inputs and helper are never dead code on a GPU-less
    // target.
    #[cfg(any(feature = "gpu-cuda", target_os = "windows", target_os = "macos"))]
    {
        // Mark a GPU provider mandatory (or not), per `strict`.
        let gpu = |dispatch: ort::ep::ExecutionProviderDispatch| {
            if strict {
                dispatch.error_on_failure()
            } else {
                dispatch
            }
        };
        #[cfg(feature = "gpu-cuda")]
        {
            let mut ep = ort::ep::CUDA::default();
            if let Some(id) = device_id {
                ep = ep.with_device_id(id);
            }
            providers.push(gpu(ep.build()));
        }
        #[cfg(target_os = "windows")]
        {
            let mut ep = ort::ep::DirectML::default();
            if let Some(id) = device_id {
                ep = ep.with_device_id(id);
            }
            providers.push(gpu(ep.build()));
        }
        #[cfg(target_os = "macos")]
        {
            let _ = device_id; // CoreML selects its own device
            providers.push(gpu(ort::ep::CoreML::default().build()));
        }
    }
    #[cfg(not(any(feature = "gpu-cuda", target_os = "windows", target_os = "macos")))]
    let _ = (strict, device_id);
    // CPU carries ops with no GPU kernel (e.g. tree ensembles) and is the
    // fallback; it is never marked `error_on_failure`.
    providers.push(ort::ep::CPU::default().build());
    providers
}

/// The CPU provider only — the universal fallback that runs on any system.
#[cfg(feature = "gpu-inference")]
fn cpu_execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
    vec![ort::ep::CPU::default().build()]
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
