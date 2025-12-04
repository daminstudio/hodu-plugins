//! Indexing operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{op_params::OpParams, ops::IndexingOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: IndexingOp) -> PluginResult<()> {
    match op {
        IndexingOp::IndexSelect => write_index_select(code, node, idx),
        IndexingOp::IndexPut => write_index_put(code, node, idx),
        IndexingOp::Gather => write_gather(code, node, idx),
        IndexingOp::Scatter => write_scatter(code, node, idx, "scatter"),
        IndexingOp::ScatterAdd => write_scatter(code, node, idx, "scatter_add"),
        IndexingOp::ScatterMax => write_scatter(code, node, idx, "scatter_max"),
        IndexingOp::ScatterMin => write_scatter(code, node, idx, "scatter_min"),
    }
}

fn get_dim(node: &SnapshotNode, num_dims: usize) -> usize {
    let d = match &node.params {
        Some(OpParams::IndexSelect(p)) => p.dim.to_i64(),
        Some(OpParams::IndexPut(p)) => p.dim.to_i64(),
        Some(OpParams::Gather(p)) => p.dim.to_i64(),
        Some(OpParams::Scatter(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterAdd(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterMax(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterMin(p)) => p.dim.to_i64(),
        _ => 0,
    };
    if d < 0 {
        (num_dims as i64 + d) as usize
    } else {
        d as usize
    }
}

fn write_index_select(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let indices = &node.input_layouts[1];
    let num_dims = inp.ndim();
    let dim = get_dim(node, num_dims);
    let num_indices: usize = indices.shape().dims().iter().product();

    let mut meta = vec![node.output_layout.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(dim);
    meta.push(num_indices);

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_index_select_{}(t[{}], (const int*)t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}

fn write_index_put(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let values = &node.input_layouts[2];
    let num_dims = inp.ndim();
    let dim = get_dim(node, num_dims);

    let mut meta = vec![node.output_layout.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(dim);
    meta.extend_from_slice(values.shape().dims());
    meta.extend_from_slice(values.strides());
    meta.push(values.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_index_put_{}(t[{}], (const int*)t[{}], t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.input_ids[1].0, node.input_ids[2].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}

fn write_gather(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let indices = &node.input_layouts[1];
    let out = &node.output_layout;
    let num_dims = inp.ndim();
    let dim = get_dim(node, num_dims);

    let mut meta = vec![out.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(dim);
    meta.extend_from_slice(indices.shape().dims());
    meta.extend_from_slice(indices.strides());
    meta.push(indices.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_gather_{}(t[{}], (const int*)t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}

fn write_scatter(code: &mut String, node: &SnapshotNode, idx: usize, name: &str) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let indices = &node.input_layouts[1];
    let src = &node.input_layouts[2];
    let out = &node.output_layout;
    let num_dims = inp.ndim();
    let dim = get_dim(node, num_dims);

    let mut meta = vec![out.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(dim);
    meta.extend_from_slice(indices.shape().dims());
    meta.extend_from_slice(indices.strides());
    meta.push(indices.offset());
    meta.extend_from_slice(src.shape().dims());
    meta.extend_from_slice(src.strides());
    meta.push(src.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], (const int*)t[{}], t[{}], t[{}], m{});",
        name, suffix, node.input_ids[0].0, node.input_ids[1].0, node.input_ids[2].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}
