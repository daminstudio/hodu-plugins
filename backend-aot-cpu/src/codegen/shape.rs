//! Shape operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_cli_plugin_sdk::{
    ops::{ShapeOp, ShapeScalarsOp},
    snapshot::SnapshotNode,
    PluginResult,
};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: ShapeOp) -> PluginResult<()> {
    match op {
        ShapeOp::Reshape | ShapeOp::Flatten | ShapeOp::Squeeze | ShapeOp::Unsqueeze | ShapeOp::Broadcast => {
            // View operations - just copy pointer (data is shared)
            writeln!(code, "    t[{}] = t[{}];", node.output_id.0, node.input_ids[0].0).unwrap();
        },
        ShapeOp::Transpose | ShapeOp::Permute => {
            // Layout change - use contiguous copy
            write_contiguous(code, node, idx)?;
        },
    }
    Ok(())
}

pub fn write_scalars(code: &mut String, node: &SnapshotNode, idx: usize, op: ShapeScalarsOp) -> PluginResult<()> {
    match op {
        ShapeScalarsOp::Slice => write_contiguous(code, node, idx),
    }
}

fn write_contiguous(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];

    let mut meta = vec![inp.shape().size(), inp.ndim()];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_contiguous_{}(t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}
