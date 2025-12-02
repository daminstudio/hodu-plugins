//! Cast operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_cli_plugin_sdk::{ops::CastOp, snapshot::SnapshotNode, PluginResult};
use hodu_core::types::DType;
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, _op: CastOp) -> PluginResult<()> {
    let inp = &node.input_layouts[0];

    // Get input dtype from first input tensor's layout
    // Note: We need to get this from somewhere - for now assume it's stored or passed differently
    let inp_suffix = dtype_suffix(get_input_dtype(node));
    let out_suffix = dtype_suffix(node.output_dtype);

    let mut meta = vec![inp.shape().size(), inp.ndim()];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_cast_{}_to_{}(t[{}], t[{}], m{});",
        inp_suffix, out_suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}

fn get_input_dtype(node: &SnapshotNode) -> DType {
    // Cast operation: input dtype is different from output dtype
    // We need to track this somewhere - for now default to F32
    // In practice, this should come from the snapshot node's input info
    node.output_dtype // Placeholder - should be input dtype
}
