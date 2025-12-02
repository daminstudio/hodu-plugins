//! Memory operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_cli_plugin_sdk::{ops::MemoryOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: MemoryOp) -> PluginResult<()> {
    match op {
        MemoryOp::Contiguous => write_contiguous(code, node, idx),
    }
}

fn write_contiguous(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let out_numel: usize = node.output_layout.shape().dims().iter().product();
    let elem_size = node.output_dtype.get_size_in_bytes();

    // Allocate output
    writeln!(code, "    t[{}] = malloc({});", node.output_id.0, out_numel * elem_size).unwrap();

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
