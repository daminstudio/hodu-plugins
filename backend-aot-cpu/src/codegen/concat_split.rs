//! Concat and Split operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_cli_plugin_sdk::{
    op_params::OpParams,
    ops::{ConcatOp, SplitOp},
    snapshot::SnapshotNode,
    PluginResult,
};
use std::fmt::Write;

pub fn write_concat(code: &mut String, node: &SnapshotNode, idx: usize, _op: ConcatOp) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let out = &node.output_layout;
    let num_inputs = node.input_ids.len();
    let num_dims = out.ndim();

    let concat_dim = match &node.params {
        Some(OpParams::Concat(p)) => {
            let d = p.dim.to_i64();
            if d < 0 {
                (num_dims as i64 + d) as usize
            } else {
                d as usize
            }
        },
        _ => 0,
    };

    let mut meta = vec![out.shape().size(), num_dims];
    meta.extend_from_slice(out.shape().dims());
    meta.push(concat_dim);
    meta.push(num_inputs);

    for inp in &node.input_layouts {
        meta.extend_from_slice(inp.shape().dims());
    }
    for inp in &node.input_layouts {
        meta.extend_from_slice(inp.strides());
    }
    for inp in &node.input_layouts {
        meta.push(inp.offset());
    }
    for _ in 0..num_inputs {
        meta.push(0); // buffer offsets (0 for separate tensors)
    }

    write_metadata(code, &format!("m{}", idx), &meta);

    write!(code, "    void* inp{}[] = {{", idx).unwrap();
    for (i, inp_id) in node.input_ids.iter().enumerate() {
        if i > 0 {
            write!(code, ",").unwrap();
        }
        write!(code, "t[{}]", inp_id.0).unwrap();
    }
    writeln!(code, "}};").unwrap();

    writeln!(
        code,
        "    hodu_cpu_concat_{}(inp{}, t[{}], m{});",
        suffix, idx, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}

pub fn write_split(code: &mut String, node: &SnapshotNode, idx: usize, _op: SplitOp) -> PluginResult<()> {
    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let out = &node.output_layout;
    let num_dims = inp.ndim();

    let (split_dim, split_offset) = match &node.params {
        Some(OpParams::Split(p)) => {
            let d = p.dim.to_i64();
            let dim = if d < 0 {
                (num_dims as i64 + d) as usize
            } else {
                d as usize
            };
            let mut offset = 0usize;
            for i in 0..p.output_index {
                if i < p.sizes.len() {
                    offset += p.sizes[i].to_usize();
                }
            }
            (dim, offset)
        },
        _ => (0, 0),
    };

    let mut meta = vec![out.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(split_dim);
    meta.push(out.shape().dims()[split_dim]);
    meta.push(split_offset);

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_split_{}(t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}
