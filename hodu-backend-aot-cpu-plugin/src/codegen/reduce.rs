//! Reduce operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{op_params::OpParams, ops::ReduceOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: ReduceOp) -> PluginResult<()> {
    let name = match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Prod => "prod",
        ReduceOp::Std => "std",
        ReduceOp::Var => "var",
        ReduceOp::Norm => "norm",
        ReduceOp::ArgMax => "argmax",
        ReduceOp::ArgMin => "argmin",
        ReduceOp::Any => "any",
        ReduceOp::All => "all",
    };

    let inp = &node.input_layouts[0];
    let out = &node.output_layout;
    let suffix = dtype_suffix(node.output_dtype);
    let shape_len = inp.ndim();

    let (reduce_dims_raw, keep_dim) = match &node.params {
        Some(OpParams::Reduce(p)) => {
            let dims: Vec<i64> = p.dims.iter().map(|s| s.to_i64()).collect();
            (dims, if p.keep_dim { 1usize } else { 0usize })
        },
        _ => (vec![], 0usize),
    };

    let mut reduce_dims: Vec<usize> = reduce_dims_raw
        .iter()
        .map(|&d| {
            if d < 0 {
                (shape_len as i64 + d) as usize
            } else {
                d as usize
            }
        })
        .collect();

    if reduce_dims.is_empty() {
        reduce_dims = (0..shape_len).collect();
    }

    let reduce_size: usize = reduce_dims.iter().map(|&d| inp.shape().dims()[d]).product();

    let mut meta = Vec::new();
    meta.push(shape_len);
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.push(out.ndim());
    meta.extend_from_slice(out.shape().dims());
    meta.push(reduce_dims.len());
    meta.extend(reduce_dims.iter().copied());
    meta.push(keep_dim);
    meta.push(reduce_size);

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], m{});",
        name, suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}
