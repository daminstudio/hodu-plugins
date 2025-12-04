//! Matrix operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{ops::MatrixOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: MatrixOp) -> PluginResult<()> {
    match op {
        MatrixOp::Matmul => write_matmul(code, node, idx),
        MatrixOp::Dot => write_dot(code, node, idx),
    }
}

fn write_matmul(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let lhs = &node.input_layouts[0];
    let rhs = &node.input_layouts[1];
    let out = &node.output_layout;
    let suffix = dtype_suffix(node.output_dtype);

    let lhs_ndim = lhs.ndim();
    let rhs_ndim = rhs.ndim();
    let batch_ndim = out.ndim().saturating_sub(2);

    let m = lhs.shape().dims()[lhs_ndim - 2];
    let k = lhs.shape().dims()[lhs_ndim - 1];
    let n = rhs.shape().dims()[rhs_ndim - 1];

    let mut meta = vec![out.shape().size(), lhs_ndim, rhs_ndim, batch_ndim];
    meta.extend_from_slice(lhs.shape().dims());
    meta.extend_from_slice(rhs.shape().dims());
    if batch_ndim > 0 {
        meta.extend_from_slice(&out.shape().dims()[..batch_ndim]);
    }
    meta.extend_from_slice(lhs.strides());
    meta.extend_from_slice(rhs.strides());
    meta.push(lhs.offset());
    meta.push(rhs.offset());
    meta.push(m);
    meta.push(k);
    meta.push(n);

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_matmul_{}(t[{}], t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}

fn write_dot(code: &mut String, node: &SnapshotNode, idx: usize) -> PluginResult<()> {
    let lhs = &node.input_layouts[0];
    let rhs = &node.input_layouts[1];
    let suffix = dtype_suffix(node.output_dtype);

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();
    let lhs_strides = lhs.strides();
    let rhs_strides = rhs.strides();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let meta = vec![
        m,
        k,
        n,
        lhs_strides[0],
        lhs_strides[1],
        rhs_strides[0],
        rhs_strides[1],
        lhs.offset(),
        rhs.offset(),
    ];

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_dot_{}(t[{}], t[{}], t[{}], m{});",
        suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}
