//! Binary operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_cli_plugin_sdk::{
    ops::{BinaryLogicalOp, BinaryOp, CmpOp},
    snapshot::SnapshotNode,
    PluginResult,
};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: BinaryOp) -> PluginResult<()> {
    let name = match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Pow => "pow",
        BinaryOp::Maximum => "maximum",
        BinaryOp::Minimum => "minimum",
    };
    write_binary_call(code, node, idx, name)
}

pub fn write_logical(code: &mut String, node: &SnapshotNode, idx: usize, op: BinaryLogicalOp) -> PluginResult<()> {
    let name = match op {
        BinaryLogicalOp::LogicalAnd => "logical_and",
        BinaryLogicalOp::LogicalOr => "logical_or",
        BinaryLogicalOp::LogicalXor => "logical_xor",
    };
    write_binary_call(code, node, idx, name)
}

pub fn write_cmp(code: &mut String, node: &SnapshotNode, idx: usize, op: CmpOp) -> PluginResult<()> {
    let name = match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    };
    write_binary_call(code, node, idx, name)
}

fn write_binary_call(code: &mut String, node: &SnapshotNode, idx: usize, op_name: &str) -> PluginResult<()> {
    let lhs = &node.input_layouts[0];
    let rhs = &node.input_layouts[1];
    let suffix = dtype_suffix(node.output_dtype);

    let mut meta = vec![lhs.shape().size(), lhs.ndim()];
    meta.extend_from_slice(lhs.shape().dims());
    meta.extend_from_slice(rhs.shape().dims());
    meta.extend_from_slice(lhs.strides());
    meta.extend_from_slice(rhs.strides());
    meta.push(lhs.offset());
    meta.push(rhs.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], t[{}], m{});",
        op_name, suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}
