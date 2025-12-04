//! Unary operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{
    ops::{CmpScalarOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp},
    snapshot::SnapshotNode,
    PluginResult,
};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: UnaryOp) -> PluginResult<()> {
    let name = match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Abs => "abs",
        UnaryOp::Sign => "sign",
        UnaryOp::Square => "square",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Recip => "recip",
        UnaryOp::Relu => "relu",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Gelu => "gelu",
        UnaryOp::Softplus => "softplus",
        UnaryOp::Silu => "silu",
        UnaryOp::Mish => "mish",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Exp => "exp",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Exp10 => "exp10",
        UnaryOp::Ln => "ln",
        UnaryOp::Log2 => "log2",
        UnaryOp::Log10 => "log10",
    };
    write_unary_call(code, node, idx, name)
}

pub fn write_logical(code: &mut String, node: &SnapshotNode, idx: usize, op: UnaryLogicalOp) -> PluginResult<()> {
    let name = match op {
        UnaryLogicalOp::LogicalNot => "logical_not",
    };
    write_unary_call(code, node, idx, name)
}

pub fn write_scalar(code: &mut String, node: &SnapshotNode, idx: usize, op: UnaryScalarOp) -> PluginResult<()> {
    let name = match op {
        UnaryScalarOp::AddScalar => "add_scalar",
        UnaryScalarOp::SubScalar => "sub_scalar",
        UnaryScalarOp::MulScalar => "mul_scalar",
        UnaryScalarOp::DivScalar => "div_scalar",
        UnaryScalarOp::PowScalar => "pow_scalar",
        UnaryScalarOp::MaximumScalar => "maximum_scalar",
        UnaryScalarOp::MinimumScalar => "minimum_scalar",
        UnaryScalarOp::LeakyRelu => "leaky_relu",
        UnaryScalarOp::Elu => "elu",
        UnaryScalarOp::Prelu => "prelu",
    };
    write_unary_scalar_call(code, node, idx, name)
}

pub fn write_cmp_scalar(code: &mut String, node: &SnapshotNode, idx: usize, op: CmpScalarOp) -> PluginResult<()> {
    let name = match op {
        CmpScalarOp::EqScalar => "eq_scalar",
        CmpScalarOp::NeScalar => "ne_scalar",
        CmpScalarOp::LtScalar => "lt_scalar",
        CmpScalarOp::LeScalar => "le_scalar",
        CmpScalarOp::GtScalar => "gt_scalar",
        CmpScalarOp::GeScalar => "ge_scalar",
    };
    write_unary_scalar_call(code, node, idx, name)
}

fn write_unary_call(code: &mut String, node: &SnapshotNode, idx: usize, op_name: &str) -> PluginResult<()> {
    let inp = &node.input_layouts[0];
    let suffix = dtype_suffix(node.output_dtype);

    let mut meta = vec![inp.shape().size(), inp.ndim()];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], m{});",
        op_name, suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}

fn write_unary_scalar_call(code: &mut String, node: &SnapshotNode, idx: usize, op_name: &str) -> PluginResult<()> {
    let inp = &node.input_layouts[0];
    let suffix = dtype_suffix(node.output_dtype);

    // Get scalar value from params
    let scalar = node
        .params
        .as_ref()
        .and_then(|p| {
            use hodu_plugin_sdk::op_params::OpParams;
            match p {
                OpParams::UnaryScalar(s) => Some(s.scalar.to_f64()),
                _ => None,
            }
        })
        .unwrap_or(0.0);

    let mut meta = vec![inp.shape().size(), inp.ndim()];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(code, "    double s{} = {};", idx, scalar).unwrap();
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], m{}, &s{});",
        op_name, suffix, node.input_ids[0].0, node.output_id.0, idx, idx
    )
    .unwrap();

    Ok(())
}
