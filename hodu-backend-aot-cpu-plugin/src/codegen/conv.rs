//! Convolution operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{op_params::OpParams, ops::ConvOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: ConvOp) -> PluginResult<()> {
    match op {
        ConvOp::Conv1d => write_conv(code, node, idx, 1, false, "conv1d"),
        ConvOp::Conv2d => write_conv(code, node, idx, 2, false, "conv2d"),
        ConvOp::Conv3d => write_conv(code, node, idx, 3, false, "conv3d"),
        ConvOp::ConvTranspose1d => write_conv(code, node, idx, 1, true, "conv_transpose1d"),
        ConvOp::ConvTranspose2d => write_conv(code, node, idx, 2, true, "conv_transpose2d"),
        ConvOp::ConvTranspose3d => write_conv(code, node, idx, 3, true, "conv_transpose3d"),
        ConvOp::Conv1dGradWeight => write_conv_grad(code, node, idx, 1, "conv1d_grad_weight"),
        ConvOp::Conv2dGradWeight => write_conv_grad(code, node, idx, 2, "conv2d_grad_weight"),
        ConvOp::Conv3dGradWeight => write_conv_grad(code, node, idx, 3, "conv3d_grad_weight"),
        ConvOp::ConvTranspose1dGradWeight => write_conv_grad(code, node, idx, 1, "conv_transpose1d_grad_weight"),
        ConvOp::ConvTranspose2dGradWeight => write_conv_grad(code, node, idx, 2, "conv_transpose2d_grad_weight"),
        ConvOp::ConvTranspose3dGradWeight => write_conv_grad(code, node, idx, 3, "conv_transpose3d_grad_weight"),
    }
}

fn write_conv(
    code: &mut String,
    node: &SnapshotNode,
    idx: usize,
    dims: usize,
    transpose: bool,
    name: &str,
) -> PluginResult<()> {
    let inp = &node.input_layouts[0];
    let weight = &node.input_layouts[1];
    let out = &node.output_layout;
    let suffix = dtype_suffix(node.output_dtype);

    let inp_shape = inp.shape().dims();
    let weight_shape = weight.shape().dims();
    let out_shape = out.shape().dims();

    let (stride, padding, dilation) = get_conv_params(&node.params, dims, transpose);

    let mut meta = vec![out.shape().size(), inp_shape[0], inp_shape[1], weight_shape[0]];

    match dims {
        1 => {
            meta.extend_from_slice(&[inp_shape[2], weight_shape[2], out_shape[2]]);
            meta.extend_from_slice(&[stride, padding, dilation]);
        },
        2 => {
            meta.extend_from_slice(&[
                inp_shape[2],
                inp_shape[3],
                weight_shape[2],
                weight_shape[3],
                out_shape[2],
                out_shape[3],
            ]);
            meta.extend_from_slice(&[stride, stride, padding, padding, dilation, dilation]);
        },
        3 => {
            meta.extend_from_slice(&[inp_shape[2], inp_shape[3], inp_shape[4]]);
            meta.extend_from_slice(&[weight_shape[2], weight_shape[3], weight_shape[4]]);
            meta.extend_from_slice(&[out_shape[2], out_shape[3], out_shape[4]]);
            meta.extend_from_slice(&[
                stride, stride, stride, padding, padding, padding, dilation, dilation, dilation,
            ]);
        },
        _ => {},
    }
    meta.extend_from_slice(&[inp.offset(), weight.offset()]);

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], t[{}], m{});",
        name, suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}

fn write_conv_grad(code: &mut String, node: &SnapshotNode, idx: usize, dims: usize, name: &str) -> PluginResult<()> {
    let inp = &node.input_layouts[0];
    let grad_out = &node.input_layouts[1];
    let weight = &node.output_layout;
    let suffix = dtype_suffix(node.output_dtype);

    let (stride, padding, dilation) = get_conv_grad_params(&node.params, dims);

    let mut meta = vec![weight.shape().size(), inp.ndim(), dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(grad_out.shape().dims());
    meta.extend_from_slice(weight.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.extend_from_slice(grad_out.strides());
    meta.push(inp.offset());
    meta.push(grad_out.offset());
    for _ in 0..dims {
        meta.push(stride);
    }
    for _ in 0..dims {
        meta.push(padding);
    }
    for _ in 0..dims {
        meta.push(dilation);
    }

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], t[{}], m{});",
        name, suffix, node.input_ids[0].0, node.input_ids[1].0, node.output_id.0, idx
    )
    .unwrap();

    Ok(())
}

fn get_conv_params(params: &Option<OpParams>, dims: usize, transpose: bool) -> (usize, usize, usize) {
    match (params, dims, transpose) {
        (Some(OpParams::Conv1d(p)), 1, false) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::Conv2d(p)), 2, false) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::Conv3d(p)), 3, false) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose1d(p)), 1, true) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose2d(p)), 2, true) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose3d(p)), 3, true) => (p.stride, p.padding, p.dilation),
        _ => (1, 0, 1),
    }
}

fn get_conv_grad_params(params: &Option<OpParams>, dims: usize) -> (usize, usize, usize) {
    match (params, dims) {
        (Some(OpParams::Conv1dGradWeight(p)), 1) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::Conv2dGradWeight(p)), 2) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::Conv3dGradWeight(p)), 3) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose1dGradWeight(p)), 1) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose2dGradWeight(p)), 2) => (p.stride, p.padding, p.dilation),
        (Some(OpParams::ConvTranspose3dGradWeight(p)), 3) => (p.stride, p.padding, p.dilation),
        _ => (1, 0, 1),
    }
}
