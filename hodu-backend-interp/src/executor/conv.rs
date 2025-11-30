//! Convolution operation executors

use super::{get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_conv, call_ops_conv_grad_weight, Kernel};
use hodu_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_conv(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::ConvOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::ConvOp;

    match op {
        ConvOp::Conv1d => execute_conv_nd(tensors, node, 1, false),
        ConvOp::Conv2d => execute_conv_nd(tensors, node, 2, false),
        ConvOp::Conv3d => execute_conv_nd(tensors, node, 3, false),
        ConvOp::ConvTranspose1d => execute_conv_nd(tensors, node, 1, true),
        ConvOp::ConvTranspose2d => execute_conv_nd(tensors, node, 2, true),
        ConvOp::ConvTranspose3d => execute_conv_nd(tensors, node, 3, true),
        ConvOp::Conv1dGradWeight => execute_conv_grad_weight(tensors, node, 1, false),
        ConvOp::Conv2dGradWeight => execute_conv_grad_weight(tensors, node, 2, false),
        ConvOp::Conv3dGradWeight => execute_conv_grad_weight(tensors, node, 3, false),
        ConvOp::ConvTranspose1dGradWeight => execute_conv_grad_weight(tensors, node, 1, true),
        ConvOp::ConvTranspose2dGradWeight => execute_conv_grad_weight(tensors, node, 2, true),
        ConvOp::ConvTranspose3dGradWeight => execute_conv_grad_weight(tensors, node, 3, true),
    }
}

fn execute_conv_nd(
    tensors: &mut HashMap<usize, TensorStorage>,
    node: &SnapshotNode,
    spatial_dims: usize,
    transpose: bool,
) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let weight_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let weight = get_tensor(tensors, weight_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let weight_ptr = weight.as_ptr();

    let input_layout = &node.input_layouts[0];
    let weight_layout = &node.input_layouts[1];
    let out_layout = &node.output_layout;
    let out_shape = out_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_conv_kernel(spatial_dims, transpose, input_dtype);

    let metadata = build_conv_metadata_from_params(
        input_layout.shape().dims(),
        weight_layout.shape().dims(),
        out_layout.shape().dims(),
        input_layout.offset(),
        weight_layout.offset(),
        &node.params,
        spatial_dims,
    );

    call_ops_conv(kernel, input_ptr, weight_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_conv_grad_weight(
    tensors: &mut HashMap<usize, TensorStorage>,
    node: &SnapshotNode,
    spatial_dims: usize,
    transpose: bool,
) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let grad_output_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let grad_output = get_tensor(tensors, grad_output_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let grad_output_ptr = grad_output.as_ptr();

    let input_layout = &node.input_layouts[0];
    let grad_output_layout = &node.input_layouts[1];
    let out_layout = &node.output_layout;
    let out_shape = out_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_conv_grad_weight_kernel(spatial_dims, transpose, input_dtype);

    let metadata = build_conv_grad_weight_metadata_from_params(
        input_layout,
        grad_output_layout,
        out_layout,
        &node.params,
        spatial_dims,
    );

    call_ops_conv_grad_weight(kernel, input_ptr, grad_output_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

#[allow(clippy::vec_init_then_push)]
fn build_conv_metadata_from_params(
    input_shape: &[usize],
    weight_shape: &[usize],
    out_shape: &[usize],
    input_offset: usize,
    weight_offset: usize,
    params: &Option<OpParams>,
    spatial_dims: usize,
) -> Vec<usize> {
    let num_els: usize = out_shape.iter().product();
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let out_channels = weight_shape[0];

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(batch);
    metadata.push(in_channels);
    metadata.push(out_channels);

    match spatial_dims {
        1 => {
            let in_width = input_shape[2];
            let kernel_width = weight_shape[2];
            let out_width = out_shape[2];

            let (stride, padding, dilation) = match params {
                Some(OpParams::Conv1d(p)) => (p.stride, p.padding, p.dilation),
                Some(OpParams::ConvTranspose1d(p)) => (p.stride, p.padding, p.dilation),
                _ => (1, 0, 1),
            };

            metadata.push(in_width);
            metadata.push(kernel_width);
            metadata.push(out_width);
            metadata.push(stride);
            metadata.push(padding);
            metadata.push(dilation);
            metadata.push(input_offset);
            metadata.push(weight_offset);
        },
        2 => {
            let in_height = input_shape[2];
            let in_width = input_shape[3];
            let kernel_height = weight_shape[2];
            let kernel_width = weight_shape[3];
            let out_height = out_shape[2];
            let out_width = out_shape[3];

            let (stride, padding, dilation) = match params {
                Some(OpParams::Conv2d(p)) => (p.stride, p.padding, p.dilation),
                Some(OpParams::ConvTranspose2d(p)) => (p.stride, p.padding, p.dilation),
                _ => (1, 0, 1),
            };

            metadata.push(in_height);
            metadata.push(in_width);
            metadata.push(kernel_height);
            metadata.push(kernel_width);
            metadata.push(out_height);
            metadata.push(out_width);
            metadata.push(stride);
            metadata.push(stride);
            metadata.push(padding);
            metadata.push(padding);
            metadata.push(dilation);
            metadata.push(dilation);
            metadata.push(input_offset);
            metadata.push(weight_offset);
        },
        3 => {
            let in_depth = input_shape[2];
            let in_height = input_shape[3];
            let in_width = input_shape[4];
            let kernel_depth = weight_shape[2];
            let kernel_height = weight_shape[3];
            let kernel_width = weight_shape[4];
            let out_depth = out_shape[2];
            let out_height = out_shape[3];
            let out_width = out_shape[4];

            let (stride, padding, dilation) = match params {
                Some(OpParams::Conv3d(p)) => (p.stride, p.padding, p.dilation),
                Some(OpParams::ConvTranspose3d(p)) => (p.stride, p.padding, p.dilation),
                _ => (1, 0, 1),
            };

            metadata.push(in_depth);
            metadata.push(in_height);
            metadata.push(in_width);
            metadata.push(kernel_depth);
            metadata.push(kernel_height);
            metadata.push(kernel_width);
            metadata.push(out_depth);
            metadata.push(out_height);
            metadata.push(out_width);
            metadata.push(stride);
            metadata.push(stride);
            metadata.push(stride);
            metadata.push(padding);
            metadata.push(padding);
            metadata.push(padding);
            metadata.push(dilation);
            metadata.push(dilation);
            metadata.push(dilation);
            metadata.push(input_offset);
            metadata.push(weight_offset);
        },
        _ => panic!("Unsupported spatial dims: {}", spatial_dims),
    }

    metadata
}

fn build_conv_grad_weight_metadata_from_params(
    input_layout: &hodu_plugin_sdk::Layout,
    grad_output_layout: &hodu_plugin_sdk::Layout,
    weight_layout: &hodu_plugin_sdk::Layout,
    params: &Option<OpParams>,
    spatial_dims: usize,
) -> Vec<usize> {
    let input_ndim = input_layout.ndim();
    let num_els: usize = weight_layout.shape().size();

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(input_ndim);
    metadata.push(spatial_dims);

    metadata.extend_from_slice(input_layout.shape().dims());
    metadata.extend_from_slice(grad_output_layout.shape().dims());
    metadata.extend_from_slice(weight_layout.shape().dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.extend_from_slice(grad_output_layout.strides());
    metadata.push(input_layout.offset());
    metadata.push(grad_output_layout.offset());

    let (stride, padding, dilation) = match params {
        Some(OpParams::Conv1dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        Some(OpParams::Conv2dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        Some(OpParams::Conv3dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        Some(OpParams::ConvTranspose1dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        Some(OpParams::ConvTranspose2dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        Some(OpParams::ConvTranspose3dGradWeight(p)) => (p.stride, p.padding, p.dilation),
        _ => (1, 0, 1),
    };

    for _ in 0..spatial_dims {
        metadata.push(stride);
    }
    for _ in 0..spatial_dims {
        metadata.push(padding);
    }
    for _ in 0..spatial_dims {
        metadata.push(dilation);
    }

    metadata
}

fn get_conv_kernel(spatial_dims: usize, transpose: bool, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (spatial_dims, transpose, dtype) {
        (1, false, DType::F8E4M3) => conv1d::F8E4M3,
        (1, false, DType::F8E5M2) => conv1d::F8E5M2,
        (1, false, DType::BF16) => conv1d::BF16,
        (1, false, DType::F16) => conv1d::F16,
        (1, false, DType::F32) => conv1d::F32,
        (1, false, DType::F64) => conv1d::F64,

        (2, false, DType::F8E4M3) => conv2d::F8E4M3,
        (2, false, DType::F8E5M2) => conv2d::F8E5M2,
        (2, false, DType::BF16) => conv2d::BF16,
        (2, false, DType::F16) => conv2d::F16,
        (2, false, DType::F32) => conv2d::F32,
        (2, false, DType::F64) => conv2d::F64,

        (3, false, DType::F8E4M3) => conv3d::F8E4M3,
        (3, false, DType::F8E5M2) => conv3d::F8E5M2,
        (3, false, DType::BF16) => conv3d::BF16,
        (3, false, DType::F16) => conv3d::F16,
        (3, false, DType::F32) => conv3d::F32,
        (3, false, DType::F64) => conv3d::F64,

        (1, true, DType::F8E4M3) => conv_transpose1d::F8E4M3,
        (1, true, DType::F8E5M2) => conv_transpose1d::F8E5M2,
        (1, true, DType::BF16) => conv_transpose1d::BF16,
        (1, true, DType::F16) => conv_transpose1d::F16,
        (1, true, DType::F32) => conv_transpose1d::F32,
        (1, true, DType::F64) => conv_transpose1d::F64,

        (2, true, DType::F8E4M3) => conv_transpose2d::F8E4M3,
        (2, true, DType::F8E5M2) => conv_transpose2d::F8E5M2,
        (2, true, DType::BF16) => conv_transpose2d::BF16,
        (2, true, DType::F16) => conv_transpose2d::F16,
        (2, true, DType::F32) => conv_transpose2d::F32,
        (2, true, DType::F64) => conv_transpose2d::F64,

        (3, true, DType::F8E4M3) => conv_transpose3d::F8E4M3,
        (3, true, DType::F8E5M2) => conv_transpose3d::F8E5M2,
        (3, true, DType::BF16) => conv_transpose3d::BF16,
        (3, true, DType::F16) => conv_transpose3d::F16,
        (3, true, DType::F32) => conv_transpose3d::F32,
        (3, true, DType::F64) => conv_transpose3d::F64,

        _ => panic!(
            "Unsupported conv kernel: dims={}, transpose={}, dtype={:?}",
            spatial_dims, transpose, dtype
        ),
    }
}

fn get_conv_grad_weight_kernel(spatial_dims: usize, transpose: bool, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (spatial_dims, transpose, dtype) {
        (1, false, DType::F8E4M3) => conv1d_grad_weight::F8E4M3,
        (1, false, DType::F8E5M2) => conv1d_grad_weight::F8E5M2,
        (1, false, DType::BF16) => conv1d_grad_weight::BF16,
        (1, false, DType::F16) => conv1d_grad_weight::F16,
        (1, false, DType::F32) => conv1d_grad_weight::F32,
        (1, false, DType::F64) => conv1d_grad_weight::F64,

        (2, false, DType::F8E4M3) => conv2d_grad_weight::F8E4M3,
        (2, false, DType::F8E5M2) => conv2d_grad_weight::F8E5M2,
        (2, false, DType::BF16) => conv2d_grad_weight::BF16,
        (2, false, DType::F16) => conv2d_grad_weight::F16,
        (2, false, DType::F32) => conv2d_grad_weight::F32,
        (2, false, DType::F64) => conv2d_grad_weight::F64,

        (3, false, DType::F8E4M3) => conv3d_grad_weight::F8E4M3,
        (3, false, DType::F8E5M2) => conv3d_grad_weight::F8E5M2,
        (3, false, DType::BF16) => conv3d_grad_weight::BF16,
        (3, false, DType::F16) => conv3d_grad_weight::F16,
        (3, false, DType::F32) => conv3d_grad_weight::F32,
        (3, false, DType::F64) => conv3d_grad_weight::F64,

        (1, true, DType::F8E4M3) => conv_transpose1d_grad_weight::F8E4M3,
        (1, true, DType::F8E5M2) => conv_transpose1d_grad_weight::F8E5M2,
        (1, true, DType::BF16) => conv_transpose1d_grad_weight::BF16,
        (1, true, DType::F16) => conv_transpose1d_grad_weight::F16,
        (1, true, DType::F32) => conv_transpose1d_grad_weight::F32,
        (1, true, DType::F64) => conv_transpose1d_grad_weight::F64,

        (2, true, DType::F8E4M3) => conv_transpose2d_grad_weight::F8E4M3,
        (2, true, DType::F8E5M2) => conv_transpose2d_grad_weight::F8E5M2,
        (2, true, DType::BF16) => conv_transpose2d_grad_weight::BF16,
        (2, true, DType::F16) => conv_transpose2d_grad_weight::F16,
        (2, true, DType::F32) => conv_transpose2d_grad_weight::F32,
        (2, true, DType::F64) => conv_transpose2d_grad_weight::F64,

        (3, true, DType::F8E4M3) => conv_transpose3d_grad_weight::F8E4M3,
        (3, true, DType::F8E5M2) => conv_transpose3d_grad_weight::F8E5M2,
        (3, true, DType::BF16) => conv_transpose3d_grad_weight::BF16,
        (3, true, DType::F16) => conv_transpose3d_grad_weight::F16,
        (3, true, DType::F32) => conv_transpose3d_grad_weight::F32,
        (3, true, DType::F64) => conv_transpose3d_grad_weight::F64,

        _ => panic!(
            "Unsupported conv grad_weight kernel: dims={}, transpose={}, dtype={:?}",
            spatial_dims, transpose, dtype
        ),
    }
}
