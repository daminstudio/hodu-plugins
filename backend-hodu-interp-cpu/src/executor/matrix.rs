//! Matrix operation executors

use super::{get_tensor, TensorStorage};
use hodu_cli_plugin_sdk::{ops, snapshot::SnapshotNode, PluginError, PluginResult};
use hodu_core::types::DType;
use hodu_cpu_kernels::{call_ops_dot, call_ops_matmul, Kernel};
use std::collections::HashMap;

pub fn execute_matrix(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::MatrixOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    use ops::MatrixOp;

    match op {
        MatrixOp::Matmul => execute_matmul(tensors, node),
        MatrixOp::Dot => execute_dot(tensors, node),
    }
}

#[allow(clippy::vec_init_then_push)]
fn execute_matmul(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> PluginResult<()> {
    let lhs_id = node.input_ids[0].0;
    let rhs_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let lhs = get_tensor(tensors, lhs_id)?;
    let rhs = get_tensor(tensors, rhs_id)?;
    let lhs_dtype = lhs.dtype;
    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();

    let lhs_layout = &node.input_layouts[0];
    let rhs_layout = &node.input_layouts[1];
    let out_layout = &node.output_layout;

    let out_shape = out_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_matmul_kernel(lhs_dtype);

    // Build matmul metadata
    let lhs_ndim = lhs_layout.ndim();
    let rhs_ndim = rhs_layout.ndim();
    let batch_ndim = out_layout.ndim().saturating_sub(2);

    let m = lhs_layout.shape().dims()[lhs_ndim - 2];
    let k = lhs_layout.shape().dims()[lhs_ndim - 1];
    let n = rhs_layout.shape().dims()[rhs_ndim - 1];

    let num_els = out_layout.shape().size();

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(lhs_ndim);
    metadata.push(rhs_ndim);
    metadata.push(batch_ndim);

    // lhs_shape
    metadata.extend_from_slice(lhs_layout.shape().dims());
    // rhs_shape
    metadata.extend_from_slice(rhs_layout.shape().dims());
    // batch_shape
    if batch_ndim > 0 {
        metadata.extend_from_slice(&out_layout.shape().dims()[..batch_ndim]);
    }
    // lhs_strides
    metadata.extend_from_slice(lhs_layout.strides());
    // rhs_strides
    metadata.extend_from_slice(rhs_layout.strides());
    // offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());
    // M, K, N
    metadata.push(m);
    metadata.push(k);
    metadata.push(n);

    call_ops_matmul(kernel, lhs_ptr, rhs_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_dot(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> PluginResult<()> {
    let lhs_id = node.input_ids[0].0;
    let rhs_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let lhs = get_tensor(tensors, lhs_id)?;
    let rhs = get_tensor(tensors, rhs_id)?;
    let lhs_dtype = lhs.dtype;
    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();

    let lhs_layout = &node.input_layouts[0];
    let rhs_layout = &node.input_layouts[1];

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_dot_kernel(lhs_dtype);

    // 2D matrix multiplication: [M, K] x [K, N] = [M, N]
    let lhs_shape = lhs_layout.shape().dims();
    let rhs_shape = rhs_layout.shape().dims();
    let lhs_strides = lhs_layout.strides();
    let rhs_strides = rhs_layout.strides();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    // metadata: [M, K, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    let metadata = vec![
        m,
        k,
        n,
        lhs_strides[0],
        lhs_strides[1],
        rhs_strides[0],
        rhs_strides[1],
        lhs_layout.offset(),
        rhs_layout.offset(),
    ];

    call_ops_dot(kernel, lhs_ptr, rhs_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_matmul_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::matmul;
    match dtype {
        DType::F8E4M3 => matmul::F8E4M3,
        DType::F8E5M2 => matmul::F8E5M2,
        DType::BF16 => matmul::BF16,
        DType::F16 => matmul::F16,
        DType::F32 => matmul::F32,
        DType::F64 => matmul::F64,
        DType::U8 => matmul::U8,
        DType::U16 => matmul::U16,
        DType::U32 => matmul::U32,
        DType::U64 => matmul::U64,
        DType::I8 => matmul::I8,
        DType::I16 => matmul::I16,
        DType::I32 => matmul::I32,
        DType::I64 => matmul::I64,
        _ => panic!("Unsupported dtype for matmul: {:?}", dtype),
    }
}

fn get_dot_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::dot;
    match dtype {
        DType::F8E4M3 => dot::F8E4M3,
        DType::F8E5M2 => dot::F8E5M2,
        DType::BF16 => dot::BF16,
        DType::F16 => dot::F16,
        DType::F32 => dot::F32,
        DType::F64 => dot::F64,
        DType::U8 => dot::U8,
        DType::U16 => dot::U16,
        DType::U32 => dot::U32,
        DType::U64 => dot::U64,
        DType::I8 => dot::I8,
        DType::I16 => dot::I16,
        DType::I32 => dot::I32,
        DType::I64 => dot::I64,
        _ => panic!("Unsupported dtype for dot: {:?}", dtype),
    }
}
