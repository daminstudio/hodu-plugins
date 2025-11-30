//! Reduce operation executors

use super::{get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_reduce, Kernel};
use hodu_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, DType, HoduError, HoduResult, Layout};
use std::collections::HashMap;

pub fn execute_reduce(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::ReduceOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::ReduceOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let input_layout = &node.input_layouts[0];
    let out_layout = &node.output_layout;

    let out_shape = out_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        ReduceOp::Sum => get_kernel("sum", input_dtype),
        ReduceOp::Mean => get_kernel("mean", input_dtype),
        ReduceOp::Max => get_kernel("max", input_dtype),
        ReduceOp::Min => get_kernel("min", input_dtype),
        ReduceOp::Prod => get_kernel("prod", input_dtype),
        ReduceOp::Std => get_kernel("std", input_dtype),
        ReduceOp::Var => get_kernel("var", input_dtype),
        ReduceOp::Norm => get_kernel("norm", input_dtype),
        ReduceOp::ArgMax => get_kernel("argmax", input_dtype),
        ReduceOp::ArgMin => get_kernel("argmin", input_dtype),
        ReduceOp::Any => get_kernel("any", input_dtype),
        ReduceOp::All => get_kernel("all", input_dtype),
    };

    let metadata = build_reduce_metadata(input_layout, out_layout, &node.params);

    call_ops_reduce(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn build_reduce_metadata(input_layout: &Layout, output_layout: &Layout, params: &Option<OpParams>) -> Vec<usize> {
    let shape_len = input_layout.ndim();
    let output_shape_len = output_layout.ndim();

    // Extract reduce_dims and keep_dim from params
    let (reduce_dims_raw, keep_dim) = match params {
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

    // If no reduce dims specified, reduce all
    if reduce_dims.is_empty() {
        reduce_dims = (0..shape_len).collect();
    }

    // Calculate reduce_size
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_layout.shape().dims()[d]).product();

    // Build metadata
    let mut metadata = Vec::new();

    // shape_len
    metadata.push(shape_len);
    // shape
    metadata.extend_from_slice(input_layout.shape().dims());
    // strides
    metadata.extend_from_slice(input_layout.strides());
    // offset
    metadata.push(input_layout.offset());
    // output_shape_len
    metadata.push(output_shape_len);
    // output_shape
    metadata.extend_from_slice(output_layout.shape().dims());
    // num_reduce_dims
    metadata.push(reduce_dims.len());
    // reduce_dims
    metadata.extend(reduce_dims.iter().copied());
    // keep_dim
    metadata.push(keep_dim);
    // reduce_size
    metadata.push(reduce_size);

    metadata
}

fn get_kernel(op_name: &str, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (op_name, dtype) {
        ("sum", DType::F8E4M3) => sum::F8E4M3,
        ("sum", DType::F8E5M2) => sum::F8E5M2,
        ("sum", DType::BF16) => sum::BF16,
        ("sum", DType::F16) => sum::F16,
        ("sum", DType::F32) => sum::F32,
        ("sum", DType::F64) => sum::F64,
        ("sum", DType::U8) => sum::U8,
        ("sum", DType::U16) => sum::U16,
        ("sum", DType::U32) => sum::U32,
        ("sum", DType::U64) => sum::U64,
        ("sum", DType::I8) => sum::I8,
        ("sum", DType::I16) => sum::I16,
        ("sum", DType::I32) => sum::I32,
        ("sum", DType::I64) => sum::I64,

        ("mean", DType::F8E4M3) => mean::F8E4M3,
        ("mean", DType::F8E5M2) => mean::F8E5M2,
        ("mean", DType::BF16) => mean::BF16,
        ("mean", DType::F16) => mean::F16,
        ("mean", DType::F32) => mean::F32,
        ("mean", DType::F64) => mean::F64,

        ("max", DType::F8E4M3) => max::F8E4M3,
        ("max", DType::F8E5M2) => max::F8E5M2,
        ("max", DType::BF16) => max::BF16,
        ("max", DType::F16) => max::F16,
        ("max", DType::F32) => max::F32,
        ("max", DType::F64) => max::F64,
        ("max", DType::U8) => max::U8,
        ("max", DType::U16) => max::U16,
        ("max", DType::U32) => max::U32,
        ("max", DType::U64) => max::U64,
        ("max", DType::I8) => max::I8,
        ("max", DType::I16) => max::I16,
        ("max", DType::I32) => max::I32,
        ("max", DType::I64) => max::I64,

        ("min", DType::F8E4M3) => min::F8E4M3,
        ("min", DType::F8E5M2) => min::F8E5M2,
        ("min", DType::BF16) => min::BF16,
        ("min", DType::F16) => min::F16,
        ("min", DType::F32) => min::F32,
        ("min", DType::F64) => min::F64,
        ("min", DType::U8) => min::U8,
        ("min", DType::U16) => min::U16,
        ("min", DType::U32) => min::U32,
        ("min", DType::U64) => min::U64,
        ("min", DType::I8) => min::I8,
        ("min", DType::I16) => min::I16,
        ("min", DType::I32) => min::I32,
        ("min", DType::I64) => min::I64,

        ("prod", DType::F8E4M3) => prod::F8E4M3,
        ("prod", DType::F8E5M2) => prod::F8E5M2,
        ("prod", DType::BF16) => prod::BF16,
        ("prod", DType::F16) => prod::F16,
        ("prod", DType::F32) => prod::F32,
        ("prod", DType::F64) => prod::F64,
        ("prod", DType::U8) => prod::U8,
        ("prod", DType::U16) => prod::U16,
        ("prod", DType::U32) => prod::U32,
        ("prod", DType::U64) => prod::U64,
        ("prod", DType::I8) => prod::I8,
        ("prod", DType::I16) => prod::I16,
        ("prod", DType::I32) => prod::I32,
        ("prod", DType::I64) => prod::I64,

        ("std", DType::F8E4M3) => std::F8E4M3,
        ("std", DType::F8E5M2) => std::F8E5M2,
        ("std", DType::BF16) => std::BF16,
        ("std", DType::F16) => std::F16,
        ("std", DType::F32) => std::F32,
        ("std", DType::F64) => std::F64,

        ("var", DType::F8E4M3) => var::F8E4M3,
        ("var", DType::F8E5M2) => var::F8E5M2,
        ("var", DType::BF16) => var::BF16,
        ("var", DType::F16) => var::F16,
        ("var", DType::F32) => var::F32,
        ("var", DType::F64) => var::F64,

        ("norm", DType::F8E4M3) => norm::F8E4M3,
        ("norm", DType::F8E5M2) => norm::F8E5M2,
        ("norm", DType::BF16) => norm::BF16,
        ("norm", DType::F16) => norm::F16,
        ("norm", DType::F32) => norm::F32,
        ("norm", DType::F64) => norm::F64,

        ("argmax", DType::BOOL) => argmax::BOOL,
        ("argmax", DType::F8E4M3) => argmax::F8E4M3,
        ("argmax", DType::F8E5M2) => argmax::F8E5M2,
        ("argmax", DType::BF16) => argmax::BF16,
        ("argmax", DType::F16) => argmax::F16,
        ("argmax", DType::F32) => argmax::F32,
        ("argmax", DType::F64) => argmax::F64,
        ("argmax", DType::U8) => argmax::U8,
        ("argmax", DType::U16) => argmax::U16,
        ("argmax", DType::U32) => argmax::U32,
        ("argmax", DType::U64) => argmax::U64,
        ("argmax", DType::I8) => argmax::I8,
        ("argmax", DType::I16) => argmax::I16,
        ("argmax", DType::I32) => argmax::I32,
        ("argmax", DType::I64) => argmax::I64,

        ("argmin", DType::BOOL) => argmin::BOOL,
        ("argmin", DType::F8E4M3) => argmin::F8E4M3,
        ("argmin", DType::F8E5M2) => argmin::F8E5M2,
        ("argmin", DType::BF16) => argmin::BF16,
        ("argmin", DType::F16) => argmin::F16,
        ("argmin", DType::F32) => argmin::F32,
        ("argmin", DType::F64) => argmin::F64,
        ("argmin", DType::U8) => argmin::U8,
        ("argmin", DType::U16) => argmin::U16,
        ("argmin", DType::U32) => argmin::U32,
        ("argmin", DType::U64) => argmin::U64,
        ("argmin", DType::I8) => argmin::I8,
        ("argmin", DType::I16) => argmin::I16,
        ("argmin", DType::I32) => argmin::I32,
        ("argmin", DType::I64) => argmin::I64,

        ("any", DType::BOOL) => any::BOOL,
        ("any", DType::F8E4M3) => any::F8E4M3,
        ("any", DType::F8E5M2) => any::F8E5M2,
        ("any", DType::BF16) => any::BF16,
        ("any", DType::F16) => any::F16,
        ("any", DType::F32) => any::F32,
        ("any", DType::F64) => any::F64,
        ("any", DType::U8) => any::U8,
        ("any", DType::U16) => any::U16,
        ("any", DType::U32) => any::U32,
        ("any", DType::U64) => any::U64,
        ("any", DType::I8) => any::I8,
        ("any", DType::I16) => any::I16,
        ("any", DType::I32) => any::I32,
        ("any", DType::I64) => any::I64,

        ("all", DType::BOOL) => all::BOOL,
        ("all", DType::F8E4M3) => all::F8E4M3,
        ("all", DType::F8E5M2) => all::F8E5M2,
        ("all", DType::BF16) => all::BF16,
        ("all", DType::F16) => all::F16,
        ("all", DType::F32) => all::F32,
        ("all", DType::F64) => all::F64,
        ("all", DType::U8) => all::U8,
        ("all", DType::U16) => all::U16,
        ("all", DType::U32) => all::U32,
        ("all", DType::U64) => all::U64,
        ("all", DType::I8) => all::I8,
        ("all", DType::I16) => all::I16,
        ("all", DType::I32) => all::I32,
        ("all", DType::I64) => all::I64,

        _ => panic!("Unsupported reduce kernel: {} for {:?}", op_name, dtype),
    }
}
