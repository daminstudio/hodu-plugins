//! Concat and Split operation executors

use super::{get_tensor, TensorStorage};
use hodu_cli_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, PluginError, PluginResult};
use hodu_core::types::DType;
use hodu_cpu_kernels::{call_ops_concat, call_ops_split, Kernel};
use std::collections::HashMap;

pub fn execute_concat(
    tensors: &mut HashMap<usize, TensorStorage>,
    _op: ops::ConcatOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    let out_id = node.output_id.0;
    let out_layout = &node.output_layout;
    let out_shape = out_layout.shape().dims().to_vec();
    let num_dims = out_layout.ndim();

    // Get first input to determine dtype
    let first_input = get_tensor(tensors, node.input_ids[0].0)?;
    let input_dtype = first_input.dtype;

    // Get concat dimension from params
    let concat_dim = if let Some(OpParams::Concat(params)) = &node.params {
        let d = params.dim.to_i64();
        if d < 0 {
            (num_dims as i64 + d) as usize
        } else {
            d as usize
        }
    } else {
        0
    };

    let num_inputs = node.input_ids.len();

    // Collect input data pointers and build combined input buffer
    let mut input_buffer = Vec::new();
    let mut input_buffer_offsets = Vec::new();

    for i in 0..num_inputs {
        let input = get_tensor(tensors, node.input_ids[i].0)?;
        input_buffer_offsets.push(input_buffer.len());
        input_buffer.extend_from_slice(&input.data);
    }

    let input_ptr = input_buffer.as_ptr() as *const std::ffi::c_void;

    let mut output = TensorStorage::new(&out_shape, node.output_dtype);
    let kernel = get_concat_kernel(input_dtype);

    // Build metadata
    let mut metadata = Vec::new();
    metadata.push(out_layout.shape().size()); // num_els
    metadata.push(num_dims);

    // output_shape
    metadata.extend_from_slice(out_layout.shape().dims());

    // concat_dim
    metadata.push(concat_dim);

    // num_inputs
    metadata.push(num_inputs);

    // input_shapes (flattened)
    for i in 0..num_inputs {
        let layout = &node.input_layouts[i];
        metadata.extend_from_slice(layout.shape().dims());
    }

    // input_strides (flattened)
    for i in 0..num_inputs {
        let layout = &node.input_layouts[i];
        metadata.extend_from_slice(layout.strides());
    }

    // input_offsets
    for i in 0..num_inputs {
        let layout = &node.input_layouts[i];
        metadata.push(layout.offset());
    }

    // input_buffer_offsets (byte offsets divided by element size)
    let elem_size = input_dtype.get_size_in_bytes();
    for offset in &input_buffer_offsets {
        metadata.push(offset / elem_size);
    }

    call_ops_concat(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

pub fn execute_split(
    tensors: &mut HashMap<usize, TensorStorage>,
    _op: ops::SplitOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let input_layout = &node.input_layouts[0];
    let out_layout = &node.output_layout;
    let out_shape = out_layout.shape().dims().to_vec();
    let num_dims = input_layout.ndim();

    let mut output = TensorStorage::new(&out_shape, node.output_dtype);
    let kernel = get_split_kernel(input_dtype);

    // Get split params
    let (split_dim, split_offset) = if let Some(OpParams::Split(params)) = &node.params {
        let d = params.dim.to_i64();
        let dim = if d < 0 {
            (num_dims as i64 + d) as usize
        } else {
            d as usize
        };
        // Calculate offset from sizes and output_index
        let mut offset = 0usize;
        for i in 0..params.output_index {
            if i < params.sizes.len() {
                offset += params.sizes[i].to_usize();
            }
        }
        (dim, offset)
    } else {
        (0, 0)
    };

    let output_size_on_dim = out_layout.shape().dims()[split_dim];

    // Build metadata
    let mut metadata = Vec::new();
    metadata.push(out_layout.shape().size()); // num_els
    metadata.push(num_dims);

    // input_shape
    metadata.extend_from_slice(input_layout.shape().dims());
    // input_strides
    metadata.extend_from_slice(input_layout.strides());
    // input_offset
    metadata.push(input_layout.offset());
    // split_dim
    metadata.push(split_dim);
    // output_size_on_dim
    metadata.push(output_size_on_dim);
    // split_offset
    metadata.push(split_offset);

    call_ops_split(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_concat_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::concat;
    match dtype {
        DType::BOOL => concat::BOOL,
        DType::F8E4M3 => concat::F8E4M3,
        DType::F8E5M2 => concat::F8E5M2,
        DType::BF16 => concat::BF16,
        DType::F16 => concat::F16,
        DType::F32 => concat::F32,
        DType::F64 => concat::F64,
        DType::U8 => concat::U8,
        DType::U16 => concat::U16,
        DType::U32 => concat::U32,
        DType::U64 => concat::U64,
        DType::I8 => concat::I8,
        DType::I16 => concat::I16,
        DType::I32 => concat::I32,
        DType::I64 => concat::I64,
    }
}

fn get_split_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::split;
    match dtype {
        DType::BOOL => split::BOOL,
        DType::F8E4M3 => split::F8E4M3,
        DType::F8E5M2 => split::F8E5M2,
        DType::BF16 => split::BF16,
        DType::F16 => split::F16,
        DType::F32 => split::F32,
        DType::F64 => split::F64,
        DType::U8 => split::U8,
        DType::U16 => split::U16,
        DType::U32 => split::U32,
        DType::U64 => split::U64,
        DType::I8 => split::I8,
        DType::I16 => split::I16,
        DType::I32 => split::I32,
        DType::I64 => split::I64,
    }
}
