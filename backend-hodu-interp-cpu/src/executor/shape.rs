//! Shape operation executors

use super::{build_unary_metadata, get_tensor, TensorStorage};
use hodu_cli_plugin_sdk::{ops, snapshot::SnapshotNode, PluginError, PluginResult};
use hodu_core::types::DType;
use hodu_cpu_kernels::{call_ops_contiguous, Kernel};
use std::collections::HashMap;

pub fn execute_shape(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::ShapeOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    use ops::ShapeOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;

    match op {
        ShapeOp::Reshape | ShapeOp::Flatten | ShapeOp::Squeeze | ShapeOp::Unsqueeze | ShapeOp::Broadcast => {
            // These are view operations - just copy with new shape
            let out_shape = node.output_layout.shape().dims().to_vec();
            let output = TensorStorage::from_data(input.data.clone(), out_shape, node.output_dtype.into());
            tensors.insert(out_id, output);
        },
        ShapeOp::Transpose | ShapeOp::Permute => {
            // These change memory layout - use contiguous kernel to copy
            execute_contiguous_copy(tensors, node)?;
        },
    }

    Ok(())
}

pub fn execute_shape_scalars(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::ShapeScalarsOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    use ops::ShapeScalarsOp;

    match op {
        ShapeScalarsOp::Slice => execute_slice(tensors, node),
    }
}

fn execute_slice(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> PluginResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    // Use contiguous copy for slice
    let kernel = get_contiguous_kernel(input_dtype);
    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_contiguous(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_contiguous_copy(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> PluginResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_contiguous_kernel(input_dtype);
    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_contiguous(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_contiguous_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::contiguous;
    match dtype {
        DType::BOOL => contiguous::BOOL,
        DType::F8E4M3 => contiguous::F8E4M3,
        DType::F8E5M2 => contiguous::F8E5M2,
        DType::BF16 => contiguous::BF16,
        DType::F16 => contiguous::F16,
        DType::F32 => contiguous::F32,
        DType::F64 => contiguous::F64,
        DType::U8 => contiguous::U8,
        DType::U16 => contiguous::U16,
        DType::U32 => contiguous::U32,
        DType::U64 => contiguous::U64,
        DType::I8 => contiguous::I8,
        DType::I16 => contiguous::I16,
        DType::I32 => contiguous::I32,
        DType::I64 => contiguous::I64,
    }
}
