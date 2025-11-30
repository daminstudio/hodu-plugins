//! Memory operation executors

use super::{build_unary_metadata, get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_contiguous, Kernel};
use hodu_plugin_sdk::{ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_memory(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::MemoryOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::MemoryOp;

    match op {
        MemoryOp::Contiguous => execute_contiguous(tensors, node),
    }
}

fn execute_contiguous(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_kernel(input_dtype);
    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_contiguous(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_kernel(dtype: DType) -> Kernel {
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
