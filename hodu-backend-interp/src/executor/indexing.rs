//! Indexing operation executors

use super::{get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_gather, call_ops_index_select, call_ops_scatter, Kernel};
use hodu_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_indexing(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::IndexingOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::IndexingOp;

    match op {
        IndexingOp::IndexSelect => execute_index_select(tensors, node),
        IndexingOp::IndexPut => execute_index_put(tensors, node),
        IndexingOp::Gather => execute_gather(tensors, node),
        IndexingOp::Scatter => execute_scatter(tensors, node, "scatter"),
        IndexingOp::ScatterAdd => execute_scatter(tensors, node, "scatter_add"),
        IndexingOp::ScatterMax => execute_scatter(tensors, node, "scatter_max"),
        IndexingOp::ScatterMin => execute_scatter(tensors, node, "scatter_min"),
    }
}

fn get_dim_from_params(node: &SnapshotNode, num_dims: usize) -> usize {
    let d = match &node.params {
        Some(OpParams::IndexSelect(p)) => p.dim.to_i64(),
        Some(OpParams::IndexPut(p)) => p.dim.to_i64(),
        Some(OpParams::Gather(p)) => p.dim.to_i64(),
        Some(OpParams::Scatter(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterAdd(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterMax(p)) => p.dim.to_i64(),
        Some(OpParams::ScatterMin(p)) => p.dim.to_i64(),
        _ => 0,
    };
    if d < 0 {
        (num_dims as i64 + d) as usize
    } else {
        d as usize
    }
}

fn execute_index_select(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let indices_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let indices = get_tensor(tensors, indices_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let indices_ptr = indices.as_ptr() as *const i32;

    let input_layout = &node.input_layouts[0];
    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_index_select_kernel(input_dtype);

    let num_dims = input_layout.ndim();
    let dim = get_dim_from_params(node, num_dims);
    let num_indices = indices.shape.iter().product::<usize>();

    let mut metadata = Vec::new();
    metadata.push(node.output_layout.shape().size());
    metadata.push(num_dims);
    metadata.extend_from_slice(input_layout.shape().dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.push(input_layout.offset());
    metadata.push(dim);
    metadata.push(num_indices);

    call_ops_index_select(kernel, input_ptr, indices_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_index_put(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let indices_id = node.input_ids[1].0;
    let values_id = node.input_ids[2].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let indices = get_tensor(tensors, indices_id)?;
    let values = get_tensor(tensors, values_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let indices_ptr = indices.as_ptr() as *const i32;
    let values_ptr = values.as_ptr();

    let input_layout = &node.input_layouts[0];
    let values_layout = &node.input_layouts[2];
    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_index_put_kernel(input_dtype);

    let num_dims = input_layout.ndim();
    let dim = get_dim_from_params(node, num_dims);
    let num_indices = indices.shape.iter().product::<usize>();

    let mut metadata = Vec::new();
    metadata.push(node.output_layout.shape().size());
    metadata.push(num_dims);
    metadata.extend_from_slice(input_layout.shape().dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.extend_from_slice(values_layout.strides());
    metadata.push(input_layout.offset());
    metadata.push(values_layout.offset());
    metadata.push(dim);
    metadata.push(num_indices);

    hodu_cpu_kernels::call_ops_index_put(
        kernel,
        input_ptr,
        indices_ptr,
        values_ptr,
        output.as_mut_ptr(),
        &metadata,
    )
    .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_gather(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let indices_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let indices = get_tensor(tensors, indices_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let indices_ptr = indices.as_ptr() as *const i32;

    let input_layout = &node.input_layouts[0];
    let indices_layout = &node.input_layouts[1];
    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_gather_kernel(input_dtype);

    let num_dims = input_layout.ndim();
    let dim = get_dim_from_params(node, num_dims);
    let indices_len = indices.shape.iter().product::<usize>();

    let mut metadata = Vec::new();
    metadata.push(node.output_layout.shape().size());
    metadata.push(num_dims);
    metadata.extend_from_slice(input_layout.shape().dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.extend_from_slice(indices_layout.strides());
    metadata.push(input_layout.offset());
    metadata.push(indices_layout.offset());
    metadata.push(dim);
    metadata.push(indices_len);

    call_ops_gather(kernel, input_ptr, indices_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn execute_scatter(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode, op_name: &str) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let indices_id = node.input_ids[1].0;
    let src_id = node.input_ids[2].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let indices = get_tensor(tensors, indices_id)?;
    let src = get_tensor(tensors, src_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();
    let indices_ptr = indices.as_ptr() as *const i32;
    let src_ptr = src.as_ptr();

    let input_layout = &node.input_layouts[0];
    let src_layout = &node.input_layouts[2];
    let indices_layout = &node.input_layouts[1];
    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_scatter_kernel(op_name, input_dtype);

    let num_dims = input_layout.ndim();
    let dim = get_dim_from_params(node, num_dims);

    let mut metadata = Vec::new();
    metadata.push(src_layout.shape().size());
    metadata.push(num_dims);
    metadata.extend_from_slice(input_layout.shape().dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.extend_from_slice(src_layout.shape().dims());
    metadata.extend_from_slice(src_layout.strides());
    metadata.extend_from_slice(indices_layout.strides());
    metadata.push(input_layout.offset());
    metadata.push(src_layout.offset());
    metadata.push(indices_layout.offset());
    metadata.push(dim);

    call_ops_scatter(kernel, input_ptr, indices_ptr, src_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_index_select_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::index_select;
    match dtype {
        DType::BOOL => index_select::BOOL,
        DType::F8E4M3 => index_select::F8E4M3,
        DType::F8E5M2 => index_select::F8E5M2,
        DType::BF16 => index_select::BF16,
        DType::F16 => index_select::F16,
        DType::F32 => index_select::F32,
        DType::F64 => index_select::F64,
        DType::U8 => index_select::U8,
        DType::U16 => index_select::U16,
        DType::U32 => index_select::U32,
        DType::U64 => index_select::U64,
        DType::I8 => index_select::I8,
        DType::I16 => index_select::I16,
        DType::I32 => index_select::I32,
        DType::I64 => index_select::I64,
    }
}

fn get_index_put_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::index_put;
    match dtype {
        DType::BOOL => index_put::BOOL,
        DType::F8E4M3 => index_put::F8E4M3,
        DType::F8E5M2 => index_put::F8E5M2,
        DType::BF16 => index_put::BF16,
        DType::F16 => index_put::F16,
        DType::F32 => index_put::F32,
        DType::F64 => index_put::F64,
        DType::U8 => index_put::U8,
        DType::U16 => index_put::U16,
        DType::U32 => index_put::U32,
        DType::U64 => index_put::U64,
        DType::I8 => index_put::I8,
        DType::I16 => index_put::I16,
        DType::I32 => index_put::I32,
        DType::I64 => index_put::I64,
    }
}

fn get_gather_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::gather;
    match dtype {
        DType::BOOL => gather::BOOL,
        DType::F8E4M3 => gather::F8E4M3,
        DType::F8E5M2 => gather::F8E5M2,
        DType::BF16 => gather::BF16,
        DType::F16 => gather::F16,
        DType::F32 => gather::F32,
        DType::F64 => gather::F64,
        DType::U8 => gather::U8,
        DType::U16 => gather::U16,
        DType::U32 => gather::U32,
        DType::U64 => gather::U64,
        DType::I8 => gather::I8,
        DType::I16 => gather::I16,
        DType::I32 => gather::I32,
        DType::I64 => gather::I64,
    }
}

fn get_scatter_kernel(op_name: &str, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::{scatter, scatter_add, scatter_max, scatter_min};
    match (op_name, dtype) {
        ("scatter", DType::BOOL) => scatter::BOOL,
        ("scatter", DType::F8E4M3) => scatter::F8E4M3,
        ("scatter", DType::F8E5M2) => scatter::F8E5M2,
        ("scatter", DType::BF16) => scatter::BF16,
        ("scatter", DType::F16) => scatter::F16,
        ("scatter", DType::F32) => scatter::F32,
        ("scatter", DType::F64) => scatter::F64,
        ("scatter", DType::U8) => scatter::U8,
        ("scatter", DType::U16) => scatter::U16,
        ("scatter", DType::U32) => scatter::U32,
        ("scatter", DType::U64) => scatter::U64,
        ("scatter", DType::I8) => scatter::I8,
        ("scatter", DType::I16) => scatter::I16,
        ("scatter", DType::I32) => scatter::I32,
        ("scatter", DType::I64) => scatter::I64,

        ("scatter_add", DType::F8E4M3) => scatter_add::F8E4M3,
        ("scatter_add", DType::F8E5M2) => scatter_add::F8E5M2,
        ("scatter_add", DType::BF16) => scatter_add::BF16,
        ("scatter_add", DType::F16) => scatter_add::F16,
        ("scatter_add", DType::F32) => scatter_add::F32,
        ("scatter_add", DType::U8) => scatter_add::U8,
        ("scatter_add", DType::U16) => scatter_add::U16,
        ("scatter_add", DType::U32) => scatter_add::U32,
        ("scatter_add", DType::U64) => scatter_add::U64,
        ("scatter_add", DType::I8) => scatter_add::I8,
        ("scatter_add", DType::I16) => scatter_add::I16,
        ("scatter_add", DType::I32) => scatter_add::I32,
        ("scatter_add", DType::I64) => scatter_add::I64,

        ("scatter_max", DType::F8E4M3) => scatter_max::F8E4M3,
        ("scatter_max", DType::F8E5M2) => scatter_max::F8E5M2,
        ("scatter_max", DType::BF16) => scatter_max::BF16,
        ("scatter_max", DType::F16) => scatter_max::F16,
        ("scatter_max", DType::F32) => scatter_max::F32,
        ("scatter_max", DType::U8) => scatter_max::U8,
        ("scatter_max", DType::U16) => scatter_max::U16,
        ("scatter_max", DType::U32) => scatter_max::U32,
        ("scatter_max", DType::U64) => scatter_max::U64,
        ("scatter_max", DType::I8) => scatter_max::I8,
        ("scatter_max", DType::I16) => scatter_max::I16,
        ("scatter_max", DType::I32) => scatter_max::I32,
        ("scatter_max", DType::I64) => scatter_max::I64,

        ("scatter_min", DType::F8E4M3) => scatter_min::F8E4M3,
        ("scatter_min", DType::F8E5M2) => scatter_min::F8E5M2,
        ("scatter_min", DType::BF16) => scatter_min::BF16,
        ("scatter_min", DType::F16) => scatter_min::F16,
        ("scatter_min", DType::F32) => scatter_min::F32,
        ("scatter_min", DType::U8) => scatter_min::U8,
        ("scatter_min", DType::U16) => scatter_min::U16,
        ("scatter_min", DType::U32) => scatter_min::U32,
        ("scatter_min", DType::U64) => scatter_min::U64,
        ("scatter_min", DType::I8) => scatter_min::I8,
        ("scatter_min", DType::I16) => scatter_min::I16,
        ("scatter_min", DType::I32) => scatter_min::I32,
        ("scatter_min", DType::I64) => scatter_min::I64,

        _ => panic!("Unsupported scatter kernel: {} for {:?}", op_name, dtype),
    }
}
