//! Windowing operation executors

use super::{get_tensor, TensorStorage};
use hodu_cli_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, PluginError, PluginResult};
use hodu_core::types::DType;
use hodu_cpu_kernels::{call_ops_reduce_window, Kernel};
use std::collections::HashMap;

pub fn execute_windowing(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::WindowingOp,
    node: &SnapshotNode,
) -> PluginResult<()> {
    use ops::WindowingOp;

    let kernel = match op {
        WindowingOp::ReduceWindowMax => get_reduce_window_max_kernel,
        WindowingOp::ReduceWindowMean => get_reduce_window_mean_kernel,
        WindowingOp::ReduceWindowSum => get_reduce_window_sum_kernel,
        WindowingOp::ReduceWindowMin => get_reduce_window_min_kernel,
    };

    execute_reduce_window(tensors, node, kernel)
}

fn execute_reduce_window(
    tensors: &mut HashMap<usize, TensorStorage>,
    node: &SnapshotNode,
    get_kernel: fn(DType) -> Kernel,
) -> PluginResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let input_layout = &node.input_layouts[0];
    let out_layout = &node.output_layout;
    let out_shape = out_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = get_kernel(input_dtype);

    // Build metadata
    let metadata = build_reduce_window_metadata(input_layout, out_layout, &node.params);

    call_ops_reduce_window(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| PluginError::Execution(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn build_reduce_window_metadata(
    input_layout: &hodu_cli_plugin_sdk::Layout,
    output_layout: &hodu_cli_plugin_sdk::Layout,
    params: &Option<OpParams>,
) -> Vec<usize> {
    let num_dims = input_layout.ndim();
    let output_size = output_layout.shape().size();

    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(num_dims);

    // input_shape
    metadata.extend_from_slice(input_layout.shape().dims());
    // input_strides
    metadata.extend_from_slice(input_layout.strides());
    // offset
    metadata.push(input_layout.offset());

    // Extract window_shape, strides, padding from params
    let (window_shape, strides, padding) = match params {
        Some(OpParams::ReduceWindow(p)) => {
            let ws = p.window_shape.clone();
            let st = p.strides.clone();
            // padding is Vec<(usize, usize)>, flatten to Vec<usize>
            let pad: Vec<usize> = p.padding.iter().flat_map(|(lo, hi)| vec![*lo, *hi]).collect();
            (ws, st, pad)
        },
        _ => (vec![], vec![], vec![]),
    };

    // window_shape
    if window_shape.len() == num_dims {
        metadata.extend(window_shape);
    } else {
        metadata.extend(vec![1usize; num_dims]);
    }

    // strides
    if strides.len() == num_dims {
        metadata.extend(strides);
    } else {
        metadata.extend(vec![1usize; num_dims]);
    }

    // padding (before and after for each dimension)
    if padding.len() == 2 * num_dims {
        metadata.extend(padding);
    } else {
        metadata.extend(vec![0usize; 2 * num_dims]);
    }

    // output_shape
    metadata.extend_from_slice(output_layout.shape().dims());

    metadata
}

fn get_reduce_window_max_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::reduce_window_max;
    match dtype {
        DType::F8E4M3 => reduce_window_max::F8E4M3,
        DType::F8E5M2 => reduce_window_max::F8E5M2,
        DType::BF16 => reduce_window_max::BF16,
        DType::F16 => reduce_window_max::F16,
        DType::F32 => reduce_window_max::F32,
        DType::F64 => reduce_window_max::F64,
        DType::U8 => reduce_window_max::U8,
        DType::U16 => reduce_window_max::U16,
        DType::U32 => reduce_window_max::U32,
        DType::U64 => reduce_window_max::U64,
        DType::I8 => reduce_window_max::I8,
        DType::I16 => reduce_window_max::I16,
        DType::I32 => reduce_window_max::I32,
        DType::I64 => reduce_window_max::I64,
        _ => panic!("Unsupported dtype for reduce_window_max: {:?}", dtype),
    }
}

fn get_reduce_window_mean_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::reduce_window_mean;
    match dtype {
        DType::F8E4M3 => reduce_window_mean::F8E4M3,
        DType::F8E5M2 => reduce_window_mean::F8E5M2,
        DType::BF16 => reduce_window_mean::BF16,
        DType::F16 => reduce_window_mean::F16,
        DType::F32 => reduce_window_mean::F32,
        DType::F64 => reduce_window_mean::F64,
        _ => panic!("Unsupported dtype for reduce_window_mean: {:?}", dtype),
    }
}

fn get_reduce_window_sum_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::reduce_window_sum;
    match dtype {
        DType::F8E4M3 => reduce_window_sum::F8E4M3,
        DType::F8E5M2 => reduce_window_sum::F8E5M2,
        DType::BF16 => reduce_window_sum::BF16,
        DType::F16 => reduce_window_sum::F16,
        DType::F32 => reduce_window_sum::F32,
        DType::F64 => reduce_window_sum::F64,
        DType::U8 => reduce_window_sum::U8,
        DType::U16 => reduce_window_sum::U16,
        DType::U32 => reduce_window_sum::U32,
        DType::U64 => reduce_window_sum::U64,
        DType::I8 => reduce_window_sum::I8,
        DType::I16 => reduce_window_sum::I16,
        DType::I32 => reduce_window_sum::I32,
        DType::I64 => reduce_window_sum::I64,
        _ => panic!("Unsupported dtype for reduce_window_sum: {:?}", dtype),
    }
}

fn get_reduce_window_min_kernel(dtype: DType) -> Kernel {
    use hodu_cpu_kernels::reduce_window_min;
    match dtype {
        DType::F8E4M3 => reduce_window_min::F8E4M3,
        DType::F8E5M2 => reduce_window_min::F8E5M2,
        DType::BF16 => reduce_window_min::BF16,
        DType::F16 => reduce_window_min::F16,
        DType::F32 => reduce_window_min::F32,
        DType::F64 => reduce_window_min::F64,
        DType::U8 => reduce_window_min::U8,
        DType::U16 => reduce_window_min::U16,
        DType::U32 => reduce_window_min::U32,
        DType::U64 => reduce_window_min::U64,
        DType::I8 => reduce_window_min::I8,
        DType::I16 => reduce_window_min::I16,
        DType::I32 => reduce_window_min::I32,
        DType::I64 => reduce_window_min::I64,
        _ => panic!("Unsupported dtype for reduce_window_min: {:?}", dtype),
    }
}
