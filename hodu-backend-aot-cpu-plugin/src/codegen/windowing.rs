//! Windowing operation codegen

use super::{dtype_suffix, write_metadata};
use hodu_plugin_sdk::{op_params::OpParams, ops::WindowingOp, snapshot::SnapshotNode, PluginResult};
use std::fmt::Write;

pub fn write(code: &mut String, node: &SnapshotNode, idx: usize, op: WindowingOp) -> PluginResult<()> {
    let name = match op {
        WindowingOp::ReduceWindowMax => "reduce_window_max",
        WindowingOp::ReduceWindowMean => "reduce_window_mean",
        WindowingOp::ReduceWindowSum => "reduce_window_sum",
        WindowingOp::ReduceWindowMin => "reduce_window_min",
    };

    let suffix = dtype_suffix(node.output_dtype);
    let inp = &node.input_layouts[0];
    let out = &node.output_layout;
    let num_dims = inp.ndim();

    let (window_shape, strides, padding) = match &node.params {
        Some(OpParams::ReduceWindow(p)) => {
            let pad: Vec<usize> = p.padding.iter().flat_map(|(lo, hi)| vec![*lo, *hi]).collect();
            (p.window_shape.clone(), p.strides.clone(), pad)
        },
        _ => (vec![1; num_dims], vec![1; num_dims], vec![0; 2 * num_dims]),
    };

    let mut meta = vec![out.shape().size(), num_dims];
    meta.extend_from_slice(inp.shape().dims());
    meta.extend_from_slice(inp.strides());
    meta.push(inp.offset());
    meta.extend(&window_shape);
    meta.extend(&strides);
    meta.extend(&padding);
    meta.extend_from_slice(out.shape().dims());

    write_metadata(code, &format!("m{}", idx), &meta);
    writeln!(
        code,
        "    hodu_cpu_{}_{}(t[{}], t[{}], m{});",
        name, suffix, node.input_ids[0].0, node.output_id.0, idx
    )
    .unwrap();
    Ok(())
}
