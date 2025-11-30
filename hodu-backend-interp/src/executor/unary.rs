//! Unary operation executors

use super::{build_unary_metadata, get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_unary, call_ops_unary_scalar, Kernel};
use hodu_plugin_sdk::{op_params::OpParams, ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_unary(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::UnaryOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::UnaryOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        UnaryOp::Neg => get_kernel("neg", input_dtype),
        UnaryOp::Abs => get_kernel("abs", input_dtype),
        UnaryOp::Sign => get_kernel("sign", input_dtype),
        UnaryOp::Square => get_kernel("square", input_dtype),
        UnaryOp::Sqrt => get_kernel("sqrt", input_dtype),
        UnaryOp::Recip => get_kernel("recip", input_dtype),
        UnaryOp::Relu => get_kernel("relu", input_dtype),
        UnaryOp::Sigmoid => get_kernel("sigmoid", input_dtype),
        UnaryOp::Tanh => get_kernel("tanh", input_dtype),
        UnaryOp::Gelu => get_kernel("gelu", input_dtype),
        UnaryOp::Softplus => get_kernel("softplus", input_dtype),
        UnaryOp::Silu => get_kernel("silu", input_dtype),
        UnaryOp::Mish => get_kernel("mish", input_dtype),
        UnaryOp::Sin => get_kernel("sin", input_dtype),
        UnaryOp::Cos => get_kernel("cos", input_dtype),
        UnaryOp::Tan => get_kernel("tan", input_dtype),
        UnaryOp::Exp => get_kernel("exp", input_dtype),
        UnaryOp::Exp2 => get_kernel("exp2", input_dtype),
        UnaryOp::Exp10 => get_kernel("exp10", input_dtype),
        UnaryOp::Ln => get_kernel("ln", input_dtype),
        UnaryOp::Log2 => get_kernel("log2", input_dtype),
        UnaryOp::Log10 => get_kernel("log10", input_dtype),
    };

    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_unary(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

pub fn execute_unary_logical(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::UnaryLogicalOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::UnaryLogicalOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        UnaryLogicalOp::LogicalNot => get_kernel("logical_not", input_dtype),
    };

    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_unary(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_kernel(op_name: &str, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (op_name, dtype) {
        ("neg", DType::BOOL) => neg::BOOL,
        ("neg", DType::F8E4M3) => neg::F8E4M3,
        ("neg", DType::F8E5M2) => neg::F8E5M2,
        ("neg", DType::BF16) => neg::BF16,
        ("neg", DType::F16) => neg::F16,
        ("neg", DType::F32) => neg::F32,
        ("neg", DType::F64) => neg::F64,
        ("neg", DType::I8) => neg::I8,
        ("neg", DType::I16) => neg::I16,
        ("neg", DType::I32) => neg::I32,
        ("neg", DType::I64) => neg::I64,

        ("abs", DType::BOOL) => abs::BOOL,
        ("abs", DType::F8E4M3) => abs::F8E4M3,
        ("abs", DType::F8E5M2) => abs::F8E5M2,
        ("abs", DType::BF16) => abs::BF16,
        ("abs", DType::F16) => abs::F16,
        ("abs", DType::F32) => abs::F32,
        ("abs", DType::F64) => abs::F64,
        ("abs", DType::U8) => abs::U8,
        ("abs", DType::U16) => abs::U16,
        ("abs", DType::U32) => abs::U32,
        ("abs", DType::U64) => abs::U64,
        ("abs", DType::I8) => abs::I8,
        ("abs", DType::I16) => abs::I16,
        ("abs", DType::I32) => abs::I32,
        ("abs", DType::I64) => abs::I64,

        ("sign", DType::BOOL) => sign::BOOL,
        ("sign", DType::F8E4M3) => sign::F8E4M3,
        ("sign", DType::F8E5M2) => sign::F8E5M2,
        ("sign", DType::BF16) => sign::BF16,
        ("sign", DType::F16) => sign::F16,
        ("sign", DType::F32) => sign::F32,
        ("sign", DType::F64) => sign::F64,
        ("sign", DType::U8) => sign::U8,
        ("sign", DType::U16) => sign::U16,
        ("sign", DType::U32) => sign::U32,
        ("sign", DType::U64) => sign::U64,
        ("sign", DType::I8) => sign::I8,
        ("sign", DType::I16) => sign::I16,
        ("sign", DType::I32) => sign::I32,
        ("sign", DType::I64) => sign::I64,

        ("square", DType::BOOL) => square::BOOL,
        ("square", DType::F8E4M3) => square::F8E4M3,
        ("square", DType::F8E5M2) => square::F8E5M2,
        ("square", DType::BF16) => square::BF16,
        ("square", DType::F16) => square::F16,
        ("square", DType::F32) => square::F32,
        ("square", DType::F64) => square::F64,
        ("square", DType::U8) => square::U8,
        ("square", DType::U16) => square::U16,
        ("square", DType::U32) => square::U32,
        ("square", DType::U64) => square::U64,
        ("square", DType::I8) => square::I8,
        ("square", DType::I16) => square::I16,
        ("square", DType::I32) => square::I32,
        ("square", DType::I64) => square::I64,

        ("sqrt", DType::BOOL) => sqrt::BOOL,
        ("sqrt", DType::F8E4M3) => sqrt::F8E4M3,
        ("sqrt", DType::F8E5M2) => sqrt::F8E5M2,
        ("sqrt", DType::BF16) => sqrt::BF16,
        ("sqrt", DType::F16) => sqrt::F16,
        ("sqrt", DType::F32) => sqrt::F32,
        ("sqrt", DType::F64) => sqrt::F64,
        ("sqrt", DType::U8) => sqrt::U8,
        ("sqrt", DType::U16) => sqrt::U16,
        ("sqrt", DType::U32) => sqrt::U32,
        ("sqrt", DType::U64) => sqrt::U64,
        ("sqrt", DType::I8) => sqrt::I8,
        ("sqrt", DType::I16) => sqrt::I16,
        ("sqrt", DType::I32) => sqrt::I32,
        ("sqrt", DType::I64) => sqrt::I64,

        ("recip", DType::BOOL) => recip::BOOL,
        ("recip", DType::F8E4M3) => recip::F8E4M3,
        ("recip", DType::F8E5M2) => recip::F8E5M2,
        ("recip", DType::BF16) => recip::BF16,
        ("recip", DType::F16) => recip::F16,
        ("recip", DType::F32) => recip::F32,
        ("recip", DType::F64) => recip::F64,
        ("recip", DType::U8) => recip::U8,
        ("recip", DType::U16) => recip::U16,
        ("recip", DType::U32) => recip::U32,
        ("recip", DType::U64) => recip::U64,
        ("recip", DType::I8) => recip::I8,
        ("recip", DType::I16) => recip::I16,
        ("recip", DType::I32) => recip::I32,
        ("recip", DType::I64) => recip::I64,

        ("relu", DType::BOOL) => relu::BOOL,
        ("relu", DType::F8E4M3) => relu::F8E4M3,
        ("relu", DType::F8E5M2) => relu::F8E5M2,
        ("relu", DType::BF16) => relu::BF16,
        ("relu", DType::F16) => relu::F16,
        ("relu", DType::F32) => relu::F32,
        ("relu", DType::F64) => relu::F64,

        ("sigmoid", DType::BOOL) => sigmoid::BOOL,
        ("sigmoid", DType::F8E4M3) => sigmoid::F8E4M3,
        ("sigmoid", DType::F8E5M2) => sigmoid::F8E5M2,
        ("sigmoid", DType::BF16) => sigmoid::BF16,
        ("sigmoid", DType::F16) => sigmoid::F16,
        ("sigmoid", DType::F32) => sigmoid::F32,
        ("sigmoid", DType::F64) => sigmoid::F64,

        ("tanh", DType::BOOL) => tanh::BOOL,
        ("tanh", DType::F8E4M3) => tanh::F8E4M3,
        ("tanh", DType::F8E5M2) => tanh::F8E5M2,
        ("tanh", DType::BF16) => tanh::BF16,
        ("tanh", DType::F16) => tanh::F16,
        ("tanh", DType::F32) => tanh::F32,
        ("tanh", DType::F64) => tanh::F64,

        ("gelu", DType::BOOL) => gelu::BOOL,
        ("gelu", DType::F8E4M3) => gelu::F8E4M3,
        ("gelu", DType::F8E5M2) => gelu::F8E5M2,
        ("gelu", DType::BF16) => gelu::BF16,
        ("gelu", DType::F16) => gelu::F16,
        ("gelu", DType::F32) => gelu::F32,
        ("gelu", DType::F64) => gelu::F64,

        ("softplus", DType::BOOL) => softplus::BOOL,
        ("softplus", DType::F8E4M3) => softplus::F8E4M3,
        ("softplus", DType::F8E5M2) => softplus::F8E5M2,
        ("softplus", DType::BF16) => softplus::BF16,
        ("softplus", DType::F16) => softplus::F16,
        ("softplus", DType::F32) => softplus::F32,
        ("softplus", DType::F64) => softplus::F64,

        ("silu", DType::BOOL) => silu::BOOL,
        ("silu", DType::F8E4M3) => silu::F8E4M3,
        ("silu", DType::F8E5M2) => silu::F8E5M2,
        ("silu", DType::BF16) => silu::BF16,
        ("silu", DType::F16) => silu::F16,
        ("silu", DType::F32) => silu::F32,
        ("silu", DType::F64) => silu::F64,

        ("mish", DType::BOOL) => mish::BOOL,
        ("mish", DType::F8E4M3) => mish::F8E4M3,
        ("mish", DType::F8E5M2) => mish::F8E5M2,
        ("mish", DType::BF16) => mish::BF16,
        ("mish", DType::F16) => mish::F16,
        ("mish", DType::F32) => mish::F32,
        ("mish", DType::F64) => mish::F64,

        ("sin", DType::BOOL) => sin::BOOL,
        ("sin", DType::F8E4M3) => sin::F8E4M3,
        ("sin", DType::F8E5M2) => sin::F8E5M2,
        ("sin", DType::BF16) => sin::BF16,
        ("sin", DType::F16) => sin::F16,
        ("sin", DType::F32) => sin::F32,
        ("sin", DType::F64) => sin::F64,

        ("cos", DType::BOOL) => cos::BOOL,
        ("cos", DType::F8E4M3) => cos::F8E4M3,
        ("cos", DType::F8E5M2) => cos::F8E5M2,
        ("cos", DType::BF16) => cos::BF16,
        ("cos", DType::F16) => cos::F16,
        ("cos", DType::F32) => cos::F32,
        ("cos", DType::F64) => cos::F64,

        ("tan", DType::BOOL) => tan::BOOL,
        ("tan", DType::F8E4M3) => tan::F8E4M3,
        ("tan", DType::F8E5M2) => tan::F8E5M2,
        ("tan", DType::BF16) => tan::BF16,
        ("tan", DType::F16) => tan::F16,
        ("tan", DType::F32) => tan::F32,
        ("tan", DType::F64) => tan::F64,

        ("exp", DType::BOOL) => exp::BOOL,
        ("exp", DType::F8E4M3) => exp::F8E4M3,
        ("exp", DType::F8E5M2) => exp::F8E5M2,
        ("exp", DType::BF16) => exp::BF16,
        ("exp", DType::F16) => exp::F16,
        ("exp", DType::F32) => exp::F32,
        ("exp", DType::F64) => exp::F64,

        ("exp2", DType::BOOL) => exp2::BOOL,
        ("exp2", DType::F8E4M3) => exp2::F8E4M3,
        ("exp2", DType::F8E5M2) => exp2::F8E5M2,
        ("exp2", DType::BF16) => exp2::BF16,
        ("exp2", DType::F16) => exp2::F16,
        ("exp2", DType::F32) => exp2::F32,
        ("exp2", DType::F64) => exp2::F64,

        ("exp10", DType::BOOL) => exp10::BOOL,
        ("exp10", DType::F8E4M3) => exp10::F8E4M3,
        ("exp10", DType::F8E5M2) => exp10::F8E5M2,
        ("exp10", DType::BF16) => exp10::BF16,
        ("exp10", DType::F16) => exp10::F16,
        ("exp10", DType::F32) => exp10::F32,
        ("exp10", DType::F64) => exp10::F64,

        ("ln", DType::BOOL) => ln::BOOL,
        ("ln", DType::F8E4M3) => ln::F8E4M3,
        ("ln", DType::F8E5M2) => ln::F8E5M2,
        ("ln", DType::BF16) => ln::BF16,
        ("ln", DType::F16) => ln::F16,
        ("ln", DType::F32) => ln::F32,
        ("ln", DType::F64) => ln::F64,

        ("log2", DType::BOOL) => log2::BOOL,
        ("log2", DType::F8E4M3) => log2::F8E4M3,
        ("log2", DType::F8E5M2) => log2::F8E5M2,
        ("log2", DType::BF16) => log2::BF16,
        ("log2", DType::F16) => log2::F16,
        ("log2", DType::F32) => log2::F32,
        ("log2", DType::F64) => log2::F64,

        ("log10", DType::BOOL) => log10::BOOL,
        ("log10", DType::F8E4M3) => log10::F8E4M3,
        ("log10", DType::F8E5M2) => log10::F8E5M2,
        ("log10", DType::BF16) => log10::BF16,
        ("log10", DType::F16) => log10::F16,
        ("log10", DType::F32) => log10::F32,
        ("log10", DType::F64) => log10::F64,

        ("logical_not", DType::BOOL) => logical_not::BOOL,
        ("logical_not", DType::F8E4M3) => logical_not::F8E4M3,
        ("logical_not", DType::F8E5M2) => logical_not::F8E5M2,
        ("logical_not", DType::BF16) => logical_not::BF16,
        ("logical_not", DType::F16) => logical_not::F16,
        ("logical_not", DType::F32) => logical_not::F32,
        ("logical_not", DType::F64) => logical_not::F64,
        ("logical_not", DType::U8) => logical_not::U8,
        ("logical_not", DType::U16) => logical_not::U16,
        ("logical_not", DType::U32) => logical_not::U32,
        ("logical_not", DType::U64) => logical_not::U64,
        ("logical_not", DType::I8) => logical_not::I8,
        ("logical_not", DType::I16) => logical_not::I16,
        ("logical_not", DType::I32) => logical_not::I32,
        ("logical_not", DType::I64) => logical_not::I64,

        _ => panic!("Unsupported unary kernel: {} for {:?}", op_name, dtype),
    }
}

pub fn execute_unary_scalar(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::UnaryScalarOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::UnaryScalarOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let (_op_name, kernel) = match op {
        UnaryScalarOp::AddScalar => ("add_scalar", get_scalar_kernel("add_scalar", input_dtype)),
        UnaryScalarOp::SubScalar => ("sub_scalar", get_scalar_kernel("sub_scalar", input_dtype)),
        UnaryScalarOp::MulScalar => ("mul_scalar", get_scalar_kernel("mul_scalar", input_dtype)),
        UnaryScalarOp::DivScalar => ("div_scalar", get_scalar_kernel("div_scalar", input_dtype)),
        UnaryScalarOp::PowScalar => ("pow_scalar", get_scalar_kernel("pow_scalar", input_dtype)),
        UnaryScalarOp::MaximumScalar => ("maximum_scalar", get_scalar_kernel("maximum_scalar", input_dtype)),
        UnaryScalarOp::MinimumScalar => ("minimum_scalar", get_scalar_kernel("minimum_scalar", input_dtype)),
        UnaryScalarOp::LeakyRelu => ("leaky_relu", get_scalar_kernel("leaky_relu", input_dtype)),
        UnaryScalarOp::Elu => ("elu", get_scalar_kernel("elu", input_dtype)),
        UnaryScalarOp::Prelu => ("prelu", get_scalar_kernel("prelu", input_dtype)),
    };

    let metadata = build_unary_metadata(&node.input_layouts[0]);

    // Get scalar value from params
    let scalar_val = match &node.params {
        Some(OpParams::UnaryScalar(p)) => p.scalar.to_f64(),
        _ => 0.0,
    };

    call_scalar_op(
        kernel,
        input_ptr,
        output.as_mut_ptr(),
        &metadata,
        scalar_val,
        input_dtype,
    )?;

    tensors.insert(out_id, output);
    Ok(())
}

pub fn execute_cmp_scalar(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::CmpScalarOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::CmpScalarOp;

    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let input_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        CmpScalarOp::EqScalar => get_scalar_kernel("eq_scalar", input_dtype),
        CmpScalarOp::NeScalar => get_scalar_kernel("ne_scalar", input_dtype),
        CmpScalarOp::LtScalar => get_scalar_kernel("lt_scalar", input_dtype),
        CmpScalarOp::LeScalar => get_scalar_kernel("le_scalar", input_dtype),
        CmpScalarOp::GtScalar => get_scalar_kernel("gt_scalar", input_dtype),
        CmpScalarOp::GeScalar => get_scalar_kernel("ge_scalar", input_dtype),
    };

    let metadata = build_unary_metadata(&node.input_layouts[0]);

    // Get scalar value from params
    let scalar_val = match &node.params {
        Some(OpParams::CmpScalar(p)) => p.scalar.to_f64(),
        _ => 0.0,
    };

    call_scalar_op(
        kernel,
        input_ptr,
        output.as_mut_ptr(),
        &metadata,
        scalar_val,
        input_dtype,
    )?;

    tensors.insert(out_id, output);
    Ok(())
}

fn call_scalar_op(
    kernel: Kernel,
    input: *const std::ffi::c_void,
    output: *mut std::ffi::c_void,
    metadata: &[usize],
    scalar_val: f64,
    dtype: DType,
) -> HoduResult<()> {
    // Call the appropriate scalar kernel based on dtype
    match dtype {
        DType::F32 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as f32)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::F64 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::I32 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as i32)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::I64 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as i64)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::U32 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as u32)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::U64 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as u64)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::I8 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as i8)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::I16 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as i16)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::U8 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as u8)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        DType::U16 => {
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as u16)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
        _ => {
            // For other types (BF16, F16, F8), use f32 as intermediate
            call_ops_unary_scalar(kernel, input, output, metadata, scalar_val as f32)
                .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;
        },
    }
    Ok(())
}

fn get_scalar_kernel(op_name: &str, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (op_name, dtype) {
        ("add_scalar", DType::BOOL) => add_scalar::BOOL,
        ("add_scalar", DType::F8E4M3) => add_scalar::F8E4M3,
        ("add_scalar", DType::F8E5M2) => add_scalar::F8E5M2,
        ("add_scalar", DType::BF16) => add_scalar::BF16,
        ("add_scalar", DType::F16) => add_scalar::F16,
        ("add_scalar", DType::F32) => add_scalar::F32,
        ("add_scalar", DType::F64) => add_scalar::F64,
        ("add_scalar", DType::U8) => add_scalar::U8,
        ("add_scalar", DType::U16) => add_scalar::U16,
        ("add_scalar", DType::U32) => add_scalar::U32,
        ("add_scalar", DType::U64) => add_scalar::U64,
        ("add_scalar", DType::I8) => add_scalar::I8,
        ("add_scalar", DType::I16) => add_scalar::I16,
        ("add_scalar", DType::I32) => add_scalar::I32,
        ("add_scalar", DType::I64) => add_scalar::I64,

        ("sub_scalar", DType::BOOL) => sub_scalar::BOOL,
        ("sub_scalar", DType::F8E4M3) => sub_scalar::F8E4M3,
        ("sub_scalar", DType::F8E5M2) => sub_scalar::F8E5M2,
        ("sub_scalar", DType::BF16) => sub_scalar::BF16,
        ("sub_scalar", DType::F16) => sub_scalar::F16,
        ("sub_scalar", DType::F32) => sub_scalar::F32,
        ("sub_scalar", DType::F64) => sub_scalar::F64,
        ("sub_scalar", DType::U8) => sub_scalar::U8,
        ("sub_scalar", DType::U16) => sub_scalar::U16,
        ("sub_scalar", DType::U32) => sub_scalar::U32,
        ("sub_scalar", DType::U64) => sub_scalar::U64,
        ("sub_scalar", DType::I8) => sub_scalar::I8,
        ("sub_scalar", DType::I16) => sub_scalar::I16,
        ("sub_scalar", DType::I32) => sub_scalar::I32,
        ("sub_scalar", DType::I64) => sub_scalar::I64,

        ("mul_scalar", DType::BOOL) => mul_scalar::BOOL,
        ("mul_scalar", DType::F8E4M3) => mul_scalar::F8E4M3,
        ("mul_scalar", DType::F8E5M2) => mul_scalar::F8E5M2,
        ("mul_scalar", DType::BF16) => mul_scalar::BF16,
        ("mul_scalar", DType::F16) => mul_scalar::F16,
        ("mul_scalar", DType::F32) => mul_scalar::F32,
        ("mul_scalar", DType::F64) => mul_scalar::F64,
        ("mul_scalar", DType::U8) => mul_scalar::U8,
        ("mul_scalar", DType::U16) => mul_scalar::U16,
        ("mul_scalar", DType::U32) => mul_scalar::U32,
        ("mul_scalar", DType::U64) => mul_scalar::U64,
        ("mul_scalar", DType::I8) => mul_scalar::I8,
        ("mul_scalar", DType::I16) => mul_scalar::I16,
        ("mul_scalar", DType::I32) => mul_scalar::I32,
        ("mul_scalar", DType::I64) => mul_scalar::I64,

        ("div_scalar", DType::BOOL) => div_scalar::BOOL,
        ("div_scalar", DType::F8E4M3) => div_scalar::F8E4M3,
        ("div_scalar", DType::F8E5M2) => div_scalar::F8E5M2,
        ("div_scalar", DType::BF16) => div_scalar::BF16,
        ("div_scalar", DType::F16) => div_scalar::F16,
        ("div_scalar", DType::F32) => div_scalar::F32,
        ("div_scalar", DType::F64) => div_scalar::F64,
        ("div_scalar", DType::U8) => div_scalar::U8,
        ("div_scalar", DType::U16) => div_scalar::U16,
        ("div_scalar", DType::U32) => div_scalar::U32,
        ("div_scalar", DType::U64) => div_scalar::U64,
        ("div_scalar", DType::I8) => div_scalar::I8,
        ("div_scalar", DType::I16) => div_scalar::I16,
        ("div_scalar", DType::I32) => div_scalar::I32,
        ("div_scalar", DType::I64) => div_scalar::I64,

        ("pow_scalar", DType::BOOL) => pow_scalar::BOOL,
        ("pow_scalar", DType::F8E4M3) => pow_scalar::F8E4M3,
        ("pow_scalar", DType::F8E5M2) => pow_scalar::F8E5M2,
        ("pow_scalar", DType::BF16) => pow_scalar::BF16,
        ("pow_scalar", DType::F16) => pow_scalar::F16,
        ("pow_scalar", DType::F32) => pow_scalar::F32,
        ("pow_scalar", DType::F64) => pow_scalar::F64,
        ("pow_scalar", DType::U8) => pow_scalar::U8,
        ("pow_scalar", DType::U16) => pow_scalar::U16,
        ("pow_scalar", DType::U32) => pow_scalar::U32,
        ("pow_scalar", DType::U64) => pow_scalar::U64,
        ("pow_scalar", DType::I8) => pow_scalar::I8,
        ("pow_scalar", DType::I16) => pow_scalar::I16,
        ("pow_scalar", DType::I32) => pow_scalar::I32,
        ("pow_scalar", DType::I64) => pow_scalar::I64,

        ("maximum_scalar", DType::BOOL) => maximum_scalar::BOOL,
        ("maximum_scalar", DType::F8E4M3) => maximum_scalar::F8E4M3,
        ("maximum_scalar", DType::F8E5M2) => maximum_scalar::F8E5M2,
        ("maximum_scalar", DType::BF16) => maximum_scalar::BF16,
        ("maximum_scalar", DType::F16) => maximum_scalar::F16,
        ("maximum_scalar", DType::F32) => maximum_scalar::F32,
        ("maximum_scalar", DType::F64) => maximum_scalar::F64,
        ("maximum_scalar", DType::U8) => maximum_scalar::U8,
        ("maximum_scalar", DType::U16) => maximum_scalar::U16,
        ("maximum_scalar", DType::U32) => maximum_scalar::U32,
        ("maximum_scalar", DType::U64) => maximum_scalar::U64,
        ("maximum_scalar", DType::I8) => maximum_scalar::I8,
        ("maximum_scalar", DType::I16) => maximum_scalar::I16,
        ("maximum_scalar", DType::I32) => maximum_scalar::I32,
        ("maximum_scalar", DType::I64) => maximum_scalar::I64,

        ("minimum_scalar", DType::BOOL) => minimum_scalar::BOOL,
        ("minimum_scalar", DType::F8E4M3) => minimum_scalar::F8E4M3,
        ("minimum_scalar", DType::F8E5M2) => minimum_scalar::F8E5M2,
        ("minimum_scalar", DType::BF16) => minimum_scalar::BF16,
        ("minimum_scalar", DType::F16) => minimum_scalar::F16,
        ("minimum_scalar", DType::F32) => minimum_scalar::F32,
        ("minimum_scalar", DType::F64) => minimum_scalar::F64,
        ("minimum_scalar", DType::U8) => minimum_scalar::U8,
        ("minimum_scalar", DType::U16) => minimum_scalar::U16,
        ("minimum_scalar", DType::U32) => minimum_scalar::U32,
        ("minimum_scalar", DType::U64) => minimum_scalar::U64,
        ("minimum_scalar", DType::I8) => minimum_scalar::I8,
        ("minimum_scalar", DType::I16) => minimum_scalar::I16,
        ("minimum_scalar", DType::I32) => minimum_scalar::I32,
        ("minimum_scalar", DType::I64) => minimum_scalar::I64,

        ("eq_scalar", DType::BOOL) => eq_scalar::BOOL,
        ("eq_scalar", DType::F8E4M3) => eq_scalar::F8E4M3,
        ("eq_scalar", DType::F8E5M2) => eq_scalar::F8E5M2,
        ("eq_scalar", DType::BF16) => eq_scalar::BF16,
        ("eq_scalar", DType::F16) => eq_scalar::F16,
        ("eq_scalar", DType::F32) => eq_scalar::F32,
        ("eq_scalar", DType::F64) => eq_scalar::F64,
        ("eq_scalar", DType::U8) => eq_scalar::U8,
        ("eq_scalar", DType::U16) => eq_scalar::U16,
        ("eq_scalar", DType::U32) => eq_scalar::U32,
        ("eq_scalar", DType::U64) => eq_scalar::U64,
        ("eq_scalar", DType::I8) => eq_scalar::I8,
        ("eq_scalar", DType::I16) => eq_scalar::I16,
        ("eq_scalar", DType::I32) => eq_scalar::I32,
        ("eq_scalar", DType::I64) => eq_scalar::I64,

        ("ne_scalar", DType::BOOL) => ne_scalar::BOOL,
        ("ne_scalar", DType::F8E4M3) => ne_scalar::F8E4M3,
        ("ne_scalar", DType::F8E5M2) => ne_scalar::F8E5M2,
        ("ne_scalar", DType::BF16) => ne_scalar::BF16,
        ("ne_scalar", DType::F16) => ne_scalar::F16,
        ("ne_scalar", DType::F32) => ne_scalar::F32,
        ("ne_scalar", DType::F64) => ne_scalar::F64,
        ("ne_scalar", DType::U8) => ne_scalar::U8,
        ("ne_scalar", DType::U16) => ne_scalar::U16,
        ("ne_scalar", DType::U32) => ne_scalar::U32,
        ("ne_scalar", DType::U64) => ne_scalar::U64,
        ("ne_scalar", DType::I8) => ne_scalar::I8,
        ("ne_scalar", DType::I16) => ne_scalar::I16,
        ("ne_scalar", DType::I32) => ne_scalar::I32,
        ("ne_scalar", DType::I64) => ne_scalar::I64,

        ("lt_scalar", DType::BOOL) => lt_scalar::BOOL,
        ("lt_scalar", DType::F8E4M3) => lt_scalar::F8E4M3,
        ("lt_scalar", DType::F8E5M2) => lt_scalar::F8E5M2,
        ("lt_scalar", DType::BF16) => lt_scalar::BF16,
        ("lt_scalar", DType::F16) => lt_scalar::F16,
        ("lt_scalar", DType::F32) => lt_scalar::F32,
        ("lt_scalar", DType::F64) => lt_scalar::F64,
        ("lt_scalar", DType::U8) => lt_scalar::U8,
        ("lt_scalar", DType::U16) => lt_scalar::U16,
        ("lt_scalar", DType::U32) => lt_scalar::U32,
        ("lt_scalar", DType::U64) => lt_scalar::U64,
        ("lt_scalar", DType::I8) => lt_scalar::I8,
        ("lt_scalar", DType::I16) => lt_scalar::I16,
        ("lt_scalar", DType::I32) => lt_scalar::I32,
        ("lt_scalar", DType::I64) => lt_scalar::I64,

        ("le_scalar", DType::BOOL) => le_scalar::BOOL,
        ("le_scalar", DType::F8E4M3) => le_scalar::F8E4M3,
        ("le_scalar", DType::F8E5M2) => le_scalar::F8E5M2,
        ("le_scalar", DType::BF16) => le_scalar::BF16,
        ("le_scalar", DType::F16) => le_scalar::F16,
        ("le_scalar", DType::F32) => le_scalar::F32,
        ("le_scalar", DType::F64) => le_scalar::F64,
        ("le_scalar", DType::U8) => le_scalar::U8,
        ("le_scalar", DType::U16) => le_scalar::U16,
        ("le_scalar", DType::U32) => le_scalar::U32,
        ("le_scalar", DType::U64) => le_scalar::U64,
        ("le_scalar", DType::I8) => le_scalar::I8,
        ("le_scalar", DType::I16) => le_scalar::I16,
        ("le_scalar", DType::I32) => le_scalar::I32,
        ("le_scalar", DType::I64) => le_scalar::I64,

        ("gt_scalar", DType::BOOL) => gt_scalar::BOOL,
        ("gt_scalar", DType::F8E4M3) => gt_scalar::F8E4M3,
        ("gt_scalar", DType::F8E5M2) => gt_scalar::F8E5M2,
        ("gt_scalar", DType::BF16) => gt_scalar::BF16,
        ("gt_scalar", DType::F16) => gt_scalar::F16,
        ("gt_scalar", DType::F32) => gt_scalar::F32,
        ("gt_scalar", DType::F64) => gt_scalar::F64,
        ("gt_scalar", DType::U8) => gt_scalar::U8,
        ("gt_scalar", DType::U16) => gt_scalar::U16,
        ("gt_scalar", DType::U32) => gt_scalar::U32,
        ("gt_scalar", DType::U64) => gt_scalar::U64,
        ("gt_scalar", DType::I8) => gt_scalar::I8,
        ("gt_scalar", DType::I16) => gt_scalar::I16,
        ("gt_scalar", DType::I32) => gt_scalar::I32,
        ("gt_scalar", DType::I64) => gt_scalar::I64,

        ("ge_scalar", DType::BOOL) => ge_scalar::BOOL,
        ("ge_scalar", DType::F8E4M3) => ge_scalar::F8E4M3,
        ("ge_scalar", DType::F8E5M2) => ge_scalar::F8E5M2,
        ("ge_scalar", DType::BF16) => ge_scalar::BF16,
        ("ge_scalar", DType::F16) => ge_scalar::F16,
        ("ge_scalar", DType::F32) => ge_scalar::F32,
        ("ge_scalar", DType::F64) => ge_scalar::F64,
        ("ge_scalar", DType::U8) => ge_scalar::U8,
        ("ge_scalar", DType::U16) => ge_scalar::U16,
        ("ge_scalar", DType::U32) => ge_scalar::U32,
        ("ge_scalar", DType::U64) => ge_scalar::U64,
        ("ge_scalar", DType::I8) => ge_scalar::I8,
        ("ge_scalar", DType::I16) => ge_scalar::I16,
        ("ge_scalar", DType::I32) => ge_scalar::I32,
        ("ge_scalar", DType::I64) => ge_scalar::I64,

        _ => panic!("Unsupported scalar kernel: {} for {:?}", op_name, dtype),
    }
}
