//! Binary operation executors

use super::{build_binary_metadata, get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_binary, Kernel};
use hodu_plugin_sdk::{ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_binary(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::BinaryOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::BinaryOp;

    let lhs_id = node.input_ids[0].0;
    let rhs_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let lhs = get_tensor(tensors, lhs_id)?;
    let rhs = get_tensor(tensors, rhs_id)?;
    let lhs_dtype = lhs.dtype;
    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        BinaryOp::Add => get_kernel("add", lhs_dtype),
        BinaryOp::Sub => get_kernel("sub", lhs_dtype),
        BinaryOp::Mul => get_kernel("mul", lhs_dtype),
        BinaryOp::Div => get_kernel("div", lhs_dtype),
        BinaryOp::Pow => get_kernel("pow", lhs_dtype),
        BinaryOp::Maximum => get_kernel("maximum", lhs_dtype),
        BinaryOp::Minimum => get_kernel("minimum", lhs_dtype),
    };

    let metadata = build_binary_metadata(&node.input_layouts[0], &node.input_layouts[1]);

    call_ops_binary(kernel, lhs_ptr, rhs_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

pub fn execute_binary_logical(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::BinaryLogicalOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::BinaryLogicalOp;

    let lhs_id = node.input_ids[0].0;
    let rhs_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let lhs = get_tensor(tensors, lhs_id)?;
    let rhs = get_tensor(tensors, rhs_id)?;
    let lhs_dtype = lhs.dtype;
    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        BinaryLogicalOp::LogicalAnd => get_kernel("logical_and", lhs_dtype),
        BinaryLogicalOp::LogicalOr => get_kernel("logical_or", lhs_dtype),
        BinaryLogicalOp::LogicalXor => get_kernel("logical_xor", lhs_dtype),
    };

    let metadata = build_binary_metadata(&node.input_layouts[0], &node.input_layouts[1]);

    call_ops_binary(kernel, lhs_ptr, rhs_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

pub fn execute_cmp(tensors: &mut HashMap<usize, TensorStorage>, op: ops::CmpOp, node: &SnapshotNode) -> HoduResult<()> {
    use ops::CmpOp;

    let lhs_id = node.input_ids[0].0;
    let rhs_id = node.input_ids[1].0;
    let out_id = node.output_id.0;

    let lhs = get_tensor(tensors, lhs_id)?;
    let rhs = get_tensor(tensors, rhs_id)?;
    let lhs_dtype = lhs.dtype;
    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();

    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, node.output_dtype);

    let kernel = match op {
        CmpOp::Eq => get_kernel("eq", lhs_dtype),
        CmpOp::Ne => get_kernel("ne", lhs_dtype),
        CmpOp::Lt => get_kernel("lt", lhs_dtype),
        CmpOp::Le => get_kernel("le", lhs_dtype),
        CmpOp::Gt => get_kernel("gt", lhs_dtype),
        CmpOp::Ge => get_kernel("ge", lhs_dtype),
    };

    let metadata = build_binary_metadata(&node.input_layouts[0], &node.input_layouts[1]);

    call_ops_binary(kernel, lhs_ptr, rhs_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_kernel(op_name: &str, dtype: DType) -> Kernel {
    use hodu_cpu_kernels::*;

    match (op_name, dtype) {
        ("add", DType::BOOL) => add::BOOL,
        ("add", DType::F8E4M3) => add::F8E4M3,
        ("add", DType::F8E5M2) => add::F8E5M2,
        ("add", DType::BF16) => add::BF16,
        ("add", DType::F16) => add::F16,
        ("add", DType::F32) => add::F32,
        ("add", DType::F64) => add::F64,
        ("add", DType::U8) => add::U8,
        ("add", DType::U16) => add::U16,
        ("add", DType::U32) => add::U32,
        ("add", DType::U64) => add::U64,
        ("add", DType::I8) => add::I8,
        ("add", DType::I16) => add::I16,
        ("add", DType::I32) => add::I32,
        ("add", DType::I64) => add::I64,

        ("sub", DType::BOOL) => sub::BOOL,
        ("sub", DType::F8E4M3) => sub::F8E4M3,
        ("sub", DType::F8E5M2) => sub::F8E5M2,
        ("sub", DType::BF16) => sub::BF16,
        ("sub", DType::F16) => sub::F16,
        ("sub", DType::F32) => sub::F32,
        ("sub", DType::F64) => sub::F64,
        ("sub", DType::U8) => sub::U8,
        ("sub", DType::U16) => sub::U16,
        ("sub", DType::U32) => sub::U32,
        ("sub", DType::U64) => sub::U64,
        ("sub", DType::I8) => sub::I8,
        ("sub", DType::I16) => sub::I16,
        ("sub", DType::I32) => sub::I32,
        ("sub", DType::I64) => sub::I64,

        ("mul", DType::BOOL) => mul::BOOL,
        ("mul", DType::F8E4M3) => mul::F8E4M3,
        ("mul", DType::F8E5M2) => mul::F8E5M2,
        ("mul", DType::BF16) => mul::BF16,
        ("mul", DType::F16) => mul::F16,
        ("mul", DType::F32) => mul::F32,
        ("mul", DType::F64) => mul::F64,
        ("mul", DType::U8) => mul::U8,
        ("mul", DType::U16) => mul::U16,
        ("mul", DType::U32) => mul::U32,
        ("mul", DType::U64) => mul::U64,
        ("mul", DType::I8) => mul::I8,
        ("mul", DType::I16) => mul::I16,
        ("mul", DType::I32) => mul::I32,
        ("mul", DType::I64) => mul::I64,

        ("div", DType::BOOL) => div::BOOL,
        ("div", DType::F8E4M3) => div::F8E4M3,
        ("div", DType::F8E5M2) => div::F8E5M2,
        ("div", DType::BF16) => div::BF16,
        ("div", DType::F16) => div::F16,
        ("div", DType::F32) => div::F32,
        ("div", DType::F64) => div::F64,
        ("div", DType::U8) => div::U8,
        ("div", DType::U16) => div::U16,
        ("div", DType::U32) => div::U32,
        ("div", DType::U64) => div::U64,
        ("div", DType::I8) => div::I8,
        ("div", DType::I16) => div::I16,
        ("div", DType::I32) => div::I32,
        ("div", DType::I64) => div::I64,

        ("pow", DType::BOOL) => pow::BOOL,
        ("pow", DType::F8E4M3) => pow::F8E4M3,
        ("pow", DType::F8E5M2) => pow::F8E5M2,
        ("pow", DType::BF16) => pow::BF16,
        ("pow", DType::F16) => pow::F16,
        ("pow", DType::F32) => pow::F32,
        ("pow", DType::F64) => pow::F64,
        ("pow", DType::U8) => pow::U8,
        ("pow", DType::U16) => pow::U16,
        ("pow", DType::U32) => pow::U32,
        ("pow", DType::U64) => pow::U64,
        ("pow", DType::I8) => pow::I8,
        ("pow", DType::I16) => pow::I16,
        ("pow", DType::I32) => pow::I32,
        ("pow", DType::I64) => pow::I64,

        ("maximum", DType::BOOL) => maximum::BOOL,
        ("maximum", DType::F8E4M3) => maximum::F8E4M3,
        ("maximum", DType::F8E5M2) => maximum::F8E5M2,
        ("maximum", DType::BF16) => maximum::BF16,
        ("maximum", DType::F16) => maximum::F16,
        ("maximum", DType::F32) => maximum::F32,
        ("maximum", DType::F64) => maximum::F64,
        ("maximum", DType::U8) => maximum::U8,
        ("maximum", DType::U16) => maximum::U16,
        ("maximum", DType::U32) => maximum::U32,
        ("maximum", DType::U64) => maximum::U64,
        ("maximum", DType::I8) => maximum::I8,
        ("maximum", DType::I16) => maximum::I16,
        ("maximum", DType::I32) => maximum::I32,
        ("maximum", DType::I64) => maximum::I64,

        ("minimum", DType::BOOL) => minimum::BOOL,
        ("minimum", DType::F8E4M3) => minimum::F8E4M3,
        ("minimum", DType::F8E5M2) => minimum::F8E5M2,
        ("minimum", DType::BF16) => minimum::BF16,
        ("minimum", DType::F16) => minimum::F16,
        ("minimum", DType::F32) => minimum::F32,
        ("minimum", DType::F64) => minimum::F64,
        ("minimum", DType::U8) => minimum::U8,
        ("minimum", DType::U16) => minimum::U16,
        ("minimum", DType::U32) => minimum::U32,
        ("minimum", DType::U64) => minimum::U64,
        ("minimum", DType::I8) => minimum::I8,
        ("minimum", DType::I16) => minimum::I16,
        ("minimum", DType::I32) => minimum::I32,
        ("minimum", DType::I64) => minimum::I64,

        ("logical_and", DType::BOOL) => logical_and::BOOL,
        ("logical_and", DType::F8E4M3) => logical_and::F8E4M3,
        ("logical_and", DType::F8E5M2) => logical_and::F8E5M2,
        ("logical_and", DType::BF16) => logical_and::BF16,
        ("logical_and", DType::F16) => logical_and::F16,
        ("logical_and", DType::F32) => logical_and::F32,
        ("logical_and", DType::F64) => logical_and::F64,
        ("logical_and", DType::U8) => logical_and::U8,
        ("logical_and", DType::U16) => logical_and::U16,
        ("logical_and", DType::U32) => logical_and::U32,
        ("logical_and", DType::U64) => logical_and::U64,
        ("logical_and", DType::I8) => logical_and::I8,
        ("logical_and", DType::I16) => logical_and::I16,
        ("logical_and", DType::I32) => logical_and::I32,
        ("logical_and", DType::I64) => logical_and::I64,

        ("logical_or", DType::BOOL) => logical_or::BOOL,
        ("logical_or", DType::F8E4M3) => logical_or::F8E4M3,
        ("logical_or", DType::F8E5M2) => logical_or::F8E5M2,
        ("logical_or", DType::BF16) => logical_or::BF16,
        ("logical_or", DType::F16) => logical_or::F16,
        ("logical_or", DType::F32) => logical_or::F32,
        ("logical_or", DType::F64) => logical_or::F64,
        ("logical_or", DType::U8) => logical_or::U8,
        ("logical_or", DType::U16) => logical_or::U16,
        ("logical_or", DType::U32) => logical_or::U32,
        ("logical_or", DType::U64) => logical_or::U64,
        ("logical_or", DType::I8) => logical_or::I8,
        ("logical_or", DType::I16) => logical_or::I16,
        ("logical_or", DType::I32) => logical_or::I32,
        ("logical_or", DType::I64) => logical_or::I64,

        ("logical_xor", DType::BOOL) => logical_xor::BOOL,
        ("logical_xor", DType::F8E4M3) => logical_xor::F8E4M3,
        ("logical_xor", DType::F8E5M2) => logical_xor::F8E5M2,
        ("logical_xor", DType::BF16) => logical_xor::BF16,
        ("logical_xor", DType::F16) => logical_xor::F16,
        ("logical_xor", DType::F32) => logical_xor::F32,
        ("logical_xor", DType::F64) => logical_xor::F64,
        ("logical_xor", DType::U8) => logical_xor::U8,
        ("logical_xor", DType::U16) => logical_xor::U16,
        ("logical_xor", DType::U32) => logical_xor::U32,
        ("logical_xor", DType::U64) => logical_xor::U64,
        ("logical_xor", DType::I8) => logical_xor::I8,
        ("logical_xor", DType::I16) => logical_xor::I16,
        ("logical_xor", DType::I32) => logical_xor::I32,
        ("logical_xor", DType::I64) => logical_xor::I64,

        ("eq", DType::BOOL) => eq::BOOL,
        ("eq", DType::F8E4M3) => eq::F8E4M3,
        ("eq", DType::F8E5M2) => eq::F8E5M2,
        ("eq", DType::BF16) => eq::BF16,
        ("eq", DType::F16) => eq::F16,
        ("eq", DType::F32) => eq::F32,
        ("eq", DType::F64) => eq::F64,
        ("eq", DType::U8) => eq::U8,
        ("eq", DType::U16) => eq::U16,
        ("eq", DType::U32) => eq::U32,
        ("eq", DType::U64) => eq::U64,
        ("eq", DType::I8) => eq::I8,
        ("eq", DType::I16) => eq::I16,
        ("eq", DType::I32) => eq::I32,
        ("eq", DType::I64) => eq::I64,

        ("ne", DType::BOOL) => ne::BOOL,
        ("ne", DType::F8E4M3) => ne::F8E4M3,
        ("ne", DType::F8E5M2) => ne::F8E5M2,
        ("ne", DType::BF16) => ne::BF16,
        ("ne", DType::F16) => ne::F16,
        ("ne", DType::F32) => ne::F32,
        ("ne", DType::F64) => ne::F64,
        ("ne", DType::U8) => ne::U8,
        ("ne", DType::U16) => ne::U16,
        ("ne", DType::U32) => ne::U32,
        ("ne", DType::U64) => ne::U64,
        ("ne", DType::I8) => ne::I8,
        ("ne", DType::I16) => ne::I16,
        ("ne", DType::I32) => ne::I32,
        ("ne", DType::I64) => ne::I64,

        ("lt", DType::BOOL) => lt::BOOL,
        ("lt", DType::F8E4M3) => lt::F8E4M3,
        ("lt", DType::F8E5M2) => lt::F8E5M2,
        ("lt", DType::BF16) => lt::BF16,
        ("lt", DType::F16) => lt::F16,
        ("lt", DType::F32) => lt::F32,
        ("lt", DType::F64) => lt::F64,
        ("lt", DType::U8) => lt::U8,
        ("lt", DType::U16) => lt::U16,
        ("lt", DType::U32) => lt::U32,
        ("lt", DType::U64) => lt::U64,
        ("lt", DType::I8) => lt::I8,
        ("lt", DType::I16) => lt::I16,
        ("lt", DType::I32) => lt::I32,
        ("lt", DType::I64) => lt::I64,

        ("le", DType::BOOL) => le::BOOL,
        ("le", DType::F8E4M3) => le::F8E4M3,
        ("le", DType::F8E5M2) => le::F8E5M2,
        ("le", DType::BF16) => le::BF16,
        ("le", DType::F16) => le::F16,
        ("le", DType::F32) => le::F32,
        ("le", DType::F64) => le::F64,
        ("le", DType::U8) => le::U8,
        ("le", DType::U16) => le::U16,
        ("le", DType::U32) => le::U32,
        ("le", DType::U64) => le::U64,
        ("le", DType::I8) => le::I8,
        ("le", DType::I16) => le::I16,
        ("le", DType::I32) => le::I32,
        ("le", DType::I64) => le::I64,

        ("gt", DType::BOOL) => gt::BOOL,
        ("gt", DType::F8E4M3) => gt::F8E4M3,
        ("gt", DType::F8E5M2) => gt::F8E5M2,
        ("gt", DType::BF16) => gt::BF16,
        ("gt", DType::F16) => gt::F16,
        ("gt", DType::F32) => gt::F32,
        ("gt", DType::F64) => gt::F64,
        ("gt", DType::U8) => gt::U8,
        ("gt", DType::U16) => gt::U16,
        ("gt", DType::U32) => gt::U32,
        ("gt", DType::U64) => gt::U64,
        ("gt", DType::I8) => gt::I8,
        ("gt", DType::I16) => gt::I16,
        ("gt", DType::I32) => gt::I32,
        ("gt", DType::I64) => gt::I64,

        ("ge", DType::BOOL) => ge::BOOL,
        ("ge", DType::F8E4M3) => ge::F8E4M3,
        ("ge", DType::F8E5M2) => ge::F8E5M2,
        ("ge", DType::BF16) => ge::BF16,
        ("ge", DType::F16) => ge::F16,
        ("ge", DType::F32) => ge::F32,
        ("ge", DType::F64) => ge::F64,
        ("ge", DType::U8) => ge::U8,
        ("ge", DType::U16) => ge::U16,
        ("ge", DType::U32) => ge::U32,
        ("ge", DType::U64) => ge::U64,
        ("ge", DType::I8) => ge::I8,
        ("ge", DType::I16) => ge::I16,
        ("ge", DType::I32) => ge::I32,
        ("ge", DType::I64) => ge::I64,

        _ => panic!("Unsupported binary kernel: {} for {:?}", op_name, dtype),
    }
}
