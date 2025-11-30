//! Cast operation executors

use super::{build_unary_metadata, get_tensor, TensorStorage};
use hodu_cpu_kernels::{call_ops_cast, CastKernel};
use hodu_plugin_sdk::{ops, snapshot::SnapshotNode, DType, HoduError, HoduResult};
use std::collections::HashMap;

pub fn execute_cast(
    tensors: &mut HashMap<usize, TensorStorage>,
    op: ops::CastOp,
    node: &SnapshotNode,
) -> HoduResult<()> {
    use ops::CastOp;

    match op {
        CastOp::ToDType => execute_to_dtype(tensors, node),
    }
}

fn execute_to_dtype(tensors: &mut HashMap<usize, TensorStorage>, node: &SnapshotNode) -> HoduResult<()> {
    let input_id = node.input_ids[0].0;
    let out_id = node.output_id.0;

    let input = get_tensor(tensors, input_id)?;
    let from_dtype = input.dtype;
    let input_ptr = input.as_ptr();

    let to_dtype = node.output_dtype;
    let out_shape = node.output_layout.shape().dims().to_vec();
    let mut output = TensorStorage::new(&out_shape, to_dtype);

    let kernel = get_cast_kernel(from_dtype, to_dtype);
    let metadata = build_unary_metadata(&node.input_layouts[0]);

    call_ops_cast(kernel, input_ptr, output.as_mut_ptr(), &metadata)
        .map_err(|e| HoduError::BackendError(format!("Kernel error: {:?}", e)))?;

    tensors.insert(out_id, output);
    Ok(())
}

fn get_cast_kernel(from: DType, to: DType) -> CastKernel {
    use hodu_cpu_kernels::cast::*;

    match (from, to) {
        // From BOOL
        (DType::BOOL, DType::BOOL) => from_bool::TO_BOOL,
        (DType::BOOL, DType::F8E4M3) => from_bool::TO_F8E4M3,
        (DType::BOOL, DType::F8E5M2) => from_bool::TO_F8E5M2,
        (DType::BOOL, DType::BF16) => from_bool::TO_BF16,
        (DType::BOOL, DType::F16) => from_bool::TO_F16,
        (DType::BOOL, DType::F32) => from_bool::TO_F32,
        (DType::BOOL, DType::F64) => from_bool::TO_F64,
        (DType::BOOL, DType::U8) => from_bool::TO_U8,
        (DType::BOOL, DType::U16) => from_bool::TO_U16,
        (DType::BOOL, DType::U32) => from_bool::TO_U32,
        (DType::BOOL, DType::U64) => from_bool::TO_U64,
        (DType::BOOL, DType::I8) => from_bool::TO_I8,
        (DType::BOOL, DType::I16) => from_bool::TO_I16,
        (DType::BOOL, DType::I32) => from_bool::TO_I32,
        (DType::BOOL, DType::I64) => from_bool::TO_I64,

        // From F8E4M3
        (DType::F8E4M3, DType::BOOL) => from_f8e4m3::TO_BOOL,
        (DType::F8E4M3, DType::F8E4M3) => from_f8e4m3::TO_F8E4M3,
        (DType::F8E4M3, DType::F8E5M2) => from_f8e4m3::TO_F8E5M2,
        (DType::F8E4M3, DType::BF16) => from_f8e4m3::TO_BF16,
        (DType::F8E4M3, DType::F16) => from_f8e4m3::TO_F16,
        (DType::F8E4M3, DType::F32) => from_f8e4m3::TO_F32,
        (DType::F8E4M3, DType::F64) => from_f8e4m3::TO_F64,
        (DType::F8E4M3, DType::U8) => from_f8e4m3::TO_U8,
        (DType::F8E4M3, DType::U16) => from_f8e4m3::TO_U16,
        (DType::F8E4M3, DType::U32) => from_f8e4m3::TO_U32,
        (DType::F8E4M3, DType::U64) => from_f8e4m3::TO_U64,
        (DType::F8E4M3, DType::I8) => from_f8e4m3::TO_I8,
        (DType::F8E4M3, DType::I16) => from_f8e4m3::TO_I16,
        (DType::F8E4M3, DType::I32) => from_f8e4m3::TO_I32,
        (DType::F8E4M3, DType::I64) => from_f8e4m3::TO_I64,

        // From F8E5M2
        (DType::F8E5M2, DType::BOOL) => from_f8e5m2::TO_BOOL,
        (DType::F8E5M2, DType::F8E4M3) => from_f8e5m2::TO_F8E4M3,
        (DType::F8E5M2, DType::F8E5M2) => from_f8e5m2::TO_F8E5M2,
        (DType::F8E5M2, DType::BF16) => from_f8e5m2::TO_BF16,
        (DType::F8E5M2, DType::F16) => from_f8e5m2::TO_F16,
        (DType::F8E5M2, DType::F32) => from_f8e5m2::TO_F32,
        (DType::F8E5M2, DType::F64) => from_f8e5m2::TO_F64,
        (DType::F8E5M2, DType::U8) => from_f8e5m2::TO_U8,
        (DType::F8E5M2, DType::U16) => from_f8e5m2::TO_U16,
        (DType::F8E5M2, DType::U32) => from_f8e5m2::TO_U32,
        (DType::F8E5M2, DType::U64) => from_f8e5m2::TO_U64,
        (DType::F8E5M2, DType::I8) => from_f8e5m2::TO_I8,
        (DType::F8E5M2, DType::I16) => from_f8e5m2::TO_I16,
        (DType::F8E5M2, DType::I32) => from_f8e5m2::TO_I32,
        (DType::F8E5M2, DType::I64) => from_f8e5m2::TO_I64,

        // From BF16
        (DType::BF16, DType::BOOL) => from_bf16::TO_BOOL,
        (DType::BF16, DType::F8E4M3) => from_bf16::TO_F8E4M3,
        (DType::BF16, DType::F8E5M2) => from_bf16::TO_F8E5M2,
        (DType::BF16, DType::BF16) => from_bf16::TO_BF16,
        (DType::BF16, DType::F16) => from_bf16::TO_F16,
        (DType::BF16, DType::F32) => from_bf16::TO_F32,
        (DType::BF16, DType::F64) => from_bf16::TO_F64,
        (DType::BF16, DType::U8) => from_bf16::TO_U8,
        (DType::BF16, DType::U16) => from_bf16::TO_U16,
        (DType::BF16, DType::U32) => from_bf16::TO_U32,
        (DType::BF16, DType::U64) => from_bf16::TO_U64,
        (DType::BF16, DType::I8) => from_bf16::TO_I8,
        (DType::BF16, DType::I16) => from_bf16::TO_I16,
        (DType::BF16, DType::I32) => from_bf16::TO_I32,
        (DType::BF16, DType::I64) => from_bf16::TO_I64,

        // From F16
        (DType::F16, DType::BOOL) => from_f16::TO_BOOL,
        (DType::F16, DType::F8E4M3) => from_f16::TO_F8E4M3,
        (DType::F16, DType::F8E5M2) => from_f16::TO_F8E5M2,
        (DType::F16, DType::BF16) => from_f16::TO_BF16,
        (DType::F16, DType::F16) => from_f16::TO_F16,
        (DType::F16, DType::F32) => from_f16::TO_F32,
        (DType::F16, DType::F64) => from_f16::TO_F64,
        (DType::F16, DType::U8) => from_f16::TO_U8,
        (DType::F16, DType::U16) => from_f16::TO_U16,
        (DType::F16, DType::U32) => from_f16::TO_U32,
        (DType::F16, DType::U64) => from_f16::TO_U64,
        (DType::F16, DType::I8) => from_f16::TO_I8,
        (DType::F16, DType::I16) => from_f16::TO_I16,
        (DType::F16, DType::I32) => from_f16::TO_I32,
        (DType::F16, DType::I64) => from_f16::TO_I64,

        // From F32
        (DType::F32, DType::BOOL) => from_f32::TO_BOOL,
        (DType::F32, DType::F8E4M3) => from_f32::TO_F8E4M3,
        (DType::F32, DType::F8E5M2) => from_f32::TO_F8E5M2,
        (DType::F32, DType::BF16) => from_f32::TO_BF16,
        (DType::F32, DType::F16) => from_f32::TO_F16,
        (DType::F32, DType::F32) => from_f32::TO_F32,
        (DType::F32, DType::F64) => from_f32::TO_F64,
        (DType::F32, DType::U8) => from_f32::TO_U8,
        (DType::F32, DType::U16) => from_f32::TO_U16,
        (DType::F32, DType::U32) => from_f32::TO_U32,
        (DType::F32, DType::U64) => from_f32::TO_U64,
        (DType::F32, DType::I8) => from_f32::TO_I8,
        (DType::F32, DType::I16) => from_f32::TO_I16,
        (DType::F32, DType::I32) => from_f32::TO_I32,
        (DType::F32, DType::I64) => from_f32::TO_I64,

        // From F64
        (DType::F64, DType::BOOL) => from_f64::TO_BOOL,
        (DType::F64, DType::F8E4M3) => from_f64::TO_F8E4M3,
        (DType::F64, DType::F8E5M2) => from_f64::TO_F8E5M2,
        (DType::F64, DType::BF16) => from_f64::TO_BF16,
        (DType::F64, DType::F16) => from_f64::TO_F16,
        (DType::F64, DType::F32) => from_f64::TO_F32,
        (DType::F64, DType::F64) => from_f64::TO_F64,
        (DType::F64, DType::U8) => from_f64::TO_U8,
        (DType::F64, DType::U16) => from_f64::TO_U16,
        (DType::F64, DType::U32) => from_f64::TO_U32,
        (DType::F64, DType::U64) => from_f64::TO_U64,
        (DType::F64, DType::I8) => from_f64::TO_I8,
        (DType::F64, DType::I16) => from_f64::TO_I16,
        (DType::F64, DType::I32) => from_f64::TO_I32,
        (DType::F64, DType::I64) => from_f64::TO_I64,

        // From U8
        (DType::U8, DType::BOOL) => from_u8::TO_BOOL,
        (DType::U8, DType::F8E4M3) => from_u8::TO_F8E4M3,
        (DType::U8, DType::F8E5M2) => from_u8::TO_F8E5M2,
        (DType::U8, DType::BF16) => from_u8::TO_BF16,
        (DType::U8, DType::F16) => from_u8::TO_F16,
        (DType::U8, DType::F32) => from_u8::TO_F32,
        (DType::U8, DType::F64) => from_u8::TO_F64,
        (DType::U8, DType::U8) => from_u8::TO_U8,
        (DType::U8, DType::U16) => from_u8::TO_U16,
        (DType::U8, DType::U32) => from_u8::TO_U32,
        (DType::U8, DType::U64) => from_u8::TO_U64,
        (DType::U8, DType::I8) => from_u8::TO_I8,
        (DType::U8, DType::I16) => from_u8::TO_I16,
        (DType::U8, DType::I32) => from_u8::TO_I32,
        (DType::U8, DType::I64) => from_u8::TO_I64,

        // From U16
        (DType::U16, DType::BOOL) => from_u16::TO_BOOL,
        (DType::U16, DType::F8E4M3) => from_u16::TO_F8E4M3,
        (DType::U16, DType::F8E5M2) => from_u16::TO_F8E5M2,
        (DType::U16, DType::BF16) => from_u16::TO_BF16,
        (DType::U16, DType::F16) => from_u16::TO_F16,
        (DType::U16, DType::F32) => from_u16::TO_F32,
        (DType::U16, DType::F64) => from_u16::TO_F64,
        (DType::U16, DType::U8) => from_u16::TO_U8,
        (DType::U16, DType::U16) => from_u16::TO_U16,
        (DType::U16, DType::U32) => from_u16::TO_U32,
        (DType::U16, DType::U64) => from_u16::TO_U64,
        (DType::U16, DType::I8) => from_u16::TO_I8,
        (DType::U16, DType::I16) => from_u16::TO_I16,
        (DType::U16, DType::I32) => from_u16::TO_I32,
        (DType::U16, DType::I64) => from_u16::TO_I64,

        // From U32
        (DType::U32, DType::BOOL) => from_u32::TO_BOOL,
        (DType::U32, DType::F8E4M3) => from_u32::TO_F8E4M3,
        (DType::U32, DType::F8E5M2) => from_u32::TO_F8E5M2,
        (DType::U32, DType::BF16) => from_u32::TO_BF16,
        (DType::U32, DType::F16) => from_u32::TO_F16,
        (DType::U32, DType::F32) => from_u32::TO_F32,
        (DType::U32, DType::F64) => from_u32::TO_F64,
        (DType::U32, DType::U8) => from_u32::TO_U8,
        (DType::U32, DType::U16) => from_u32::TO_U16,
        (DType::U32, DType::U32) => from_u32::TO_U32,
        (DType::U32, DType::U64) => from_u32::TO_U64,
        (DType::U32, DType::I8) => from_u32::TO_I8,
        (DType::U32, DType::I16) => from_u32::TO_I16,
        (DType::U32, DType::I32) => from_u32::TO_I32,
        (DType::U32, DType::I64) => from_u32::TO_I64,

        // From U64
        (DType::U64, DType::BOOL) => from_u64::TO_BOOL,
        (DType::U64, DType::F8E4M3) => from_u64::TO_F8E4M3,
        (DType::U64, DType::F8E5M2) => from_u64::TO_F8E5M2,
        (DType::U64, DType::BF16) => from_u64::TO_BF16,
        (DType::U64, DType::F16) => from_u64::TO_F16,
        (DType::U64, DType::F32) => from_u64::TO_F32,
        (DType::U64, DType::F64) => from_u64::TO_F64,
        (DType::U64, DType::U8) => from_u64::TO_U8,
        (DType::U64, DType::U16) => from_u64::TO_U16,
        (DType::U64, DType::U32) => from_u64::TO_U32,
        (DType::U64, DType::U64) => from_u64::TO_U64,
        (DType::U64, DType::I8) => from_u64::TO_I8,
        (DType::U64, DType::I16) => from_u64::TO_I16,
        (DType::U64, DType::I32) => from_u64::TO_I32,
        (DType::U64, DType::I64) => from_u64::TO_I64,

        // From I8
        (DType::I8, DType::BOOL) => from_i8::TO_BOOL,
        (DType::I8, DType::F8E4M3) => from_i8::TO_F8E4M3,
        (DType::I8, DType::F8E5M2) => from_i8::TO_F8E5M2,
        (DType::I8, DType::BF16) => from_i8::TO_BF16,
        (DType::I8, DType::F16) => from_i8::TO_F16,
        (DType::I8, DType::F32) => from_i8::TO_F32,
        (DType::I8, DType::F64) => from_i8::TO_F64,
        (DType::I8, DType::U8) => from_i8::TO_U8,
        (DType::I8, DType::U16) => from_i8::TO_U16,
        (DType::I8, DType::U32) => from_i8::TO_U32,
        (DType::I8, DType::U64) => from_i8::TO_U64,
        (DType::I8, DType::I8) => from_i8::TO_I8,
        (DType::I8, DType::I16) => from_i8::TO_I16,
        (DType::I8, DType::I32) => from_i8::TO_I32,
        (DType::I8, DType::I64) => from_i8::TO_I64,

        // From I16
        (DType::I16, DType::BOOL) => from_i16::TO_BOOL,
        (DType::I16, DType::F8E4M3) => from_i16::TO_F8E4M3,
        (DType::I16, DType::F8E5M2) => from_i16::TO_F8E5M2,
        (DType::I16, DType::BF16) => from_i16::TO_BF16,
        (DType::I16, DType::F16) => from_i16::TO_F16,
        (DType::I16, DType::F32) => from_i16::TO_F32,
        (DType::I16, DType::F64) => from_i16::TO_F64,
        (DType::I16, DType::U8) => from_i16::TO_U8,
        (DType::I16, DType::U16) => from_i16::TO_U16,
        (DType::I16, DType::U32) => from_i16::TO_U32,
        (DType::I16, DType::U64) => from_i16::TO_U64,
        (DType::I16, DType::I8) => from_i16::TO_I8,
        (DType::I16, DType::I16) => from_i16::TO_I16,
        (DType::I16, DType::I32) => from_i16::TO_I32,
        (DType::I16, DType::I64) => from_i16::TO_I64,

        // From I32
        (DType::I32, DType::BOOL) => from_i32::TO_BOOL,
        (DType::I32, DType::F8E4M3) => from_i32::TO_F8E4M3,
        (DType::I32, DType::F8E5M2) => from_i32::TO_F8E5M2,
        (DType::I32, DType::BF16) => from_i32::TO_BF16,
        (DType::I32, DType::F16) => from_i32::TO_F16,
        (DType::I32, DType::F32) => from_i32::TO_F32,
        (DType::I32, DType::F64) => from_i32::TO_F64,
        (DType::I32, DType::U8) => from_i32::TO_U8,
        (DType::I32, DType::U16) => from_i32::TO_U16,
        (DType::I32, DType::U32) => from_i32::TO_U32,
        (DType::I32, DType::U64) => from_i32::TO_U64,
        (DType::I32, DType::I8) => from_i32::TO_I8,
        (DType::I32, DType::I16) => from_i32::TO_I16,
        (DType::I32, DType::I32) => from_i32::TO_I32,
        (DType::I32, DType::I64) => from_i32::TO_I64,

        // From I64
        (DType::I64, DType::BOOL) => from_i64::TO_BOOL,
        (DType::I64, DType::F8E4M3) => from_i64::TO_F8E4M3,
        (DType::I64, DType::F8E5M2) => from_i64::TO_F8E5M2,
        (DType::I64, DType::BF16) => from_i64::TO_BF16,
        (DType::I64, DType::F16) => from_i64::TO_F16,
        (DType::I64, DType::F32) => from_i64::TO_F32,
        (DType::I64, DType::F64) => from_i64::TO_F64,
        (DType::I64, DType::U8) => from_i64::TO_U8,
        (DType::I64, DType::U16) => from_i64::TO_U16,
        (DType::I64, DType::U32) => from_i64::TO_U32,
        (DType::I64, DType::U64) => from_i64::TO_U64,
        (DType::I64, DType::I8) => from_i64::TO_I8,
        (DType::I64, DType::I16) => from_i64::TO_I16,
        (DType::I64, DType::I32) => from_i64::TO_I32,
        (DType::I64, DType::I64) => from_i64::TO_I64,
    }
}
