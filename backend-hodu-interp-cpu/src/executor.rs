//! Executor modules for different operation types

pub mod binary;
pub mod cast;
pub mod concat_split;
pub mod conv;
pub mod indexing;
pub mod matrix;
pub mod memory;
pub mod reduce;
pub mod shape;
pub mod unary;
pub mod windowing;

use hodu_cli_plugin_sdk::{Layout, PluginError, PluginResult, SdkDType, TensorData};
use hodu_core::types::DType;
use std::collections::HashMap;

/// Storage for tensor data during execution
///
/// Uses `DType` internally for compatibility with `hodu_cpu_kernels`.
/// Converts to `SdkDType` when creating `TensorData` for FFI boundary.
pub struct TensorStorage {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorStorage {
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let size = numel * dtype.get_size_in_bytes();
        Self {
            data: vec![0u8; size],
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn from_data(data: Vec<u8>, shape: Vec<usize>, dtype: SdkDType) -> Self {
        Self {
            data,
            shape,
            dtype: dtype.into(),
        }
    }

    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.data.as_ptr() as *const std::ffi::c_void
    }

    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.data.as_mut_ptr() as *mut std::ffi::c_void
    }

    pub fn to_tensor_data(&self) -> TensorData {
        TensorData::new(self.data.clone(), self.shape.clone(), self.dtype.into())
    }
}

/// Get tensor from storage by ID
pub fn get_tensor(tensors: &HashMap<usize, TensorStorage>, id: usize) -> PluginResult<&TensorStorage> {
    tensors
        .get(&id)
        .ok_or_else(|| PluginError::Execution(format!("Tensor {} not found", id)))
}

/// Build metadata for binary operations
/// Layout: [num_els, num_dims, lhs_shape..., rhs_shape..., lhs_strides..., rhs_strides..., lhs_offset, rhs_offset]
pub fn build_binary_metadata(lhs_layout: &Layout, rhs_layout: &Layout) -> Vec<usize> {
    let num_els = lhs_layout.shape().size();
    let num_dims = lhs_layout.ndim();

    let mut metadata = Vec::with_capacity(2 + 4 * num_dims + 2);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(lhs_layout.shape().dims());
    metadata.extend_from_slice(rhs_layout.shape().dims());
    metadata.extend_from_slice(lhs_layout.strides());
    metadata.extend_from_slice(rhs_layout.strides());
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    metadata
}

/// Build metadata for unary operations
/// Layout: [num_els, num_dims, shape..., strides..., offset]
pub fn build_unary_metadata(layout: &Layout) -> Vec<usize> {
    let num_els = layout.shape().size();
    let num_dims = layout.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(layout.shape().dims());
    metadata.extend_from_slice(layout.strides());
    metadata.push(layout.offset());

    metadata
}
