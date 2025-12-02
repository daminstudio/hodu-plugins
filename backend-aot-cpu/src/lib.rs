//! AOT CPU backend plugin for Hodu
//!
//! Compiles Hodu Snapshots to native code via C code generation.

mod codegen;
mod compile;

use hodu_cli_plugin_sdk::{
    BackendPlugin, BuildFormat, BuildTarget, Device, PluginError, PluginResult, SdkDType, Snapshot, TensorData,
};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default, BackendPlugin)]
pub struct AotCpuBackend;

impl AotCpuBackend {
    pub fn supported_formats(&self, _target: &BuildTarget) -> Vec<BuildFormat> {
        vec![BuildFormat::SharedLib, BuildFormat::StaticLib, BuildFormat::Object]
    }

    pub fn build(
        &self,
        snapshot: &Snapshot,
        target: &BuildTarget,
        format: BuildFormat,
        output: &Path,
    ) -> PluginResult<()> {
        if target.device != Device::CPU {
            return Err(PluginError::NotSupported(format!(
                "Unsupported device: {}. Only CPU is supported.",
                target.device
            )));
        }

        let c_code = codegen::generate(snapshot)?;
        compile::compile(&c_code, target, format, output)
    }

    pub fn run(
        &self,
        snapshot: &Snapshot,
        device: Device,
        inputs: &[(&str, TensorData)],
    ) -> PluginResult<HashMap<String, TensorData>> {
        if device != Device::CPU {
            return Err(PluginError::NotSupported(format!(
                "Unsupported device: {}. Only CPU is supported.",
                device
            )));
        }

        // Build to temp shared library
        let temp_dir = std::env::temp_dir().join("hodu_aot_run");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| PluginError::Execution(format!("Failed to create temp dir: {}", e)))?;

        let lib_path = temp_dir.join("model.so");
        let target = BuildTarget::host(device);
        self.build(snapshot, &target, BuildFormat::SharedLib, &lib_path)?;

        // Load the library
        let lib = unsafe { libloading::Library::new(&lib_path) }
            .map_err(|e| PluginError::Execution(format!("Failed to load library: {}", e)))?;

        // Get model function name
        let model_name = snapshot
            .name
            .as_deref()
            .unwrap_or("model")
            .replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
        let func_name = format!("{}_run", model_name);

        // Get function pointer
        type RunFn = unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut *mut std::ffi::c_void);
        let run_fn: libloading::Symbol<RunFn> = unsafe { lib.get(func_name.as_bytes()) }
            .map_err(|e| PluginError::Execution(format!("Failed to find function {}: {}", func_name, e)))?;

        // Build input name to index mapping
        let input_map: HashMap<&str, usize> = snapshot
            .inputs
            .iter()
            .enumerate()
            .map(|(i, inp)| (inp.name.as_str(), i))
            .collect();

        // Prepare input pointers
        let mut input_ptrs: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); snapshot.inputs.len()];
        for (name, data) in inputs {
            let idx = input_map
                .get(name)
                .ok_or_else(|| PluginError::Execution(format!("Unknown input: {}", name)))?;
            input_ptrs[*idx] = data.data.as_ptr() as *mut std::ffi::c_void;
        }

        // Prepare output pointers (will be filled by the function)
        let mut output_ptrs: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); snapshot.targets.len()];

        // Call the function
        unsafe {
            run_fn(input_ptrs.as_mut_ptr(), output_ptrs.as_mut_ptr());
        }

        // Build output info map (target_id -> (shape, dtype))
        let output_info = build_output_info(snapshot);

        // Collect outputs
        let mut results = HashMap::new();
        for (i, target) in snapshot.targets.iter().enumerate() {
            let ptr = output_ptrs[i];
            if ptr.is_null() {
                return Err(PluginError::Execution(format!(
                    "Output {} was not set by the model",
                    target.name
                )));
            }

            let (shape, dtype) = output_info
                .get(&target.id.0)
                .ok_or_else(|| PluginError::Execution(format!("Could not find output info for {}", target.name)))?;

            let numel: usize = shape.iter().product();
            let byte_size = numel * dtype.size_in_bytes();

            // Copy data from the allocated buffer
            let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_size) }.to_vec();

            // Free the buffer allocated by the model
            unsafe {
                libc::free(ptr);
            }

            results.insert(target.name.clone(), TensorData::new(data, shape.clone(), *dtype));
        }

        // Cleanup: drop library first, then delete temp files
        drop(lib);
        let _ = std::fs::remove_dir_all(&temp_dir);

        Ok(results)
    }
}

/// Build a map from tensor id to (shape, dtype) for all outputs
fn build_output_info(snapshot: &Snapshot) -> HashMap<usize, (Vec<usize>, SdkDType)> {
    let mut info = HashMap::new();

    // Add input info
    for input in &snapshot.inputs {
        info.insert(input.id.0, (input.shape.dims().to_vec(), SdkDType::from(input.dtype)));
    }

    // Add constant info
    for constant in &snapshot.constants {
        info.insert(
            constant.id.0,
            (constant.shape.dims().to_vec(), SdkDType::from(constant.dtype)),
        );
    }

    // Add node output info
    for node in &snapshot.nodes {
        info.insert(
            node.output_id.0,
            (
                node.output_layout.shape().dims().to_vec(),
                SdkDType::from(node.output_dtype),
            ),
        );
    }

    info
}
