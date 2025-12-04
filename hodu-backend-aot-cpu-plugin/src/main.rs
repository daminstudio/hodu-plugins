//! AOT CPU backend plugin for Hodu
//!
//! Compiles Hodu Snapshots to native code via C code generation.

mod codegen;
mod compile;

use hodu_plugin_sdk::{
    hdss, hdt, notify_progress,
    rpc::{BuildParams, RpcError, RunParams, RunResult, TensorOutput},
    server::PluginServer,
    CoreDevice, DType, Snapshot, Tensor,
};
use std::collections::HashMap;
use std::path::Path;

fn main() {
    let server = PluginServer::new("aot-cpu", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])
        .method("backend.run", handle_run)
        .method("backend.build", handle_build)
        .method("backend.supported_targets", handle_list_targets);

    if let Err(e) = server.run() {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

fn handle_list_targets(_params: serde_json::Value) -> Result<serde_json::Value, RpcError> {
    Ok(serde_json::json!({
        "targets": compile::SUPPORTED_TARGETS,
        "formatted": compile::list_supported_targets()
    }))
}

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {
    // Validate device
    if params.device.to_lowercase() != "cpu" {
        return Err(RpcError::not_supported(&format!(
            "Unsupported device: {}. Only CPU is supported.",
            params.device
        )));
    }

    notify_progress(Some(0), "Loading snapshot...");

    // Load snapshot
    let snapshot = hdss::load(&params.snapshot_path)
        .map_err(|e| RpcError::internal_error(format!("Failed to load snapshot: {}", e)))?;

    notify_progress(Some(10), "Loading input tensors...");

    // Load input tensors
    let mut inputs: Vec<(&str, Vec<u8>)> = Vec::new();
    let mut input_tensors = Vec::new();
    for input in &params.inputs {
        let tensor = hdt::load(&input.path)
            .map_err(|e| RpcError::internal_error(format!("Failed to load tensor {}: {}", input.name, e)))?;
        input_tensors.push((input.name.clone(), tensor));
    }

    // Convert to the format expected by run_model
    for (name, tensor) in &input_tensors {
        let data = tensor
            .to_bytes()
            .map_err(|e| RpcError::internal_error(format!("Failed to get tensor data: {}", e)))?;
        inputs.push((name.as_str(), data));
    }

    notify_progress(Some(20), "Compiling model...");

    // Run the model
    let results = run_model(&snapshot, &inputs)?;

    notify_progress(Some(80), "Saving output tensors...");

    // Save outputs and build result
    let temp_dir = std::env::temp_dir().join("hodu_aot_cpu_output");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| RpcError::internal_error(format!("Failed to create temp dir: {}", e)))?;

    let mut outputs = Vec::new();
    for (name, (data, shape, dtype)) in results {
        let output_path = temp_dir.join(format!("{}.hdt", name));

        // Create tensor and save
        let tensor = Tensor::from_bytes(&data, shape, dtype, CoreDevice::CPU)
            .map_err(|e| RpcError::internal_error(format!("Failed to create tensor: {}", e)))?;
        hdt::save(&tensor, &output_path)
            .map_err(|e| RpcError::internal_error(format!("Failed to save tensor: {}", e)))?;

        outputs.push(TensorOutput {
            name,
            path: output_path.to_string_lossy().to_string(),
        });
    }

    notify_progress(Some(100), "Done");

    Ok(RunResult { outputs })
}

fn handle_build(params: BuildParams) -> Result<serde_json::Value, RpcError> {
    // Validate device
    if params.device.to_lowercase() != "cpu" {
        return Err(RpcError::not_supported(&format!(
            "Unsupported device: {}. Only CPU is supported.",
            params.device
        )));
    }

    notify_progress(Some(0), "Loading snapshot...");

    // Load snapshot
    let snapshot = hdss::load(&params.snapshot_path)
        .map_err(|e| RpcError::internal_error(format!("Failed to load snapshot: {}", e)))?;

    notify_progress(Some(20), "Generating C code...");

    // Generate C code
    let c_code =
        codegen::generate(&snapshot).map_err(|e| RpcError::internal_error(format!("Code generation failed: {}", e)))?;

    notify_progress(Some(50), "Compiling...");

    // Compile
    let format = match params.format.as_str() {
        "executable" | "exe" | "bin" => compile::BuildFormat::Executable,
        "sharedlib" | "dylib" | "so" | "dll" => compile::BuildFormat::SharedLib,
        "staticlib" | "a" | "lib" => compile::BuildFormat::StaticLib,
        "object" | "o" | "obj" => compile::BuildFormat::Object,
        _ => {
            return Err(RpcError::not_supported(&format!(
                "Unsupported format: '{}'. Supported: executable, sharedlib, staticlib, object",
                params.format
            )))
        },
    };

    let target = compile::BuildTarget {
        triple: params.target.clone(),
        device: params.device.clone(),
    };

    compile::compile(&c_code, &target, format, Path::new(&params.output_path))
        .map_err(|e| RpcError::internal_error(format!("Compilation failed: {}", e)))?;

    notify_progress(Some(100), "Done");

    Ok(serde_json::json!({}))
}

type TensorResult = (Vec<u8>, Vec<usize>, DType);

/// Run the model by compiling to a shared library and executing
fn run_model(snapshot: &Snapshot, inputs: &[(&str, Vec<u8>)]) -> Result<HashMap<String, TensorResult>, RpcError> {
    // Build to temp shared library
    let temp_dir = std::env::temp_dir().join("hodu_aot_run");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| RpcError::internal_error(format!("Failed to create temp dir: {}", e)))?;

    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let lib_path = temp_dir.join(format!("model.{}", lib_ext));

    // Generate and compile
    let c_code =
        codegen::generate(snapshot).map_err(|e| RpcError::internal_error(format!("Code generation failed: {}", e)))?;

    let target = compile::BuildTarget {
        triple: current_target_triple(),
        device: "cpu".to_string(),
    };

    compile::compile(&c_code, &target, compile::BuildFormat::SharedLib, &lib_path)
        .map_err(|e| RpcError::internal_error(format!("Compilation failed: {}", e)))?;

    // Load the library
    let lib = unsafe { libloading::Library::new(&lib_path) }
        .map_err(|e| RpcError::internal_error(format!("Failed to load library: {}", e)))?;

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
        .map_err(|e| RpcError::internal_error(format!("Failed to find function {}: {}", func_name, e)))?;

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
            .ok_or_else(|| RpcError::internal_error(format!("Unknown input: {}", name)))?;
        input_ptrs[*idx] = data.as_ptr() as *mut std::ffi::c_void;
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
            return Err(RpcError::internal_error(format!(
                "Output {} was not set by the model",
                target.name
            )));
        }

        let (shape, dtype) = output_info
            .get(&target.id.0)
            .ok_or_else(|| RpcError::internal_error(format!("Could not find output info for {}", target.name)))?;

        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size_in_bytes();

        // Copy data from the allocated buffer
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_size) }.to_vec();

        // Free the buffer allocated by the model
        unsafe {
            libc::free(ptr);
        }

        results.insert(target.name.clone(), (data, shape.clone(), *dtype));
    }

    // Cleanup: drop library first, then delete temp files
    drop(lib);
    let _ = std::fs::remove_dir_all(&temp_dir);

    Ok(results)
}

/// Build a map from tensor id to (shape, dtype) for all outputs
fn build_output_info(snapshot: &Snapshot) -> HashMap<usize, (Vec<usize>, DType)> {
    let mut info = HashMap::new();

    // Add input info
    for input in &snapshot.inputs {
        info.insert(input.id.0, (input.shape.dims().to_vec(), input.dtype));
    }

    // Add constant info
    for constant in &snapshot.constants {
        info.insert(constant.id.0, (constant.shape.dims().to_vec(), constant.dtype));
    }

    // Add node output info
    for node in &snapshot.nodes {
        info.insert(
            node.output_id.0,
            (node.output_layout.shape().dims().to_vec(), node.output_dtype),
        );
    }

    info
}

fn current_target_triple() -> String {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu".to_string();
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu".to_string();
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin".to_string();
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin".to_string();
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc".to_string();
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return String::new();
}
