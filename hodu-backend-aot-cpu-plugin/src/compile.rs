//! C code compilation using cc crate with cross-compilation support

use hodu_plugin_sdk::{PluginError, PluginResult};
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy)]
pub enum BuildFormat {
    Executable,
    SharedLib,
    StaticLib,
    Object,
}

#[derive(Debug, Clone)]
pub struct BuildTarget {
    pub triple: String,
    #[allow(dead_code)]
    pub device: String,
}

/// Supported cross-compilation targets
pub const SUPPORTED_TARGETS: &[&str] = &[
    // Desktop/Server
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
    // Embedded - ARM Cortex-M
    "thumbv6m-none-eabi",    // Cortex-M0, M0+
    "thumbv7m-none-eabi",    // Cortex-M3
    "thumbv7em-none-eabi",   // Cortex-M4, M7 (no FPU)
    "thumbv7em-none-eabihf", // Cortex-M4F, M7F (with FPU)
    // Embedded - ARM Cortex-A (bare metal)
    "aarch64-unknown-none",
    "armv7a-none-eabi",
    // Embedded - RISC-V
    "riscv32imac-unknown-none-elf",
    "riscv64gc-unknown-none-elf",
];

/// Check if a target is supported
pub fn is_supported_target(target: &str) -> bool {
    SUPPORTED_TARGETS.contains(&target)
}

/// Get list of supported targets as formatted string
pub fn list_supported_targets() -> String {
    let mut result = String::new();
    result.push_str("Desktop/Server:\n");
    for t in SUPPORTED_TARGETS.iter().take(5) {
        result.push_str(&format!("  {}\n", t));
    }
    result.push_str("\nEmbedded - ARM Cortex-M:\n");
    for t in SUPPORTED_TARGETS.iter().skip(5).take(4) {
        result.push_str(&format!("  {}\n", t));
    }
    result.push_str("\nEmbedded - ARM Cortex-A:\n");
    for t in SUPPORTED_TARGETS.iter().skip(9).take(2) {
        result.push_str(&format!("  {}\n", t));
    }
    result.push_str("\nEmbedded - RISC-V:\n");
    for t in SUPPORTED_TARGETS.iter().skip(11) {
        result.push_str(&format!("  {}\n", t));
    }
    result
}

pub fn compile(c_code: &str, target: &BuildTarget, format: BuildFormat, output: &Path) -> PluginResult<()> {
    // Validate target
    if !target.triple.is_empty() && !is_supported_target(&target.triple) {
        return Err(PluginError::NotSupported(format!(
            "Unsupported target: '{}'\n\nSupported targets:\n{}",
            target.triple,
            list_supported_targets()
        )));
    }

    let kernels_dir = hodu_cpu_kernels::KERNELS_DIR;

    let out_dir = output.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(out_dir)
        .map_err(|e| PluginError::Execution(format!("Failed to create output dir: {}", e)))?;

    // Set environment variables for cc crate (normally set by cargo)
    std::env::set_var("HOST", current_host_triple());
    std::env::set_var("TARGET", &target.triple);
    std::env::set_var("OUT_DIR", out_dir);
    std::env::set_var("OPT_LEVEL", "3");

    let cross_compiler = CrossCompiler::detect(&target.triple);
    let blas_config = BlasConfig::detect(&target.triple, &cross_compiler);

    let model_c = out_dir.join("model.c");
    std::fs::File::create(&model_c)
        .and_then(|mut f| f.write_all(c_code.as_bytes()))
        .map_err(|e| PluginError::Execution(format!("Failed to write model.c: {}", e)))?;

    match format {
        BuildFormat::Executable => {
            compile_executable(kernels_dir, &model_c, target, &cross_compiler, &blas_config, output)
        },
        BuildFormat::SharedLib => compile_shared(kernels_dir, &model_c, target, &cross_compiler, &blas_config, output),
        BuildFormat::StaticLib => compile_static(
            kernels_dir,
            &model_c,
            out_dir,
            target,
            &cross_compiler,
            &blas_config,
            output,
        ),
        BuildFormat::Object => compile_object(kernels_dir, &model_c, target, &cross_compiler, output),
    }
}

fn current_host_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu";
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin";
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin";
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return "unknown";
}

/// Cross-compiler configuration
#[derive(Debug, Clone)]
struct CrossCompiler {
    compiler: String,
    is_zig: bool,
    is_cross: bool,
    zig_target: Option<String>,
}

impl CrossCompiler {
    fn detect(target_triple: &str) -> Self {
        let host = current_host_triple();
        let is_cross = !target_triple.is_empty() && target_triple != host;

        // Check if zig is available for cross-compilation
        if is_cross && Command::new("zig").arg("version").output().is_ok() {
            if let Some(zig_target) = triple_to_zig_target(target_triple) {
                return Self {
                    compiler: "zig".to_string(),
                    is_zig: true,
                    is_cross: true,
                    zig_target: Some(zig_target),
                };
            }
        }

        // Check for target-specific cross-compiler
        if is_cross {
            if let Some(cc) = find_cross_compiler(target_triple) {
                return Self {
                    compiler: cc,
                    is_zig: false,
                    is_cross: true,
                    zig_target: None,
                };
            }
        }

        // Fall back to clang with -target
        Self {
            compiler: "clang".to_string(),
            is_zig: false,
            is_cross,
            zig_target: None,
        }
    }

    fn to_command(&self, target_triple: &str) -> Command {
        if self.is_zig {
            let mut cmd = Command::new("zig");
            cmd.arg("cc");
            if let Some(ref zig_target) = self.zig_target {
                cmd.arg(format!("-target={}", zig_target));
            }
            cmd
        } else {
            let mut cmd = Command::new(&self.compiler);
            if self.is_cross && !target_triple.is_empty() {
                cmd.arg("-target").arg(target_triple);
            }
            cmd
        }
    }
}

fn triple_to_zig_target(triple: &str) -> Option<String> {
    match triple {
        // Desktop/Server
        "x86_64-unknown-linux-gnu" => Some("x86_64-linux-gnu".to_string()),
        "aarch64-unknown-linux-gnu" => Some("aarch64-linux-gnu".to_string()),
        "x86_64-unknown-linux-musl" => Some("x86_64-linux-musl".to_string()),
        "aarch64-unknown-linux-musl" => Some("aarch64-linux-musl".to_string()),
        "x86_64-apple-darwin" => Some("x86_64-macos".to_string()),
        "aarch64-apple-darwin" => Some("aarch64-macos".to_string()),
        // Embedded - ARM Cortex-M
        "thumbv6m-none-eabi" => Some("thumb-freestanding-none".to_string()),
        "thumbv7m-none-eabi" => Some("thumb-freestanding-none".to_string()),
        "thumbv7em-none-eabi" => Some("thumb-freestanding-none".to_string()),
        "thumbv7em-none-eabihf" => Some("thumb-freestanding-none".to_string()),
        // Embedded - ARM Cortex-A
        "aarch64-unknown-none" => Some("aarch64-freestanding-none".to_string()),
        "armv7a-none-eabi" => Some("arm-freestanding-none".to_string()),
        // Embedded - RISC-V
        "riscv32imac-unknown-none-elf" => Some("riscv32-freestanding-none".to_string()),
        "riscv64gc-unknown-none-elf" => Some("riscv64-freestanding-none".to_string()),
        _ => None,
    }
}

fn find_cross_compiler(target_triple: &str) -> Option<String> {
    let prefixes: Vec<&str> = match target_triple {
        // Desktop/Server Linux
        t if t.contains("x86_64") && t.contains("linux") => {
            vec!["x86_64-linux-gnu-", "x86_64-unknown-linux-gnu-"]
        },
        t if t.contains("aarch64") && t.contains("linux") => {
            vec!["aarch64-linux-gnu-", "aarch64-unknown-linux-gnu-"]
        },
        // Embedded - ARM
        t if t.starts_with("thumb") || t.contains("arm") => {
            vec!["arm-none-eabi-", "arm-unknown-none-eabi-"]
        },
        t if t.contains("aarch64") && t.contains("none") => {
            vec!["aarch64-none-elf-", "aarch64-unknown-none-"]
        },
        // Embedded - RISC-V
        t if t.contains("riscv32") => {
            vec!["riscv32-unknown-elf-", "riscv32-none-elf-"]
        },
        t if t.contains("riscv64") => {
            vec!["riscv64-unknown-elf-", "riscv64-none-elf-"]
        },
        _ => vec![],
    };

    for prefix in prefixes {
        let cc = format!("{}gcc", prefix);
        if Command::new(&cc).arg("--version").output().is_ok() {
            return Some(cc);
        }
        let cc = format!("{}clang", prefix);
        if Command::new(&cc).arg("--version").output().is_ok() {
            return Some(cc);
        }
    }

    None
}

/// Check if target is an embedded (bare metal) target
fn is_embedded_target(triple: &str) -> bool {
    triple.contains("none") || triple.contains("eabi") && !triple.contains("linux")
}

fn compile_executable(
    kernels_dir: &str,
    model_c: &Path,
    target: &BuildTarget,
    cross: &CrossCompiler,
    blas_config: &BlasConfig,
    output: &Path,
) -> PluginResult<()> {
    // Embedded targets need special handling
    if is_embedded_target(&target.triple) {
        return Err(PluginError::NotSupported(
            "Standalone executables not supported for embedded targets. Use object or static library format."
                .to_string(),
        ));
    }

    let mut cmd = cross.to_command(&target.triple);

    cmd.arg("-O3").arg("-I").arg(kernels_dir).arg(model_c);

    for src in kernel_sources(kernels_dir, &target.triple) {
        cmd.arg(&src);
    }

    blas_config.apply_to_command(&mut cmd);

    // Link math library
    cmd.arg("-lm");

    cmd.arg("-o").arg(output);

    let output_result = cmd
        .output()
        .map_err(|e| PluginError::Execution(format!("Compiler failed to start: {}", e)))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(PluginError::Execution(format!("Compilation failed: {}", stderr)));
    }

    Ok(())
}

fn compile_shared(
    kernels_dir: &str,
    model_c: &Path,
    target: &BuildTarget,
    cross: &CrossCompiler,
    blas_config: &BlasConfig,
    output: &Path,
) -> PluginResult<()> {
    // Embedded targets don't support shared libraries
    if is_embedded_target(&target.triple) {
        return Err(PluginError::NotSupported(
            "Shared libraries not supported for embedded targets. Use static library or object format.".to_string(),
        ));
    }

    let mut cmd = cross.to_command(&target.triple);

    cmd.arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-I")
        .arg(kernels_dir)
        .arg(model_c);

    for src in kernel_sources(kernels_dir, &target.triple) {
        cmd.arg(&src);
    }

    blas_config.apply_to_command(&mut cmd);

    cmd.arg("-o").arg(output);

    let output_result = cmd
        .output()
        .map_err(|e| PluginError::Execution(format!("Compiler failed to start: {}", e)))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(PluginError::Execution(format!("Compilation failed: {}", stderr)));
    }

    Ok(())
}

fn compile_static(
    kernels_dir: &str,
    model_c: &Path,
    temp_dir: &Path,
    target: &BuildTarget,
    cross: &CrossCompiler,
    blas_config: &BlasConfig,
    output: &Path,
) -> PluginResult<()> {
    let is_embedded = is_embedded_target(&target.triple);

    // Compile all source files to object files
    let mut obj_files = Vec::new();

    let sources: Vec<_> = std::iter::once(model_c.to_path_buf())
        .chain(
            kernel_sources(kernels_dir, &target.triple)
                .into_iter()
                .map(|s| s.into()),
        )
        .collect();

    for (i, src) in sources.iter().enumerate() {
        let obj = temp_dir.join(format!("obj_{}.o", i));
        let mut cmd = cross.to_command(&target.triple);
        cmd.arg("-c").arg("-O3").arg("-I").arg(kernels_dir);

        // Embedded-specific flags
        if is_embedded {
            cmd.arg("-ffreestanding").arg("-nostdlib").arg("-fno-exceptions");
        } else {
            cmd.arg("-fPIC");
        }

        for define in &blas_config.defines {
            cmd.arg(format!("-D{}", define));
        }
        for path in &blas_config.include_paths {
            cmd.arg("-I").arg(path);
        }

        cmd.arg(src).arg("-o").arg(&obj);

        let output_result = cmd
            .output()
            .map_err(|e| PluginError::Execution(format!("Compiler failed: {}", e)))?;
        if !output_result.status.success() {
            let stderr = String::from_utf8_lossy(&output_result.stderr);
            return Err(PluginError::Execution(format!("Compilation failed: {}", stderr)));
        }
        obj_files.push(obj);
    }

    // Create static library with ar
    let ar = if cross.is_zig { "zig" } else { "ar" };
    let mut ar_cmd = Command::new(ar);
    if cross.is_zig {
        ar_cmd.arg("ar");
    }
    ar_cmd.arg("rcs").arg(output);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }

    let output_result = ar_cmd
        .output()
        .map_err(|e| PluginError::Execution(format!("ar failed: {}", e)))?;
    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(PluginError::Execution(format!("ar failed: {}", stderr)));
    }

    Ok(())
}

fn compile_object(
    kernels_dir: &str,
    model_c: &Path,
    target: &BuildTarget,
    cross: &CrossCompiler,
    output: &Path,
) -> PluginResult<()> {
    let is_embedded = is_embedded_target(&target.triple);
    let mut cmd = cross.to_command(&target.triple);

    cmd.arg("-c").arg("-O3").arg("-I").arg(kernels_dir);

    // Embedded-specific flags
    if is_embedded {
        cmd.arg("-ffreestanding").arg("-nostdlib").arg("-fno-exceptions");
    } else {
        cmd.arg("-fPIC");
    }

    cmd.arg(model_c).arg("-o").arg(output);

    let output_result = cmd
        .output()
        .map_err(|e| PluginError::Execution(format!("Compiler failed: {}", e)))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(PluginError::Execution(format!("Compilation failed: {}", stderr)));
    }

    Ok(())
}

fn kernel_sources(kernels_dir: &str, triple: &str) -> Vec<String> {
    let mut sources = vec![
        format!("{}/ops_binary.c", kernels_dir),
        format!("{}/ops_cast.c", kernels_dir),
        format!("{}/ops_concat_split.c", kernels_dir),
        format!("{}/ops_conv.c", kernels_dir),
        format!("{}/ops_indexing.c", kernels_dir),
        format!("{}/ops_matrix.c", kernels_dir),
        format!("{}/ops_memory.c", kernels_dir),
        format!("{}/ops_reduce.c", kernels_dir),
        format!("{}/ops_unary.c", kernels_dir),
        format!("{}/ops_windowing.c", kernels_dir),
    ];

    // Add BLAS-optimized sources for supported targets
    if triple.contains("apple-darwin") {
        sources.push(format!("{}/ops_conv_blas_aarch64_apple_darwin.c", kernels_dir));
        sources.push(format!("{}/ops_matrix_blas_aarch64_apple_darwin.c", kernels_dir));
        sources.push(format!("{}/ops_unary_blas_aarch64_apple_darwin.c", kernels_dir));
    }

    sources
}

#[derive(Debug, Clone, Default)]
struct BlasConfig {
    defines: Vec<String>,
    include_paths: Vec<String>,
    frameworks: Vec<String>,
}

impl BlasConfig {
    fn detect(triple: &str, cross: &CrossCompiler) -> Self {
        // Only use Accelerate on macOS when not cross-compiling to non-Darwin targets
        if triple.contains("apple-darwin") && !cross.is_cross {
            Self::accelerate()
        } else if triple.contains("apple-darwin") && cross.is_cross {
            // Cross-compiling to macOS - still try Accelerate but it may not work
            Self::accelerate_cross()
        } else {
            Self::default()
        }
    }

    fn accelerate() -> Self {
        let mut config = Self {
            defines: vec!["USE_BLAS".into(), "ACCELERATE_NEW_LAPACK".into()],
            frameworks: vec!["Accelerate".into()],
            ..Default::default()
        };

        if let Ok(output) = Command::new("xcrun").args(["--show-sdk-path"]).output() {
            if output.status.success() {
                let sdk_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                config.include_paths.push(format!(
                    "{}/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers",
                    sdk_path
                ));
            }
        }

        config
    }

    fn accelerate_cross() -> Self {
        // For cross-compilation to macOS, we need the SDK
        // User should set SDKROOT environment variable
        if let Ok(sdk_path) = std::env::var("SDKROOT") {
            Self {
                defines: vec!["USE_BLAS".into(), "ACCELERATE_NEW_LAPACK".into()],
                include_paths: vec![format!(
                    "{}/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers",
                    sdk_path
                )],
                frameworks: vec!["Accelerate".into()],
            }
        } else {
            // Fall back to no BLAS for cross-compilation without SDK
            Self::default()
        }
    }

    fn apply_to_command(&self, cmd: &mut Command) {
        for define in &self.defines {
            cmd.arg(format!("-D{}", define));
        }
        for path in &self.include_paths {
            cmd.arg("-I").arg(path);
        }
        for framework in &self.frameworks {
            cmd.arg("-framework").arg(framework);
        }
    }
}
