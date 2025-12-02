//! C code compilation using cc crate

use hodu_cli_plugin_sdk::{BuildFormat, BuildTarget, PluginError, PluginResult};
use std::io::Write as IoWrite;
use std::path::Path;

pub fn compile(c_code: &str, target: &BuildTarget, format: BuildFormat, output: &Path) -> PluginResult<()> {
    let kernels_dir = hodu_cpu_kernels::KERNELS_DIR;

    let out_dir = output.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(out_dir)
        .map_err(|e| PluginError::Execution(format!("Failed to create output dir: {}", e)))?;

    // Set environment variables for cc crate (normally set by cargo)
    std::env::set_var("HOST", &target.triple);
    std::env::set_var("TARGET", &target.triple);
    std::env::set_var("OUT_DIR", out_dir);
    std::env::set_var("OPT_LEVEL", "3");

    let blas_config = BlasConfig::detect(&target.triple);

    let model_c = out_dir.join("model.c");
    std::fs::File::create(&model_c)
        .and_then(|mut f| f.write_all(c_code.as_bytes()))
        .map_err(|e| PluginError::Execution(format!("Failed to write model.c: {}", e)))?;

    match format {
        BuildFormat::SharedLib => compile_shared(kernels_dir, &model_c, target, &blas_config, output),
        BuildFormat::StaticLib => compile_static(kernels_dir, &model_c, out_dir, target, &blas_config, output),
        BuildFormat::Object => compile_object(kernels_dir, &model_c, target, output),
        _ => Err(PluginError::NotSupported(format!("Format {:?} not supported", format))),
    }
}

fn compile_shared(
    kernels_dir: &str,
    model_c: &Path,
    target: &BuildTarget,
    blas_config: &BlasConfig,
    output: &Path,
) -> PluginResult<()> {
    let mut build = cc::Build::new();
    setup_build(&mut build, kernels_dir, model_c, target);

    let compiler = build.get_compiler();
    let mut cmd = compiler.to_command();

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

    if !target.triple.is_empty() {
        cmd.arg("-target").arg(&target.triple);
    }

    cmd.arg("-o").arg(output);

    let status = cmd
        .status()
        .map_err(|e| PluginError::Execution(format!("Compiler failed: {}", e)))?;

    if !status.success() {
        return Err(PluginError::Execution("Compilation failed".to_string()));
    }

    Ok(())
}

fn compile_static(
    kernels_dir: &str,
    model_c: &Path,
    temp_dir: &Path,
    target: &BuildTarget,
    blas_config: &BlasConfig,
    output: &Path,
) -> PluginResult<()> {
    let mut build = cc::Build::new();
    setup_build(&mut build, kernels_dir, model_c, target);

    for src in kernel_sources(kernels_dir, &target.triple) {
        build.file(src);
    }

    for define in &blas_config.defines {
        build.define(define, None);
    }
    for path in &blas_config.include_paths {
        build.include(path);
    }

    build.out_dir(temp_dir);
    build.compile("model");

    let lib_path = temp_dir.join("libmodel.a");
    if lib_path.exists() {
        std::fs::copy(&lib_path, output).map_err(|e| PluginError::Execution(format!("Failed to copy: {}", e)))?;
    }

    Ok(())
}

fn compile_object(kernels_dir: &str, model_c: &Path, target: &BuildTarget, output: &Path) -> PluginResult<()> {
    let mut build = cc::Build::new();
    setup_build(&mut build, kernels_dir, model_c, target);

    let compiler = build.get_compiler();
    let mut cmd = compiler.to_command();

    cmd.arg("-c")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-I")
        .arg(kernels_dir)
        .arg(model_c);

    if !target.triple.is_empty() {
        cmd.arg("-target").arg(&target.triple);
    }

    cmd.arg("-o").arg(output);

    let status = cmd
        .status()
        .map_err(|e| PluginError::Execution(format!("Compiler failed: {}", e)))?;

    if !status.success() {
        return Err(PluginError::Execution("Compilation failed".to_string()));
    }

    Ok(())
}

fn setup_build(build: &mut cc::Build, kernels_dir: &str, model_c: &Path, target: &BuildTarget) {
    build
        .file(model_c)
        .include(kernels_dir)
        .opt_level(3)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .flag_if_supported("-fPIC")
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fno-trapping-math");

    if !target.triple.is_empty() {
        build.target(&target.triple);
    }
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
    fn detect(triple: &str) -> Self {
        if triple.contains("apple-darwin") {
            Self::accelerate()
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

        if let Ok(output) = std::process::Command::new("xcrun").args(["--show-sdk-path"]).output() {
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

    fn apply_to_command(&self, cmd: &mut std::process::Command) {
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
