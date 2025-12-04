# hodu-backend-aot-cpu-plugin

AOT (Ahead-of-Time) CPU backend plugin for Hodu. Compiles Hodu Snapshots to native code via C code generation.

## Features

- **Run**: JIT compile and execute models on CPU
- **Build**: AOT compile models to native libraries (shared lib, static lib, object file)
- **Cross-compilation**: Support for multiple target architectures
- **Embedded targets**: ARM Cortex-M, Cortex-A, and RISC-V support

## Supported Targets

### Desktop/Server
- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`

### Embedded - ARM Cortex-M
- `thumbv6m-none-eabi` (Cortex-M0, M0+)
- `thumbv7m-none-eabi` (Cortex-M3)
- `thumbv7em-none-eabi` (Cortex-M4, M7 without FPU)
- `thumbv7em-none-eabihf` (Cortex-M4F, M7F with FPU)

### Embedded - ARM Cortex-A
- `aarch64-unknown-none`
- `armv7a-none-eabi`

### Embedded - RISC-V
- `riscv32imac-unknown-none-elf`
- `riscv64gc-unknown-none-elf`

## Installation

```bash
hodu plugin install --path /path/to/hodu-backend-aot-cpu-plugin
```

## Usage

### Run inference

```bash
hodu run model.hdss -i input=data.hdt --backend aot-cpu
```

### Build to native binary

```bash
# Standalone executable
hodu build model.hdss -o model

# Shared library
hodu build model.hdss -o model.so

# Static library
hodu build model.hdss -o model.a -f staticlib

# Object file
hodu build model.hdss -o model.o -f object

# Cross-compile for embedded target
hodu build model.hdss -o model.o -t thumbv7em-none-eabihf -f object
```

### List supported targets

```bash
hodu build --backend aot-cpu --list-targets
```

## Cross-Compilation

The plugin automatically detects available cross-compilers in the following order:

1. **Zig CC** (`zig cc`): Recommended for cross-compilation
2. **Target-specific toolchain**: e.g., `aarch64-linux-gnu-gcc`
3. **Clang with target flag**: `clang --target=<triple>`

For embedded targets, the plugin automatically adds appropriate compiler flags:
- `-ffreestanding`
- `-nostdlib`
- `-fno-exceptions`

## Output Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| `executable` | (none), `.exe`, `.bin` | Standalone executable binary |
| `sharedlib` | `.so`, `.dylib`, `.dll` | Shared/dynamic library |
| `staticlib` | `.a`, `.lib` | Static library |
| `object` | `.o`, `.obj` | Object file |

## Using Compiled Library

The compiled library exports a `<model_name>_run` function. Example usage in C:

```c
#include <stdio.h>
#include <stdlib.h>

// Function signature exported by compiled model
extern void model_run(void** inputs, void** outputs);

int main() {
    // Prepare input tensor (example: float[2,3])
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    void* inputs[] = {input_data};

    // Output will be allocated by the model
    void* outputs[1] = {NULL};

    // Run inference
    model_run(inputs, outputs);

    // Use output (remember to free after use)
    float* output = (float*)outputs[0];
    printf("Output: %f\n", output[0]);
    free(output);

    return 0;
}
```

Build with the compiled library:

```bash
# Build model to shared library
hodu build model.hdss -o libmodel.so

# Compile and link your application
gcc main.c -L. -lmodel -o app
LD_LIBRARY_PATH=. ./app
```

Or with static library:

```bash
# Build model to static library
hodu build model.hdss -o libmodel.a

# Compile and link statically
gcc main.c libmodel.a -o app -lm
./app
```

## License

BSD-3-Clause
