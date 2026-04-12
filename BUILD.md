# Building gLSF with AdaptiveCpp

This project uses a **Makefile** with **AdaptiveCpp** (hip SYCL) for vendor-agnostic GPU support.

## Prerequisites

- **AdaptiveCpp** (formerly hipSYCL): https://github.com/AdaptiveCpp/AdaptiveCpp
  - On Linux: `sudo apt install adaptivecpp`
  - On macOS: `brew install adaptivecpp`
  - On Windows: Download from releases or build from source
  
- **Standard build tools**: `make`, C++17 compiler (clang++)

## Quick Start

### Release (optimized) build:
```bash
make
# or explicitly:
make release
```

### Debug build (with symbols):
```bash
make debug
```

### Clean build artifacts:
```bash
make clean
```

### Run the executable:
```bash
make run                          # Show help message
./bin/glsf -v data/sample.nii     # Run with verbose output
```

### View configuration:
```bash
make info
```

## Customization

### Target a specific accelerator

The default (`native`) auto-detects available hardware. To target specific backends:

**OpenMP (CPU only):**
```bash
make SYCL_TARGETS=omp
```

**NVIDIA GPU (CUDA):**
```bash
make SYCL_TARGETS=cuda:sm_70
```

**AMD GPU (HIP):**
```bash
make SYCL_TARGETS=hip:gfx906
```

**CPU (explicit):**
```bash
make SYCL_TARGETS=omp
```

### Combining options:

```bash
make SYCL_TARGETS=cuda:sm_70 BUILD_MODE=debug
```

## Compiler Information

The Makefile uses `syclcc`, which is part of the AdaptiveCpp distribution:
- Wrapper around Clang++ with SYCL support and kernel offloading
- Automatically handles device code compilation and linking
- Supports all backends AdaptiveCpp provides

## Troubleshooting

**"syclcc not found"**: Ensure AdaptiveCpp is installed and in your `PATH`.
```bash
which syclcc
syclcc --version
```

**Build failures with specific targets**: Verify the backend tools are installed (e.g., CUDA Toolkit for `cuda:*`, ROCm for `hip:*`).

**Debug builds are slow**: Use `make release` for performance testing.

---

For more info, see `make help`.
