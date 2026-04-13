# Building gLSF

## Requirements

- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (`acpp` in `PATH`)
- `make`, C++17

## Build

```bash
make          # release (default)
make debug    # with debug symbols
make clean    # remove build artifacts
make run      # build and run with -h
make info     # show build configuration
```

## SYCL Targets

Default is `generic` (auto-detect). Override with `SYCL_TARGETS`:

| Backend | Flag |
|---|---|
| CPU (OpenMP) | `omp` |
| NVIDIA GPU | `cuda` |
| AMD GPU | `hip` |

```bash
make SYCL_TARGETS=cuda
make SYCL_TARGETS=omp BUILD_MODE=debug
```

## Troubleshooting

- **`acpp` not found**: ensure AdaptiveCpp is installed and in `PATH`
- **CUDA/HIP failures**: verify CUDA Toolkit or ROCm is installed
