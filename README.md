# pLSF
Parallelized Lower Star Filtration

Computes the lower-star filtration of a 3D cubical complex derived from a
NIfTI scalar image. Each cell is assigned the maximum value of its vertices
(vertex-maximum rule), cells are sorted into a filtration order on the GPU,
and the resulting boundary matrix is written in PHAT binary format for downstream
persistent homology computation.

## Build

**Requirements:** [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (`acpp` in `PATH`), `make`, C++17

```bash
make          # release (default)
make debug    # with debug symbols
make clean
```

Target a specific backend with `SYCL_TARGETS` (`generic` by default):

| Backend | Flag |
|---|---|
| CPU (OpenMP) | `omp` |
| NVIDIA GPU | `cuda` |
| AMD GPU | `hip` |

```bash
make SYCL_TARGETS=cuda
make SYCL_TARGETS=omp BUILD_MODE=debug
```

**Troubleshooting:**
- **`acpp` not found**: ensure AdaptiveCpp is installed and in `PATH`
- **CUDA/HIP failures**: verify CUDA Toolkit or ROCm is installed

## Usage

```
./bin/plsf [options] <input.nii>

Options:
  -d, --device  <mode>  SYCL device selector: cpu|gpu|default  [gpu]
  -o, --output  <stem>  Write boundary matrix to <stem>.bin and
                        filtration values to <stem>.vals
  -t, --timings         Report per-step wall time and memory usage
  -v, --verbose         Print device info and grid statistics
  -h, --help            Show this message and exit
```

Input must be an uncompressed NIfTI-1/2 file (`.nii`).
