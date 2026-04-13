# pLSF
Parallelized Lower Star Filtration

Computes the lower-star filtration of a 3D cubical complex derived from a
NIfTI scalar image. Each cell is assigned the maximum value of its vertices
(vertex-maximum rule), cells are sorted into a filtration order on the GPU,
and the boundary matrix is constructed directly in PHAT's
`bit_tree_pivot_column` representation. Optionally computes persistence
pairs via PHAT's twist reduction algorithm.

## Build

**Requirements:** [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (`acpp` in `PATH`), `make`, C++17

The [PHAT](https://github.com/blazs/phat) library is included as a Git
submodule.  After cloning, initialise it with:

```bash
git submodule update --init --recursive
```

Then build normally:

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
  -o, --output  <stem>  Write output files with the given stem
  -p, --pairs           Compute persistence pairs via phat
                        (default: output boundary matrix only)
  -t, --timings         Report per-step wall time and memory usage
  -v, --verbose         Print device info and grid statistics
  -h, --help            Show this message and exit
```

Input must be an uncompressed NIfTI-1/2 file (`.nii`).

### Output modes (require `-o <stem>`)

| Flag | Files produced | Description |
|---|---|---|
| (none) | `<stem>.bin`, `<stem>.vals` | PHAT binary boundary matrix + filtration values |
| `-p` | `<stem>.pairs`, `<stem>.vals` | Persistence pairs (PHAT binary) + filtration values |
