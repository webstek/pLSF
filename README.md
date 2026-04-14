# pLSF
Parallelized Lower Star Filtration

Computes the lower-star filtration of a 3D cubical complex derived from a
NIfTI scalar image. Each cell is assigned the maximum value of its vertices
(vertex-maximum rule), cells are sorted into a filtration order on the GPU,
and the boundary matrix is constructed directly in PHAT's
`bit_tree_pivot_column` representation. Optionally computes persistence
pairs via PHAT's twist reduction algorithm.

## Usage

```
./bin/plsf [options] <input.nii>

Options:
  -d, --device  <mode>  CUDA device selector: gpu|default  [gpu]
  -o, --output  <stem>  Write output files with the given stem
  -p, --pairs           Compute persistence pairs via phat
                        (default: output boundary matrix only)
  -f, --filtration-only Stop after filtration is computed
                        (skip boundary matrix and output)
  -x, --compress        Use uint8_t filtration values instead of the NIfTI
                        native type to reduce GPU memory pressure  [off]
  -l, --lossy           Encode cell dimension in the 2 LSBs of the sortable
                        float key, enabling a single-pass sort
                        (float/double only; incompatible with --compress)  [off]
  -t, --timings         Report per-step wall time and memory usage  [off]
  -v, --verbose         Print device info and grid statistics  [off]
  -h, --help            Show this message and exit
```

Input must be an uncompressed NIfTI-1/2 file (`.nii`).

### Output modes (require `-o <stem>`)

| Flag | Files produced | Description |
|---|---|---|
| (none) | `<stem>.bin`, `<stem>.vals` | PHAT binary boundary matrix + filtration values |
| `-p` | `<stem>.pairs`, `<stem>.vals` | Persistence pairs (PHAT binary) + filtration values |

## Build

**Requirements:** NVIDIA CUDA Toolkit (for `nvcc`), `make`, C++17 toolchain (`g++`)

The [PHAT](https://github.com/blazs/phat) library is included as a Git
submodule at `lib/phat` (headers under `lib/phat/include`). After cloning,
initialise it with:

```bash
git submodule update --init --recursive
```

```bash
make                  # release build (default)
make release          # release build (explicit)
make debug            # build with debug symbols
make clean            # remove build artifacts
make run              # build and run with default settings
make info             # print resolved build configuration
make CUDA_ARCH=sm_XX  # target a specific GPU architecture
make help             # prints available targets and variables
```

The host boundary-matrix step uses **OpenMP** automatically when `g++` is present; no extra flags are required.

### CubicalRipser (benchmark dependency)

The benchmark script requires the `cubicalripser` binary, built from the
`lib/CubicalRipser` submodule:

```bash
cmake -S lib/CubicalRipser -B lib/CubicalRipser/build
cmake --build lib/CubicalRipser/build
```

The compiled binary will be placed at `lib/CubicalRipser/build/cubicalripser`.

## Benchmark

`examples/benchmark.py` compares the filtration construction and sorting
phases of pLSF (GPU) against Cubical Ripser (CPU) on a 3-D NIfTI volume.
Neither pipeline performs persistent-homology reduction; only the complex
enumeration and sort times are measured.

**Python dependencies:** `pip install nibabel numpy`

```
python examples/benchmark.py <input.nii> [options]

Options:
  --plsf PATH           Path to plsf binary         [bin/plsf]
  --cubicalripser PATH   Path to cubicalripser binary [lib/CubicalRipser/build/cubicalripser]
  --device {gpu,default} CUDA device selector for plsf [gpu]
  --maxdim N            Max cell dimension for Cubical Ripser [3]
  --skip-plsf           Run Cubical Ripser only
  --skip-cripser        Run pLSF only
  -v, --verbose         Print raw subprocess output
```

Sample data files are provided in `examples/data/`.

```bash
# compare both pipelines on the bundled 1 mm atlas
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii

# pLSF only (no CUDA Ripser needed)
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii --skip-cripser

# use CUDA default device selector and show per-step output
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii --device default -v
```
