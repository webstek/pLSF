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

### DIPHA (benchmark dependency)

The benchmark script can optionally compare against
[DIPHA](https://github.com/webstek/dipha) (Distributed Persistent Homology
Algorithm). This fork includes a `--filtration-only` mode that computes the
filtration ordering and prints machine-parseable timing lines. DIPHA is
included as a Git submodule at `lib/dipha`.

**Requirements:** CMake, MPI (`libopenmpi-dev` or equivalent)

```bash
cmake -S lib/dipha -B lib/dipha/build
cmake --build lib/dipha/build
```

The compiled binary will be placed at `lib/dipha/build/dipha`.

## Benchmark

`examples/benchmark.py` compares the filtration construction time of pLSF
(GPU), Cubical Ripser (CPU), GUDHI (CPU), and DIPHA (CPU/MPI) on 3-D NIfTI
volumes. It accepts one or more `.nii` files and produces a summary table
per file plus an optional plot of filtration time vs number of cells.

**Python dependencies:** `pip install nibabel numpy gudhi matplotlib`

```
python examples/benchmark.py <input.nii> [input2.nii ...] [options]

Options:
  --plsf PATH            Path to plsf binary          [bin/plsf]
  --cubicalripser PATH   Path to cubicalripser binary  [lib/CubicalRipser/build/cubicalripser]
  --dipha PATH           Path to dipha binary          [lib/dipha/build/dipha]
  --device {gpu,default} CUDA device selector for plsf [gpu]
  --maxdim N             Max cell dimension for Cubical Ripser [3]
  --dipha-nodes N        Number of MPI processes for DIPHA [1]
  --skip-plsf            Skip the pLSF pipeline
  --skip-cripser         Skip the Cubical Ripser pipeline
  --skip-gudhi           Skip the GUDHI pipeline
  --skip-dipha           Skip the DIPHA pipeline
  --plot PATH            Save time-vs-cells plot (e.g. filtration.png)
  -x, --compress         Pass -x (uint8 mode) to pLSF
  -l, --lossy            Pass -l (lossy dim encoding) to pLSF
  -v, --verbose          Print raw subprocess output
```

If a tool errors on a particular file (e.g. out of memory), the remaining
tools still run.

Sample data files are provided in `examples/data/`.

```bash
# compare all four tools on the bundled 1 mm atlas
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii

# multiple files with a scaling plot
python examples/benchmark.py examples/data/*.nii --plot filtration.png

# pLSF only (no other tools needed)
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii \
    --skip-cripser --skip-gudhi --skip-dipha

# run DIPHA across 4 MPI processes
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii \
    --skip-plsf --skip-cripser --skip-gudhi --dipha-nodes 4

# use CUDA default device selector and show per-step output
python examples/benchmark.py examples/data/full_cls_1000um_2009b_sym.nii --device default -v
```
