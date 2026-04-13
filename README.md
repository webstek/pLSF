# pLSF
Parallelized Lower Star Filtration

Computes the lower-star filtration of a 3D cubical complex derived from a
NIfTI scalar image. Each cell is assigned the maximum value of its vertices
(vertex-maximum rule), cells are sorted into a filtration order on the GPU,
and the boundary matrix is constructed directly in PHAT's
`bit_tree_pivot_column` representation. Optionally computes persistence
pairs via PHAT's twist reduction algorithm.

## Build

**Requirements:** NVIDIA CUDA Toolkit (for `nvcc`), `make`, C++17 toolchain (`g++`)

The [PHAT](https://github.com/blazs/phat) library is included as a Git
submodule at `lib/phat` (headers under `lib/phat/include`). After cloning,
initialise it with:

```bash
git submodule update --init --recursive
```

Then build normally:

```bash
make          # release (default)
make debug    # with debug symbols
make clean
```

Target a specific GPU architecture when needed:

```bash
make CUDA_ARCH=sm_80
```

**Troubleshooting:**
- **`nvcc` not found**: ensure CUDA Toolkit is installed and `nvcc` is in `PATH`
- **Submodule issues**: run `git submodule sync --recursive && git submodule update --init --recursive`
- **CUDA build failures**: verify driver/toolkit compatibility for your GPU

## Usage

```
./bin/plsf [options] <input.nii>

Options:
  -d, --device  <mode>  CUDA device selector: gpu|default  [gpu]
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
