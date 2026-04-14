// ****************************************************************************
/// @file cuda/kernels.cu
/// @author Kyle Webster
/// @version 0.2
/// @date 13 Apr 2026
/// @brief CUDA kernel implementations for lower-star filtration
// ****************************************************************************

#include "kernels.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace plsf
{
namespace cuda
{

//  Compute-complex kernel (voxel-centric, shared memory) 

/// Tile dimensions for the voxel-centric kernel.
/// Each thread block processes BX×BY×BZ voxels and cooperatively loads a
/// (BX+1)×(BY+1)×(BZ+1) shared-memory tile that includes the +1 halo
/// required for neighbour lookups in the positive direction.
constexpr int CC_BX = 8;
constexpr int CC_BY = 8;
constexpr int CC_BZ = 4;
constexpr int CC_SX = CC_BX + 1;
constexpr int CC_SY = CC_BY + 1;
constexpr int CC_SZ = CC_BZ + 1;

/// @brief Device max helper (works for any ordered type)
template <typename T>
__device__ __forceinline__ T scalar_max (T a, T b)
{
  return a > b ? a : b;
}

/// @brief Convert a filtration value to a sortable uint32 key.
/// @details For float: flips sign bit for positives, all bits for negatives,
///          giving a monotonic mapping from float order to uint32 order.
///          For uint8_t: direct zero-extension already sorts correctly.
template <typename T>
__device__ __forceinline__ uint32_t to_sortable_key (T val);

template <>
__device__ __forceinline__ uint32_t to_sortable_key<float> (float val)
{
  uint32_t f    = __float_as_uint (val);
  uint32_t mask = (-static_cast<int32_t> (f >> 31)) | 0x80000000u;
  return f ^ mask;
}

template <>
__device__ __forceinline__ uint32_t to_sortable_key<uint8_t> (uint8_t val)
{
  return static_cast<uint32_t> (val);
}

/// @brief Kernel: assign each cell its filtration value via vertex-max rule
/// @details One thread per voxel.  Each thread block cooperatively loads a
///          (BX+1)×(BY+1)×(BZ+1) tile of the scalar grid into shared memory,
///          then every thread writes up to 8 cubical-complex cells (vertex,
///          edges, faces, cube) in the "+" neighbourhood of its voxel.
///          Boundary voxels skip cells whose neighbour falls outside the grid.
template <typename scalar>
__global__ void compute_complex_kernel (const scalar *__restrict__ d_grid,
    scalar *__restrict__ d_cube_map, uint64_t Nx, uint64_t Ny, uint64_t Nz,
    uint64_t Mx, uint64_t My, uint64_t MxMy)
{
  __shared__ scalar s_tile[CC_SX * CC_SY * CC_SZ];

  const int tid
      = threadIdx.x + threadIdx.y * CC_BX + threadIdx.z * CC_BX * CC_BY;
  constexpr int BLOCK_THREADS = CC_BX * CC_BY * CC_BZ;
  constexpr int TILE_SIZE     = CC_SX * CC_SY * CC_SZ;

  const uint64_t bx0 = static_cast<uint64_t> (blockIdx.x) * CC_BX;
  const uint64_t by0 = static_cast<uint64_t> (blockIdx.y) * CC_BY;
  const uint64_t bz0 = static_cast<uint64_t> (blockIdx.z) * CC_BZ;

  //  Cooperatively load the tile (including +1 halo) 
  for (int s = tid; s < TILE_SIZE; s += BLOCK_THREADS)
    {
      const int sx = s % CC_SX;
      const int sy = (s / CC_SX) % CC_SY;
      const int sz = s / (CC_SX * CC_SY);

      const uint64_t gx = bx0 + sx;
      const uint64_t gy = by0 + sy;
      const uint64_t gz = bz0 + sz;

      s_tile[s] = (gx < Nx && gy < Ny && gz < Nz)
                      ? d_grid[gx + Nx * gy + Nx * Ny * gz]
                      : scalar (0);
    }
  __syncthreads ();

  //  This thread's voxel 
  const uint64_t vx = bx0 + threadIdx.x;
  const uint64_t vy = by0 + threadIdx.y;
  const uint64_t vz = bz0 + threadIdx.z;

  if (vx >= Nx || vy >= Ny || vz >= Nz)
    return;

  const int lx = threadIdx.x;
  const int ly = threadIdx.y;
  const int lz = threadIdx.z;

#define S(x, y, z) s_tile[(x) + CC_SX * (y) + CC_SX * CC_SY * (z)]

  // Load the voxel and all required neighbours from shared memory
  const scalar v000 = S (lx, ly, lz);

  const bool hx = (vx + 1 < Nx);
  const bool hy = (vy + 1 < Ny);
  const bool hz = (vz + 1 < Nz);

  const scalar v100 = hx               ? S (lx + 1, ly, lz)         : scalar (0);
  const scalar v010 = hy               ? S (lx, ly + 1, lz)         : scalar (0);
  const scalar v001 = hz               ? S (lx, ly, lz + 1)         : scalar (0);
  const scalar v110 = (hx && hy)       ? S (lx + 1, ly + 1, lz)     : scalar (0);
  const scalar v101 = (hx && hz)       ? S (lx + 1, ly, lz + 1)     : scalar (0);
  const scalar v011 = (hy && hz)       ? S (lx, ly + 1, lz + 1)     : scalar (0);
  const scalar v111 = (hx && hy && hz) ? S (lx + 1, ly + 1, lz + 1) : scalar (0);

#undef S

  // Doubled coordinates of this vertex in the cubical complex
  const uint64_t cx = 2 * vx;
  const uint64_t cy = 2 * vy;
  const uint64_t cz = 2 * vz;

#define CUBE(a, b, c) d_cube_map[(a) + Mx * (b) + MxMy * (c)]

  // Vertex (dim 0) 
  CUBE (cx, cy, cz) = v000;

  // Edges (dim 1) 
  if (hx)
    CUBE (cx + 1, cy, cz) = scalar_max (v000, v100);
  if (hy)
    CUBE (cx, cy + 1, cz) = scalar_max (v000, v010);
  if (hz)
    CUBE (cx, cy, cz + 1) = scalar_max (v000, v001);

  // Faces (dim 2) 
  if (hx && hy)
    CUBE (cx + 1, cy + 1, cz)
        = scalar_max (scalar_max (v000, v100), scalar_max (v010, v110));
  if (hx && hz)
    CUBE (cx + 1, cy, cz + 1)
        = scalar_max (scalar_max (v000, v100), scalar_max (v001, v101));
  if (hy && hz)
    CUBE (cx, cy + 1, cz + 1)
        = scalar_max (scalar_max (v000, v010), scalar_max (v001, v011));

  // Cube (dim 3) 
  if (hx && hy && hz)
    CUBE (cx + 1, cy + 1, cz + 1)
        = scalar_max (
            scalar_max (scalar_max (v000, v100), scalar_max (v010, v110)),
            scalar_max (scalar_max (v001, v101), scalar_max (v011, v111)));

#undef CUBE
}

// Explicit template instantiations
template __global__ void compute_complex_kernel<float> (const float *,
    float *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
template __global__ void compute_complex_kernel<uint8_t> (const uint8_t *,
    uint8_t *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

template <typename scalar>
static void launch_compute_complex_impl (const scalar *d_grid,
    scalar *d_cube_map, uint64_t Nx, uint64_t Ny, uint64_t Nz)
{
  const uint64_t Mx   = 2 * Nx - 1;
  const uint64_t My   = 2 * Ny - 1;
  const uint64_t MxMy = Mx * My;

  const dim3 block (CC_BX, CC_BY, CC_BZ);
  const dim3 grid_dim (static_cast<unsigned> ((Nx + CC_BX - 1) / CC_BX),
      static_cast<unsigned> ((Ny + CC_BY - 1) / CC_BY),
      static_cast<unsigned> ((Nz + CC_BZ - 1) / CC_BZ));

  compute_complex_kernel<scalar><<<grid_dim, block>>> (
      d_grid, d_cube_map, Nx, Ny, Nz, Mx, My, MxMy);

  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaDeviceSynchronize ());
}

void launch_compute_complex (const float *d_grid, float *d_cube_map,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t /*n_cells*/)
{
  launch_compute_complex_impl (d_grid, d_cube_map, Nx, Ny, Nz);
}

void launch_compute_complex (const uint8_t *d_grid, uint8_t *d_cube_map,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t /*n_cells*/)
{
  launch_compute_complex_impl (d_grid, d_cube_map, Nx, Ny, Nz);
}
// ****************************************************************************

// Sort-filtration kernel 

/// @brief Kernel: initialise ordering indices to identity permutation
__global__ void init_indices_kernel (
    uint32_t *__restrict__ d_indices, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  d_indices[idx] = static_cast<uint32_t> (idx);
}

/// @brief Kernel: build 32-bit dimension keys (0..3)
/// @details The dimension is derived directly from the cubemap index
///          (count of odd doubled-coordinates) — no separate dims array.
__global__ void compute_dim_keys_kernel (uint32_t *__restrict__ d_keys,
    uint64_t Mx, uint64_t My, uint64_t MxMy, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  const uint64_t ci = idx % Mx;
  const uint64_t cj = (idx / Mx) % My;
  const uint64_t ck = idx / MxMy;

  d_keys[idx] = (ci & 1u) + (cj & 1u) + (ck & 1u);
}

/// @brief Kernel: build sortable 32-bit filtration keys
template <typename scalar>
__global__ void compute_value_keys_kernel (
    const scalar *__restrict__ d_cube_map,
    uint32_t *__restrict__ d_keys, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  d_keys[idx] = to_sortable_key (d_cube_map[idx]);
}

/// @brief Kernel: build composite 32-bit sort key for lossy single-pass sort
/// @details Encodes both filtration value and cell dimension into one key by
///          clearing the bottom 2 bits of the sortable value key and OR-ing
///          in the dimension (0–3).  Cells whose filtration values differ only
///          in those 2 bits (≤3 ULPs apart for float) will be ordered by
///          dimension between them — the intended 2-bit imprecision trade-off.
template <typename scalar>
__global__ void compute_composite_keys_kernel (
    const scalar *__restrict__ d_cube_map,
    uint32_t *__restrict__ d_keys,
    uint64_t Mx, uint64_t My, uint64_t MxMy, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  const uint64_t ci  = idx % Mx;
  const uint64_t cj  = (idx / Mx) % My;
  const uint64_t ck  = idx / MxMy;
  const uint32_t dim = (ci & 1u) + (cj & 1u) + (ck & 1u);

  d_keys[idx] = (to_sortable_key (d_cube_map[idx]) & ~3u) | dim;
}

// Explicit template instantiations
template __global__ void compute_value_keys_kernel<float> (
    const float *, uint32_t *, uint64_t);
template __global__ void compute_value_keys_kernel<uint8_t> (
    const uint8_t *, uint32_t *, uint64_t);
template __global__ void compute_composite_keys_kernel<float> (
    const float *, uint32_t *, uint64_t, uint64_t, uint64_t, uint64_t);

template <typename scalar>
static void launch_sort_filtration_impl (const scalar *d_cube_map,
    uint32_t *d_ordering, uint64_t Nx, uint64_t Ny, uint64_t Nz,
    uint64_t n_cells, bool lossy)
{
  const uint64_t Mx = 2 * Nx - 1;
  const uint64_t My = 2 * Ny - 1;
  const uint64_t MxMy = Mx * My;

  uint32_t *d_keys = nullptr;
  CUDA_CHECK (cudaMalloc (&d_keys, n_cells * sizeof (uint32_t)));

  constexpr int block_size = 256;
  const int     grid_size
      = static_cast<int> ((n_cells + block_size - 1) / block_size);

  init_indices_kernel<<<grid_size, block_size>>> (d_ordering, n_cells);
  CUDA_CHECK (cudaGetLastError ());

  if (lossy)
    {
      // Single-pass: composite key packs the truncated filtration value in
      // bits [31:2] and the cell dimension in bits [1:0].  One non-stable
      // sort is sufficient — the key already encodes the full ordering.
      compute_composite_keys_kernel<scalar><<<grid_size, block_size>>> (
          d_cube_map, d_keys, Mx, My, MxMy, n_cells);
      CUDA_CHECK (cudaGetLastError ());
      CUDA_CHECK (cudaDeviceSynchronize ());

      thrust::device_ptr<uint32_t> keys_ptr (d_keys);
      thrust::device_ptr<uint32_t> vals_ptr (d_ordering);
      thrust::sort_by_key (keys_ptr, keys_ptr + n_cells, vals_ptr);
    }
  else
    {
      // Two-pass: stable sort by dimension, then stable sort by value.
      compute_dim_keys_kernel<<<grid_size, block_size>>> (
          d_keys, Mx, My, MxMy, n_cells);
      CUDA_CHECK (cudaGetLastError ());
      CUDA_CHECK (cudaDeviceSynchronize ());

      thrust::device_ptr<uint32_t> keys_ptr (d_keys);
      thrust::device_ptr<uint32_t> vals_ptr (d_ordering);
      thrust::stable_sort_by_key (keys_ptr, keys_ptr + n_cells, vals_ptr);

      // Primary sort key: filtration value (ascending), stable to preserve
      // dimension ordering on equal filtration values.
      compute_value_keys_kernel<scalar><<<grid_size, block_size>>> (
          d_cube_map, d_keys, n_cells);
      CUDA_CHECK (cudaGetLastError ());
      CUDA_CHECK (cudaDeviceSynchronize ());
      thrust::stable_sort_by_key (keys_ptr, keys_ptr + n_cells, vals_ptr);
    }

  CUDA_CHECK (cudaFree (d_keys));
}

void launch_sort_filtration (const float *d_cube_map, uint32_t *d_ordering,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells, bool lossy)
{
  launch_sort_filtration_impl (d_cube_map, d_ordering, Nx, Ny, Nz, n_cells,
      lossy);
}

void launch_sort_filtration (const uint8_t *d_cube_map, uint32_t *d_ordering,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells, bool /*lossy*/)
{
  // uint8_t (compressed) path: lossy mode is not applicable — always use the
  // two-pass sort so that all 8 bits of the quantised value are respected.
  launch_sort_filtration_impl (d_cube_map, d_ordering, Nx, Ny, Nz, n_cells,
      false);
}

} // namespace cuda
} // namespace plsf
