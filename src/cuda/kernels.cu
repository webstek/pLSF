// ****************************************************************************
/// @file cuda/kernels.cu
/// @author Kyle Webster
/// @version 0.1
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

// ── Compute-complex kernel ──────────────────────────────────────────────────

/// @brief Kernel: assign each cell its filtration value via vertex-max rule
/// @details Each thread processes one cell in the doubled-coordinate cubical
///          complex.  For a cell at doubled coordinates (ci, cj, ck):
///            even coord → single vertex index  (ci/2)
///            odd  coord → two vertex indices   (ci/2, ci/2+1)
///          The cell's value is the maximum over the 1–8 contributing vertices.
__global__ void compute_complex_kernel (const float *__restrict__ d_grid,
    float *__restrict__ d_cube_map, uint64_t Nx, uint64_t Ny, uint64_t Nz,
    uint64_t Mx, uint64_t My, uint64_t MxMy, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  const uint64_t ci = idx % Mx;
  const uint64_t cj = (idx / Mx) % My;
  const uint64_t ck = idx / MxMy;

  const uint64_t vx0 = ci >> 1;
  const uint64_t vx1 = vx0 + (ci & 1u);
  const uint64_t vy0 = cj >> 1;
  const uint64_t vy1 = vy0 + (cj & 1u);
  const uint64_t vz0 = ck >> 1;
  const uint64_t vz1 = vz0 + (ck & 1u);

  float max_val = d_grid[vx0 + Nx * vy0 + Nx * Ny * vz0];
  for (uint64_t vz = vz0; vz <= vz1; ++vz)
    for (uint64_t vy = vy0; vy <= vy1; ++vy)
      for (uint64_t vx = vx0; vx <= vx1; ++vx)
        {
          const float v = d_grid[vx + Nx * vy + Nx * Ny * vz];
          if (v > max_val)
            max_val = v;
        }

  d_cube_map[idx] = max_val;
}

void launch_compute_complex (const float *d_grid, float *d_cube_map,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells)
{
  const uint64_t Mx = 2 * Nx - 1;
  const uint64_t My = 2 * Ny - 1;
  const uint64_t MxMy = Mx * My;

  constexpr int block_size = 256;
  const int     grid_size
      = static_cast<int> ((n_cells + block_size - 1) / block_size);

  compute_complex_kernel<<<grid_size, block_size>>> (
      d_grid, d_cube_map, Nx, Ny, Nz, Mx, My, MxMy, n_cells);

  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaDeviceSynchronize ());
}

// ── Sort-filtration kernel ──────────────────────────────────────────────────

/// @brief Convert a float to a uint32 that sorts in the same order
/// @details Flips sign bit for positive floats, all bits for negative, giving
///          a monotonic mapping from float comparison order to uint32 order.
__device__ __forceinline__ uint32_t float_to_sortable (float val)
{
  uint32_t f = __float_as_uint (val);
  uint32_t mask = (-static_cast<int32_t> (f >> 31)) | 0x80000000u;
  return f ^ mask;
}

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
__global__ void compute_value_keys_kernel (const float *__restrict__ d_cube_map,
    uint32_t *__restrict__ d_keys, uint64_t n_cells)
{
  const uint64_t idx
      = static_cast<uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;

  d_keys[idx] = float_to_sortable (d_cube_map[idx]);
}

void launch_sort_filtration (const float *d_cube_map, uint32_t *d_ordering,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells)
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

  // Secondary sort key: dimension (ascending)
  compute_dim_keys_kernel<<<grid_size, block_size>>> (
      d_keys, Mx, My, MxMy, n_cells);
  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaDeviceSynchronize ());

  thrust::device_ptr<uint32_t> keys_ptr (d_keys);
  thrust::device_ptr<uint32_t> vals_ptr (d_ordering);
  thrust::stable_sort_by_key (keys_ptr, keys_ptr + n_cells, vals_ptr);

  // Primary sort key: filtration value (ascending), stable to preserve
  // dimension ordering on equal filtration values.
  compute_value_keys_kernel<<<grid_size, block_size>>> (
      d_cube_map, d_keys, n_cells);
  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaDeviceSynchronize ());
  thrust::stable_sort_by_key (keys_ptr, keys_ptr + n_cells, vals_ptr);

  CUDA_CHECK (cudaFree (d_keys));
}

} // namespace cuda
} // namespace plsf
