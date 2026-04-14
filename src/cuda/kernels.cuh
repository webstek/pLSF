#pragma once
// ****************************************************************************
/// @file cuda/kernels.cuh
/// @author Kyle Webster
/// @version 0.1
/// @date 13 Apr 2026
/// @brief CUDA kernel declarations for lower-star filtration
// ****************************************************************************

#include <cstdint>

namespace plsf
{
namespace cuda
{

/// @brief Compute the cubical complex filtration values (vertex-max rule)
/// @param d_grid      Device pointer to the input scalar grid (Nx*Ny*Nz)
/// @param d_cube_map  Device pointer to the output cube map (Mx*My*Mz)
/// @param Nx, Ny, Nz  Grid dimensions
/// @param n_cells     Total number of cells in the cubical complex
void launch_compute_complex (const float *d_grid, float *d_cube_map,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells);
void launch_compute_complex (const uint8_t *d_grid, uint8_t *d_cube_map,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells);

/// @brief Sort the filtration by (value asc, dimension asc) on the GPU
/// @param d_cube_map  Device pointer to the cube map (read-only, Mx*My*Mz)
/// @param d_ordering  Device pointer to the output index array (Mx*My*Mz).
///                    On return, contains cell indices sorted by filtration.
/// @param Nx, Ny, Nz  Grid dimensions
/// @param n_cells     Total number of cells in the cubical complex
/// @param lossy       If true (float only), encode dimension in the 2 LSBs
///                    of the sortable value key and perform a single sort pass
///                    instead of two stable passes.  Values differing only in
///                    those 2 bits will be ordered by dimension between them.
void launch_sort_filtration (const float *d_cube_map, uint32_t *d_ordering,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells,
    bool lossy = false);
void launch_sort_filtration (const uint8_t *d_cube_map, uint32_t *d_ordering,
    uint64_t Nx, uint64_t Ny, uint64_t Nz, uint64_t n_cells,
    bool lossy = false);

} // namespace cuda
} // namespace plsf
