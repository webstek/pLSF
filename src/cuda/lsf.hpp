#pragma once
// ****************************************************************************
/// @file cuda/lsf.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 13 Apr 2026
/// @brief Lower Star Filtration using CUDA
// ****************************************************************************
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "../grid.hpp"
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace plsf
{

/// @brief lower star filtration of a 3D cubical complex (CUDA backend)
/// @tparam scalar datatype of filtration
template <typename scalar> struct LowerStarFiltration
{
  Grid<scalar>           grid;
  CubicalComplex<scalar> cc;
  std::vector<uint32_t>  ordering;

  /// Device cube map — persists across compute_complex → compute_ordering
  /// so that the data stays on the GPU for the sort without a round-trip.
  scalar *d_cube_map = nullptr;

  LowerStarFiltration (Grid<scalar> const &grid) : grid (grid) {}

  ~LowerStarFiltration ()
  {
    if (d_cube_map)
      {
        cudaFree (d_cube_map);
        d_cube_map = nullptr;
      }
  }

  /// @brief Does all computation required to compute the lower star filtration
  void compute ()
  {
    compute_complex ();
    compute_ordering ();
  }

  /// @brief Creates the cubical complex using vertex maximum rule (CUDA)
  /// @details Allocates d_grid and d_cube_map on the device.  d_grid is freed
  ///          after the kernel completes; d_cube_map is kept for
  ///          compute_ordering so the sort can run entirely on the GPU.
  void compute_complex ()
  {
    const uint64_t Nx = grid.Nx;
    const uint64_t Ny = grid.Ny;
    const uint64_t Nz = grid.Nz;
    const uint64_t n_grid = Nx * Ny * Nz;
    const uint64_t n_cells
        = (2 * Nx - 1) * (2 * Ny - 1) * (2 * Nz - 1);

    cc.Nx = grid.Nx;
    cc.Ny = grid.Ny;
    cc.Nz = grid.Nz;
    cc.cube_map.resize (n_cells);

    scalar *d_grid = nullptr;

    CUDA_CHECK (cudaMalloc (&d_grid, n_grid * sizeof (scalar)));
    CUDA_CHECK (cudaMalloc (&d_cube_map, n_cells * sizeof (scalar)));

    CUDA_CHECK (cudaMemcpy (d_grid, grid.data.data (),
        n_grid * sizeof (scalar), cudaMemcpyHostToDevice));

    cuda::launch_compute_complex (d_grid, d_cube_map, Nx, Ny, Nz, n_cells);

    CUDA_CHECK (cudaFree (d_grid));

    CUDA_CHECK (cudaMemcpy (cc.cube_map.data (), d_cube_map,
        n_cells * sizeof (scalar), cudaMemcpyDeviceToHost));
    // d_cube_map remains allocated for compute_ordering
  }

  /// @brief Generates the filtration order of indices into the cubical complex
  /// @details
  /// The ordering is determined first by filtration value and secondly by
  /// dimension of the d-cubes with the same filtration value (so that
  /// vertices appear before faces with them as their boundaries, i.e.
  /// dimension ascending).
  ///
  /// Cell dimension is derived on-the-fly from each cell's cubemap index
  /// (count of odd doubled-coordinates), encoded together with the
  /// filtration value into a composite sort key, and the entire sort is
  /// performed on the GPU via Thrust radix sort.
  void compute_ordering ()
  {
    const uint64_t Nx = cc.Nx;
    const uint64_t Ny = cc.Ny;
    const uint64_t Nz = cc.Nz;
    const uint64_t n_cells
        = (2 * Nx - 1) * (2 * Ny - 1) * (2 * Nz - 1);

    if (n_cells > 4294967295ULL)
      throw std::runtime_error (
        "grid too large: ordering indices exceed uint32_t range");

    ordering.resize (n_cells);

    uint32_t *d_ordering = nullptr;
    CUDA_CHECK (cudaMalloc (&d_ordering, n_cells * sizeof (uint32_t)));

    cuda::launch_sort_filtration (
        d_cube_map, d_ordering, Nx, Ny, Nz, n_cells);

    CUDA_CHECK (cudaMemcpy (ordering.data (), d_ordering,
      n_cells * sizeof (uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK (cudaFree (d_ordering));
    CUDA_CHECK (cudaFree (d_cube_map));
    d_cube_map = nullptr;
  }
};

} // namespace plsf
