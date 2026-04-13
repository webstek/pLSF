#pragma once
// ****************************************************************************
/// @file lsf.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief Lower Star Filtration using SYCL
// ****************************************************************************
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "grid.hpp"

namespace plsf
{

/// @brief lower star filtration of a 3D cubical complex
/// @tparam scalar datatype of filtration
template <typename scalar> struct LowerStarFiltration
{
  Grid<scalar>           grid;
  CubicalComplex<scalar> cc;
  std::vector<uint64_t>  ordering;

  LowerStarFiltration (Grid<scalar> const &grid) : grid (grid) {}

  /// @brief Does all computation required to compute the lower star filtration
  void compute (sycl::queue &q)
  {
    compute_complex (q);
    compute_ordering (q);
  }

  /// @brief Creates the cubical complex using vertex maximum rule
  /// @details
  /// Uses the cell-centered values from the grid for vertex values, propagates
  /// these values to cubes that contain that vertex, i.e.
  ///   f(c) = max {f(v) : v is a vertex of c}
  void compute_complex (sycl::queue &q)
  {
    const uint64_t Nx = grid.Nx;
    const uint64_t Ny = grid.Ny;
    const uint64_t Nz = grid.Nz;
    const uint64_t Mx = 2 * Nx - 1;
    const uint64_t My = 2 * Ny - 1;
    const uint64_t Mz = 2 * Nz - 1;
    const uint64_t MxMy = Mx * My;
    const uint64_t n_grid = Nx * Ny * Nz;
    const uint64_t n_cells = Mx * My * Mz;

    cc.Nx = grid.Nx;
    cc.Ny = grid.Ny;
    cc.Nz = grid.Nz;
    cc.cube_map.resize (n_cells);

    scalar *d_grid = sycl::malloc_device<scalar> (n_grid, q);
    scalar *d_cube_map = sycl::malloc_device<scalar> (n_cells, q);

    if (!d_grid || !d_cube_map)
      throw std::runtime_error ("SYCL device allocation failed");

    q.memcpy (d_grid, grid.data.data (), n_grid * sizeof (scalar)).wait ();

    q.submit ([&] (sycl::handler &h) {
       h.parallel_for (sycl::range<1> (n_cells), [=] (sycl::id<1> id) {
         const uint64_t idx = static_cast<uint64_t> (id[0]);
         const uint64_t ci = idx % Mx;
         const uint64_t cj = (idx / Mx) % My;
         const uint64_t ck = idx / MxMy;

         // Vertex grid-index ranges per axis: one index if coord is even,
         // two (coord/2 and coord/2+1) if odd
         const uint64_t vx0 = ci >> 1;
         const uint64_t vx1 = vx0 + (ci & 1u);
         const uint64_t vy0 = cj >> 1;
         const uint64_t vy1 = vy0 + (cj & 1u);
         const uint64_t vz0 = ck >> 1;
         const uint64_t vz1 = vz0 + (ck & 1u);

         scalar max_val = d_grid[vx0 + Nx * vy0 + Nx * Ny * vz0];
         for (uint64_t vz = vz0; vz <= vz1; ++vz)
           for (uint64_t vy = vy0; vy <= vy1; ++vy)
             for (uint64_t vx = vx0; vx <= vx1; ++vx)
               {
                 const scalar v = d_grid[vx + Nx * vy + Nx * Ny * vz];
                 if (v > max_val)
                   max_val = v;
               }

         d_cube_map[idx] = max_val;
       });
     }).wait ();

    q.memcpy (cc.cube_map.data (), d_cube_map, n_cells * sizeof (scalar))
        .wait ();

    sycl::free (d_grid, q);
    sycl::free (d_cube_map, q);
  }

  /// @brief Generates the filtration order of indices into the cubical complex
  /// @details
  /// The ordering is determined first by filtration value and secondly by
  /// dimension of the d-cubes with the same filtration value (so that
  /// vertices appear before faces with them as their boundaries, i.e.
  /// dimension ascending).
  void compute_ordering (sycl::queue &q)
  {
    const uint64_t Nx = cc.Nx;
    const uint64_t Ny = cc.Ny;
    const uint64_t Nz = cc.Nz;
    const uint64_t Mx = 2 * Nx - 1;
    const uint64_t My = 2 * Ny - 1;
    const uint64_t Mz = 2 * Nz - 1;
    const uint64_t MxMy = Mx * My;
    const uint64_t n_cells = Mx * My * Mz;

    // SYCL kernel: compute the dimension of each cell in parallel
    int *d_dims = sycl::malloc_device<int> (n_cells, q);
    if (!d_dims)
      throw std::runtime_error ("SYCL device allocation failed");

    q.submit ([&] (sycl::handler &h) {
       h.parallel_for (sycl::range<1> (n_cells), [=] (sycl::id<1> id) {
         const uint64_t idx = static_cast<uint64_t> (id[0]);
         const uint64_t ci = idx % Mx;
         const uint64_t cj = (idx / Mx) % My;
         const uint64_t ck = idx / MxMy;
         d_dims[idx] = static_cast<int> ((ci & 1u) + (cj & 1u) + (ck & 1u));
       });
     }).wait ();

    std::vector<int> dims (n_cells);
    q.memcpy (dims.data (), d_dims, n_cells * sizeof (int)).wait ();
    sycl::free (d_dims, q);

    // Sort indices by (filtration value asc, dimension asc)
    ordering.resize (n_cells);
    std::iota (ordering.begin (), ordering.end (), uint64_t{ 0 });

    const scalar *filt = cc.cube_map.data ();
    const int    *dim = dims.data ();
    std::sort (ordering.begin (), ordering.end (),
        [filt, dim] (uint64_t a, uint64_t b) {
          if (filt[a] != filt[b])
            return filt[a] < filt[b];
          return dim[a] < dim[b];
        });
  }
};
// ****************************************************************************

} // namespace plsf
// ****************************************************************************