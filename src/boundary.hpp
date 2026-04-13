#pragma once
// ****************************************************************************
/// @file boundary.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief PHAT boundary matrix computation from a lower-star filtration
/// @details
/// Computes the boundary matrix of a completed LowerStarFiltration directly
/// into a phat::boundary_matrix<phat::bit_tree_pivot_column> in a single
/// OpenMP-parallel pass over all cells.
///
/// The inverse ordering (cube-map index → filtration position) is built as
/// a uint32_t lookup table — the only intermediate allocation beyond the
/// phat representation itself.  Each column's dimension and boundary entries
/// are computed on-the-fly from the doubled-coordinate representation,
/// avoiding separate dims, offsets, and boundary arrays.
///
/// phat's data structures are CPU-based (std::vector) so SYCL device
/// kernels cannot populate columns directly.  OpenMP is used instead,
/// consistent with phat's own parallel initialisation path.
// ****************************************************************************
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <phat/boundary_matrix.h>
#include <phat/representations/bit_tree_pivot_column.h>

#include "lsf.hpp"

namespace plsf
{

// ****************************************************************************
/// @brief Result of boundary matrix computation
struct BoundaryResult
{
  phat::boundary_matrix<phat::bit_tree_pivot_column>
      matrix; ///< boundary matrix in bit_tree_pivot_column representation
  std::vector<double> filt_values; ///< filtration value per filtration position
};
// ****************************************************************************

// ****************************************************************************
/// @brief Compute the PHAT boundary matrix from a completed lower-star
/// filtration
/// @tparam scalar filtration value type
/// @param lsf  completed lower-star filtration (cc and ordering must be
/// populated)
/// @return     BoundaryResult with phat boundary matrix and filtration values
template <typename scalar>
BoundaryResult compute_boundary_matrix (LowerStarFiltration<scalar> const &lsf)
{
  const uint64_t Nx = lsf.cc.Nx;
  const uint64_t Ny = lsf.cc.Ny;
  const uint64_t Nz = lsf.cc.Nz;
  const uint64_t Mx = 2 * Nx - 1;
  const uint64_t My = 2 * Ny - 1;
  const uint64_t Mz = 2 * Nz - 1;
  const uint64_t MxMy = Mx * My;
  const uint64_t num_cells = Mx * My * Mz;

  if (num_cells != static_cast<uint64_t> (lsf.ordering.size ()))
    throw std::runtime_error (
        "ordering size does not match cubical complex size");

  // Filtration positions stored as uint32_t; verify they fit.
  if (num_cells > 4294967295ULL)
    throw std::runtime_error (
        "grid too large: filtration positions exceed uint32_t range");

  // Build inverse ordering: cube-map index → filtration position.
  // This is the only intermediate allocation (4 bytes/cell).
  std::vector<uint32_t> inv (num_cells);

#pragma omp parallel for schedule(static)
  for (int64_t p = 0; p < static_cast<int64_t> (num_cells); ++p)
    inv[lsf.ordering[p]] = static_cast<uint32_t> (p);

  // Populate phat boundary matrix in a single parallel pass.
  // For each filtration position p we:
  //   1. Recover the cube-map index from ordering[p]
  //   2. Compute the cell dimension from doubled coordinates
  //   3. Enumerate the 2*dim boundary face cube-map indices
  //   4. Look up each face's filtration position via inv
  //   5. Sort and set the column in the phat representation
  BoundaryResult result;
  result.matrix.set_num_cols (static_cast<phat::index> (num_cells));
  result.filt_values.resize (num_cells);

  const uint64_t *ordering = lsf.ordering.data ();
  const scalar   *cube_map = lsf.cc.cube_map.data ();
  const uint32_t *inv_ptr = inv.data ();
  double         *fv = result.filt_values.data ();

#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t p = 0; p < static_cast<int64_t> (num_cells); ++p)
    {
      const uint64_t idx = ordering[p];
      const uint64_t ci = idx % Mx;
      const uint64_t cj = (idx / Mx) % My;
      const uint64_t ck = idx / MxMy;
      const uint8_t  dim
          = static_cast<uint8_t> ((ci & 1u) + (cj & 1u) + (ck & 1u));

      result.matrix.set_dim (
          static_cast<phat::index> (p), static_cast<phat::dimension> (dim));

      if (dim > 0)
        {
          phat::column col;
          col.reserve (2 * dim);

          if (ci & 1u)
            {
              col.push_back (static_cast<phat::index> (
                  inv_ptr[(ci - 1) + cj * Mx + ck * MxMy]));
              col.push_back (static_cast<phat::index> (
                  inv_ptr[(ci + 1) + cj * Mx + ck * MxMy]));
            }
          if (cj & 1u)
            {
              col.push_back (static_cast<phat::index> (
                  inv_ptr[ci + (cj - 1) * Mx + ck * MxMy]));
              col.push_back (static_cast<phat::index> (
                  inv_ptr[ci + (cj + 1) * Mx + ck * MxMy]));
            }
          if (ck & 1u)
            {
              col.push_back (static_cast<phat::index> (
                  inv_ptr[ci + cj * Mx + (ck - 1) * MxMy]));
              col.push_back (static_cast<phat::index> (
                  inv_ptr[ci + cj * Mx + (ck + 1) * MxMy]));
            }

          std::sort (col.begin (), col.end ());
          result.matrix.set_col (static_cast<phat::index> (p), col);
        }

      fv[p] = static_cast<double> (cube_map[ordering[p]]);
    }

  return result;
}
// ****************************************************************************

} // namespace plsf
// ****************************************************************************
