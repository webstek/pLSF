#pragma once
// ****************************************************************************
/// @file boundary.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief PHAT boundary matrix computation from a lower-star filtration
/// @details
/// Computes the boundary matrix of a completed LowerStarFiltration directly
/// into a phat::boundary_matrix<phat::bit_tree_pivot_column>, using two
/// SYCL-accelerated passes:
///
///   Pass 1 – build the inverse index map and cell dimensions in one kernel:
///              inv_ordering[ordering[p]] = p  (scatter, no conflicts)
///              dims[p] = #odd coords of ordering[p]
///
///   Pass 2 – for each filtration position p, enumerate the 2*dim boundary
///              face global CubeMap indices and look up their filtration
///              position through inv_ordering; write results to boundaries[].
///
/// The resulting boundary matrix is stored in phat's bit_tree_pivot_column
/// representation, ready for persistence pair computation via phat's
/// reduction algorithms.  A companion filtration-values vector stores one
/// double per cell, mapping column indices to birth/death values.
// ****************************************************************************
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
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
/// @param q    SYCL queue for GPU-accelerated passes
/// @return     BoundaryResult with phat boundary matrix and filtration values
template <typename scalar>
BoundaryResult compute_boundary_matrix (
    LowerStarFiltration<scalar> const &lsf, sycl::queue &q)
{
  const uint64_t Nx = lsf.cc.Nx;
  const uint64_t Ny = lsf.cc.Ny;
  const uint64_t Nz = lsf.cc.Nz;
  const uint64_t Mx = 2 * Nx - 1;
  const uint64_t My = 2 * Ny - 1;
  const uint64_t Mz = 2 * Nz - 1;
  const uint64_t MxMy = Mx * My;
  const uint64_t n_cells = Mx * My * Mz;
  const uint64_t num_cells = static_cast<uint64_t> (lsf.ordering.size ());

  if (num_cells != n_cells)
    throw std::runtime_error (
        "ordering size does not match cubical complex size");

  // Filtration positions are stored as uint32_t; verify they fit.
  if (num_cells > 4294967295ULL)
    throw std::runtime_error (
        "grid too large: filtration positions exceed uint32_t range");

  auto *d_ordering = sycl::malloc_device<uint64_t> (num_cells, q);
  auto *d_inv = sycl::malloc_device<uint64_t> (num_cells, q); // inv_ordering
  auto *d_dims = sycl::malloc_device<uint8_t> (num_cells, q);

  if (!d_ordering || !d_inv || !d_dims)
    throw std::runtime_error ("SYCL device allocation failed");

  q.memcpy (d_ordering, lsf.ordering.data (), num_cells * sizeof (uint64_t))
      .wait ();

  // d_inv[ordering[p]] = p  (permutation scatter – no write conflicts)
  // d_dims[p]          = number of odd doubled coordinates of ordering[p]
  q.submit ([&] (sycl::handler &h) {
     h.parallel_for (sycl::range<1> (num_cells), [=] (sycl::id<1> id) {
       const uint64_t p = static_cast<uint64_t> (id[0]);
       const uint64_t idx = d_ordering[p];

       d_inv[idx] = p;

       const uint64_t ci = idx % Mx;
       const uint64_t cj = (idx / Mx) % My;
       const uint64_t ck = idx / MxMy;

       d_dims[p] = static_cast<uint8_t> ((ci & 1u) + (cj & 1u) + (ck & 1u));
     });
   }).wait ();

  std::vector<uint8_t> dims (num_cells);
  q.memcpy (dims.data (), d_dims, num_cells * sizeof (uint8_t)).wait ();

  // Compute offsets (temporary; not stored in BoundaryMatrix)
  std::vector<uint64_t> offsets (num_cells + 1);
  offsets[0] = 0;
  for (uint64_t p = 0; p < num_cells; ++p)
    offsets[p + 1] = offsets[p] + 2u * dims[p];

  const uint64_t total_bnd = offsets[num_cells];
  const uint64_t bnd_alloc = total_bnd > 0 ? total_bnd : uint64_t{ 1 };

  auto *d_offsets = sycl::malloc_device<uint64_t> (num_cells + 1, q);
  auto *d_boundaries = sycl::malloc_device<uint32_t> (bnd_alloc, q);

  if (!d_offsets || !d_boundaries)
    throw std::runtime_error ("SYCL device allocation failed");

  q.memcpy (d_offsets, offsets.data (), (num_cells + 1) * sizeof (uint64_t))
      .wait ();

  // For each odd doubled coordinate c at axis k, generate the two boundary
  // faces at c-1 and c+1 along that axis, convert to global CubeMap index,
  // and look up the filtration position via d_inv.
  q.submit ([&] (sycl::handler &h) {
     h.parallel_for (sycl::range<1> (num_cells), [=] (sycl::id<1> id) {
       const uint64_t p = static_cast<uint64_t> (id[0]);
       const uint64_t idx = d_ordering[p];

       const uint64_t ci = idx % Mx;
       const uint64_t cj = (idx / Mx) % My;
       const uint64_t ck = idx / MxMy;

       uint64_t slot = d_offsets[p];

       if (ci & 1u)
         {
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[(ci - 1) + cj * Mx + ck * MxMy]);
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[(ci + 1) + cj * Mx + ck * MxMy]);
         }
       if (cj & 1u)
         {
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[ci + (cj - 1) * Mx + ck * MxMy]);
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[ci + (cj + 1) * Mx + ck * MxMy]);
         }
       if (ck & 1u)
         {
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[ci + cj * Mx + (ck - 1) * MxMy]);
           d_boundaries[slot++]
               = static_cast<uint32_t> (d_inv[ci + cj * Mx + (ck + 1) * MxMy]);
         }
     });
   }).wait ();

  std::vector<uint32_t> boundaries (bnd_alloc);
  q.memcpy (boundaries.data (), d_boundaries, bnd_alloc * sizeof (uint32_t))
      .wait ();
  boundaries.resize (static_cast<std::size_t> (total_bnd));

  sycl::free (d_ordering, q);
  sycl::free (d_inv, q);
  sycl::free (d_dims, q);
  sycl::free (d_offsets, q);
  sycl::free (d_boundaries, q);

  // Populate phat boundary matrix with parallel column insertion.
  // Each column has at most 6 boundary entries (cubes in 3D); PHAT requires
  // them sorted in ascending order.
  BoundaryResult result;
  result.matrix.set_num_cols (static_cast<phat::index> (num_cells));

#pragma omp parallel for schedule(dynamic, 1024)
  for (int64_t p = 0; p < static_cast<int64_t> (num_cells); ++p)
    {
      result.matrix.set_dim (
          static_cast<phat::index> (p), static_cast<phat::dimension> (dims[p]));

      const uint64_t beg = offsets[p];
      const uint64_t end = offsets[p + 1];
      phat::column   col (end - beg);
      for (uint64_t b = beg; b < end; ++b)
        col[b - beg] = static_cast<phat::index> (boundaries[b]);
      std::sort (col.begin (), col.end ());
      result.matrix.set_col (static_cast<phat::index> (p), col);
    }

  result.filt_values.resize (num_cells);
  for (uint64_t p = 0; p < num_cells; ++p)
    result.filt_values[p]
        = static_cast<double> (lsf.cc.cube_map[lsf.ordering[p]]);

  return result;
}
// ****************************************************************************

} // namespace plsf
// ****************************************************************************
