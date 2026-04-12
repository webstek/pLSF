#pragma once
// lower_star.hpp
// ─────────────────────────────────────────────────────────────────────────────
// GPU-accelerated lower-star filtration for 3D cubical grids.
//
// Background
// ──────────
// For a scalar field f on a cubical grid, the lower-star filtration assigns
// each cell σ the value
//
//   f(σ) = max{ f(v) : v is a vertex of σ }
//
// Sorting all cells by this value and sweeping in order yields a filtered
// cubical complex whose persistent homology encodes topological features
// (connected components, loops, voids) together with their lifetimes.
//
// Pipeline (implemented in lower_star.cpp)
// ─────────────────────────────────────────
//   Phase 1 – Filtration assignment
//     Parallel SYCL kernel: one work-item per cell computes
//     f(σ) = max vertex value and writes (global_cell_index, f(σ)).
//
//   Phase 2 – Stable sort
//     Sort (index, value) pairs by value, breaking ties by index.
//     Target: oneDPL device_policy or SYCL group sort for GPU parallelism;
//     std::stable_sort as fallback.
//
// ─────────────────────────────────────────────────────────────────────────────

#include "cubical_grid.hpp"

#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace glsf
{

// ── Lower-star filtration
// ──────────────────────────────────────────────────────
template <typename Scalar> class LowerStarFiltration
{
public:
  using scalar_type = Scalar;
  using grid_type   = CubicalGrid3D<Scalar>;
  using pair_type   = PersistencePair<Scalar>;

  LowerStarFiltration() = default;

  // ── Entry point
  // ─────────────────────────────────────────────────────────────
  /// Run the full filtration + persistence computation.
  /// @param queue  Active SYCL queue; device memory is allocated internally.
  /// @param grid   Input scalar field (host memory, read-only).
  void compute(sycl::queue& queue, const grid_type& grid);

  // ── Result accessors ──────────────────────────────────────────────────────
  /// All birth–death pairs produced by the last call to compute().
  const std::vector<pair_type>& pairs() const noexcept { return pairs_; }

  /// Pairs with persistence (death − birth) strictly greater than @p threshold.
  std::vector<pair_type> significant_pairs(Scalar threshold) const;

private:
  // ── Phase 1 ───────────────────────────────────────────────────────────────
  // Assign filtration values to every cell.
  // Returns a flat array of (global_cell_index, filtration_value).
  std::vector<std::pair<uint32_t, Scalar>>
  assign_filtration_values(sycl::queue& queue, const grid_type& grid);

  // ── Phase 2 ───────────────────────────────────────────────────────────────
  // Stable-sort cells by (filtration_value, global_cell_index).
  void sort_cells(sycl::queue&                              queue,
                  std::vector<std::pair<uint32_t, Scalar>>& cells);

  std::vector<pair_type> pairs_;
};

} // namespace glsf
