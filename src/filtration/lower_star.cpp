// lower_star.cpp
// ─────────────────────────────────────────────────────────────────────────────
// SYCL kernel stubs for the lower-star filtration pipeline.
// Compile this file with SYCL enabled (icpx -fsycl, or via AdaptiveCpp).
// ─────────────────────────────────────────────────────────────────────────────

#include "lower_star.hpp"
#include "cubical_grid.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace glsf {

// ═════════════════════════════════════════════════════════════════════════════
// CubicalGrid3D<Scalar>::cell_vertices
// ─────────────────────────────────────────────────────────────────────────────
// Returns the flat vertex indices of every corner of `cell`.
// Slots beyond cell_vertex_count(type) carry UINT32_MAX and are ignored
// downstream.  This function is called from both host code and SYCL kernels
// (after inlining), so no virtual dispatch or heap allocation is used.
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::array<uint32_t, 8>
CubicalGrid3D<Scalar>::cell_vertices(CellId cell) const noexcept
{
    std::array<uint32_t, 8> v;
    v.fill(std::numeric_limits<uint32_t>::max());

    const uint32_t o  = cell.origin;
    const uint32_t ix = static_cast<uint32_t>(o % nx_);
    const uint32_t iy = static_cast<uint32_t>((o / nx_) % ny_);
    const uint32_t iz = static_cast<uint32_t>(o / (nx_ * ny_));

    // Helper: flat index of corner offset (di, dj, dk) from cell origin.
    auto vi = [&](uint32_t di, uint32_t dj, uint32_t dk) -> uint32_t {
        return static_cast<uint32_t>(
            vertex_index(ix + di, iy + dj, iz + dk));
    };

    switch (cell.type) {
    case CellType::Vertex:
        v[0] = vi(0,0,0);
        break;

    case CellType::EdgeX:
        v[0] = vi(0,0,0);  v[1] = vi(1,0,0);
        break;

    case CellType::EdgeY:
        v[0] = vi(0,0,0);  v[1] = vi(0,1,0);
        break;

    case CellType::EdgeZ:
        v[0] = vi(0,0,0);  v[1] = vi(0,0,1);
        break;

    case CellType::FaceXY:
        v[0] = vi(0,0,0);  v[1] = vi(1,0,0);
        v[2] = vi(0,1,0);  v[3] = vi(1,1,0);
        break;

    case CellType::FaceXZ:
        v[0] = vi(0,0,0);  v[1] = vi(1,0,0);
        v[2] = vi(0,0,1);  v[3] = vi(1,0,1);
        break;

    case CellType::FaceYZ:
        v[0] = vi(0,0,0);  v[1] = vi(0,1,0);
        v[2] = vi(0,0,1);  v[3] = vi(0,1,1);
        break;

    case CellType::Cube:
        v[0] = vi(0,0,0);  v[1] = vi(1,0,0);
        v[2] = vi(0,1,0);  v[3] = vi(1,1,0);
        v[4] = vi(0,0,1);  v[5] = vi(1,0,1);
        v[6] = vi(0,1,1);  v[7] = vi(1,1,1);
        break;

    default: break;
    }

    return v;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 1 – assign_filtration_values
// ─────────────────────────────────────────────────────────────────────────────
// For every cell in the complex, compute
//   f(σ) = max{ f(v) : v is a vertex of σ }
// and return a flat array of (global_cell_index, f(σ)) pairs.
//
// TODO: replace the host-side stub below with a SYCL parallel kernel.
//
// Suggested SYCL implementation sketch:
//
//   const std::size_t n_verts = grid.num_vertices();
//   const std::size_t n_cells = grid.num_cells();
//
//   sycl::buffer<Scalar,   1> d_field (grid.data().data(), n_verts);
//   sycl::buffer<uint32_t, 1> d_cidx  (n_cells);   // output: global index
//   sycl::buffer<Scalar,   1> d_fval  (n_cells);   // output: filtration value
//
//   queue.submit([&](sycl::handler& h) {
//       auto field = d_field.template get_access<sycl::access::mode::read>(h);
//       auto cidx  = d_cidx .template get_access<sycl::access::mode::write>(h);
//       auto fval  = d_fval .template get_access<sycl::access::mode::write>(h);
//
//       h.parallel_for(sycl::range<1>{n_cells}, [=](sycl::id<1> gid) {
//           const uint32_t global_idx = static_cast<uint32_t>(gid[0]);
//           // Decode (type, origin) from global_idx using offset_for() logic.
//           // Compute cell vertices, read scalar values, take the maximum.
//           // Write result to cidx[gid] and fval[gid].
//       });
//   });
//   queue.wait_and_throw();
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::vector<std::pair<uint32_t, Scalar>>
LowerStarFiltration<Scalar>::assign_filtration_values(
    sycl::queue& /*queue*/, const grid_type& /*grid*/)
{
    // TODO: implement SYCL parallel kernel (see comment above).
    throw std::runtime_error(
        "LowerStarFiltration::assign_filtration_values – not yet implemented");
    return {};
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 2 – sort_cells
// ─────────────────────────────────────────────────────────────────────────────
// Stable-sort cells by (filtration_value, global_cell_index).
//
// The host fallback below is correct but scalar.  For large grids, replace it
// with a GPU radix sort from oneDPL:
//
//   #include <oneapi/dpl/algorithm>
//   #include <oneapi/dpl/execution>
//
//   auto policy = oneapi::dpl::execution::make_device_policy(queue);
//   oneapi::dpl::stable_sort(policy, cells.begin(), cells.end(), comparator);
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void LowerStarFiltration<Scalar>::sort_cells(
    sycl::queue& /*queue*/,
    std::vector<std::pair<uint32_t, Scalar>>& cells)
{
    // TODO: replace with GPU-parallel sort (see comment above).
    std::stable_sort(cells.begin(), cells.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second
                || (a.second == b.second && a.first < b.first);
        });
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 3 – reduce
// ─────────────────────────────────────────────────────────────────────────────
// Extract persistence pairs by reducing the boundary matrix of the filtered
// cubical complex.
//
// Starting references for GPU-parallel approaches:
//   • Bauer, Kerber, Reininghaus – "Clear and Compress: Computing Persistent
//     Homology in Chunks", TopoInVis 2013.
//   • Morozov, Weber – "Distributed Cohomology Computation", SoCG 2014.
//   • Zhang, Xin, et al. – "GPU-Accelerated Computation of Vietoris-Rips
//     Persistence Barcodes", ICML 2020 (ripser-like implicit coboundary).
//
// For a cubical complex, the coboundary is implicit (no explicit sparse matrix
// storage needed); each face of a cell can be enumerated from (type, origin).
//
// Implementation outline:
//   1. Dimension 0 (connected components): parallel union-find over vertices,
//      processing vertex-birth and edge-birth events in filtration order.
//   2. Dimensions 1–2: chunk-based column reduction or twist reduction with
//      atomics for concurrent column updates.
//   3. Dimension 3 (voids): dual to dim-0 by Alexander duality on the grid.
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void LowerStarFiltration<Scalar>::reduce(
    sycl::queue& /*queue*/,
    const grid_type& /*grid*/,
    std::vector<std::pair<uint32_t, Scalar>>& /*sorted_cells*/)
{
    // TODO: implement GPU boundary-matrix reduction (see comment above).
    // Populate this->pairs_ with the resulting PersistencePair values.
    throw std::runtime_error(
        "LowerStarFiltration::reduce – not yet implemented");
}

// ═════════════════════════════════════════════════════════════════════════════
// compute  (orchestrates phases 1–3)
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void LowerStarFiltration<Scalar>::compute(sycl::queue& queue,
                                           const grid_type& grid)
{
    pairs_.clear();
    auto cells = assign_filtration_values(queue, grid);
    sort_cells(queue, cells);
    reduce(queue, grid, cells);
}

// ═════════════════════════════════════════════════════════════════════════════
// significant_pairs  (host post-processing)
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::vector<PersistencePair<Scalar>>
LowerStarFiltration<Scalar>::significant_pairs(Scalar threshold) const
{
    std::vector<pair_type> result;
    result.reserve(pairs_.size());
    for (const auto& p : pairs_)
        if ((p.death - p.birth) > threshold)
            result.push_back(p);
    return result;
}

// ── Explicit template instantiations ─────────────────────────────────────────
// Add further scalar types (e.g. int32_t, uint16_t) as needed.
template class CubicalGrid3D<float>;
template class CubicalGrid3D<double>;
template class CubicalGrid3D<int16_t>;

template class LowerStarFiltration<float>;
template class LowerStarFiltration<double>;
template class LowerStarFiltration<int16_t>;

} // namespace glsf
