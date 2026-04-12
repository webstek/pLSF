#pragma once
// cubical_grid.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Uniform 3D cubical grid data structures for lower-star filtration.
//
// Coordinate convention
// ─────────────────────
// A grid of (nx × ny × nz) vertices has the flat vertex index
//
//   v(i, j, k) = i  +  j * nx  +  k * nx * ny        (x is the stride-1 axis)
//
// Cell types
// ──────────
// The full cubical complex built from the grid contains eight cell types:
//
//   Vertex  (dim 0) : the (i,j,k) grid point itself
//   EdgeX   (dim 1) : edge from (i,j,k) to (i+1,j,k)
//   EdgeY   (dim 1) : edge from (i,j,k) to (i,j+1,k)
//   EdgeZ   (dim 1) : edge from (i,j,k) to (i,j,k+1)
//   FaceXY  (dim 2) : axis-aligned square in the XY plane
//   FaceXZ  (dim 2) : axis-aligned square in the XZ plane
//   FaceYZ  (dim 2) : axis-aligned square in the YZ plane
//   Cube    (dim 3) : full 3-cube
//
// Each cell is identified by a (CellType, origin) pair, where `origin` is the
// flat index of the cell's lower-left vertex (smallest i, j, k corner).
//
// cell_vertices() returns the flat indices of all vertices of a cell so that
// the filtration value (= max vertex value) can be computed on the GPU.
// ─────────────────────────────────────────────────────────────────────────────

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace glsf {

// ── Cell type ──────────────────────────────────────────────────────────────────
enum class CellType : uint8_t {
    Vertex = 0,
    EdgeX  = 1,   ///< edge in the +x direction
    EdgeY  = 2,   ///< edge in the +y direction
    EdgeZ  = 3,   ///< edge in the +z direction
    FaceXY = 4,   ///< square in the xy plane
    FaceXZ = 5,   ///< square in the xz plane
    FaceYZ = 6,   ///< square in the yz plane
    Cube   = 7,   ///< 3-cube
    Count  = 8    ///< sentinel – number of distinct cell types
};

inline constexpr int cell_dimension(CellType t) noexcept {
    switch (t) {
    case CellType::Vertex:                                     return 0;
    case CellType::EdgeX: case CellType::EdgeY:
    case CellType::EdgeZ:                                      return 1;
    case CellType::FaceXY: case CellType::FaceXZ:
    case CellType::FaceYZ:                                     return 2;
    case CellType::Cube:                                       return 3;
    default:                                                   return -1;
    }
}

// ── Number of vertices per cell type ──────────────────────────────────────────
inline constexpr int cell_vertex_count(CellType t) noexcept {
    return 1 << cell_dimension(t);   // 1, 2, 4, or 8
}

// ── Cell identifier ────────────────────────────────────────────────────────────
// Lightweight POD so it can live in SYCL device memory without modification.
struct CellId {
    uint32_t origin;   ///< flat vertex index of the lower-left corner
    CellType type;

    bool operator==(const CellId& o) const noexcept {
        return origin == o.origin && type == o.type;
    }
};

// ── 3D cubical scalar grid ─────────────────────────────────────────────────────
template <typename Scalar>
class CubicalGrid3D {
public:
    using scalar_type = Scalar;

    // ── Construction ────────────────────────────────────────────────────────────
    CubicalGrid3D() = default;

    /// @param nx, ny, nz  Vertex counts along each axis.
    /// @param data        Scalar values, x-major (stride 1 in x), length nx*ny*nz.
    CubicalGrid3D(std::size_t nx, std::size_t ny, std::size_t nz,
                  std::vector<Scalar> data)
        : nx_(nx), ny_(ny), nz_(nz), data_(std::move(data))
    {
        assert(data_.size() == nx_ * ny_ * nz_);
    }

    // ── Grid dimensions ──────────────────────────────────────────────────────────
    std::size_t nx() const noexcept { return nx_; }
    std::size_t ny() const noexcept { return ny_; }
    std::size_t nz() const noexcept { return nz_; }

    std::size_t num_vertices() const noexcept { return nx_ * ny_ * nz_; }

    // ── Scalar field access ───────────────────────────────────────────────────────
    Scalar value(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data_[i + j * nx_ + k * nx_ * ny_];
    }
    Scalar value(std::size_t flat) const noexcept { return data_[flat]; }

    const std::vector<Scalar>& data() const noexcept { return data_; }
          std::vector<Scalar>& data()       noexcept { return data_; }

    // ── Cell counts by type ────────────────────────────────────────────────────
    std::size_t num_edges_x()  const noexcept { return (nx_-1) * ny_    * nz_;     }
    std::size_t num_edges_y()  const noexcept { return  nx_    *(ny_-1) * nz_;     }
    std::size_t num_edges_z()  const noexcept { return  nx_    * ny_    *(nz_-1);  }
    std::size_t num_faces_xy() const noexcept { return (nx_-1) *(ny_-1) * nz_;     }
    std::size_t num_faces_xz() const noexcept { return (nx_-1) * ny_    *(nz_-1);  }
    std::size_t num_faces_yz() const noexcept { return  nx_    *(ny_-1) *(nz_-1);  }
    std::size_t num_cubes()    const noexcept { return (nx_-1) *(ny_-1) *(nz_-1);  }

    std::size_t num_cells() const noexcept {
        return num_vertices()
             + num_edges_x()  + num_edges_y()  + num_edges_z()
             + num_faces_xy() + num_faces_xz() + num_faces_yz()
             + num_cubes();
    }

    // ── Global cell index layout ──────────────────────────────────────────────
    // Cells are laid out contiguously in the order:
    //   [vertices | edges_x | edges_y | edges_z |
    //    faces_xy | faces_xz | faces_yz | cubes]
    //
    // offset_for(type) returns the base index of the first cell of that type.
    std::size_t offset_for(CellType t) const noexcept {
        switch (t) {
        case CellType::Vertex:  return 0;
        case CellType::EdgeX:   return num_vertices();
        case CellType::EdgeY:   return num_vertices()  + num_edges_x();
        case CellType::EdgeZ:   return num_vertices()  + num_edges_x()  + num_edges_y();
        case CellType::FaceXY:  return num_vertices()  + num_edges_x()  + num_edges_y()
                                     + num_edges_z();
        case CellType::FaceXZ:  return offset_for(CellType::FaceXY) + num_faces_xy();
        case CellType::FaceYZ:  return offset_for(CellType::FaceXZ) + num_faces_xz();
        case CellType::Cube:    return offset_for(CellType::FaceYZ)  + num_faces_yz();
        default:                return num_cells();
        }
    }

    // ── Flat vertex index ─────────────────────────────────────────────────────
    std::size_t vertex_index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return i + j * nx_ + k * nx_ * ny_;
    }

    // ── Vertex indices of a cell ───────────────────────────────────────────────
    // Returns the flat vertex indices of all corners.  Unused slots carry
    // the value std::numeric_limits<uint32_t>::max().
    // Implemented in lower_star.cpp (needs a translation unit for explicit
    // template instantiation with SYCL device code).
    std::array<uint32_t, 8> cell_vertices(CellId cell) const noexcept;

private:
    std::size_t      nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<Scalar> data_;
};

} // namespace glsf
