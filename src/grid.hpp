#pragma once
// ****************************************************************************
/// @file grid.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief Cubical grid
// ****************************************************************************
#include <cstdint>
#include <vector>

namespace plsf
{

/// @brief 3D uniform grid
/// @tparam scalar type of data stored on grid at cell centers
template <typename scalar> struct Grid
{
  uint64_t            Nx, Ny, Nz; ///< number of cells in each dimension
  std::vector<scalar> data;       ///< data
  scalar              at (uint64_t i, uint64_t j, uint64_t k)
  { // returns the scalar data at 3d index (i,j,k)
    return data[i + Nx * j + Nx * Ny * k];
  }
};
// ********************************************************

/// @brief 3D cubical complex using the CubeMap representation
/// @tparam scalar
/// @details
/// A CubeMap representation of a cubical complex stores the vertices, edges,
/// faces, and cubes in a single flattened array. The CubeMap has the
/// following mapping from indices to components:
///   i,j,k even: vertex
///     j,k even: x-edge
///     i,k even: y-edge
///     i,j even: z-edge
///       k even: xy-face
///       j even: xz-face
///       i even: yz-face
///   i,j,k  odd: cube
/// with the following counts:
///   vertices: NxNyNz
///   edges: (Nx-1)NyNz+Nx(Ny-1)Nz+NxNy(Nz-1)
///   faces: (Nx-1)(Ny-1)Nz+(Nx-1)Ny(Nz-1)+Nx(Ny-1)(Nz-1)
///   cubes: (Nx-1)(Ny-1)(Nz-1)
/// for a total of (2Nx-1)(2Ny-1)(2Nz-1) entries
template <typename scalar> struct CubicalComplex
{
  uint64_t            Nx, Ny, Nz; ///< number of vertices in each dimension
  std::vector<scalar> cube_map;   ///< CubeMap vector of cubical complexes
};
// ********************************************************

} // namespace plsf