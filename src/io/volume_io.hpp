#pragma once
// volume_io.hpp
// ─────────────────────────────────────────────────────────────────────────────
// File I/O for 3D scalar volumes from medical imaging and scientific
// simulations.
//
// Supported formats (implemented in volume_io.cpp):
//   • Raw binary     (.raw / .bin)    – headerless flat voxel arrays
//   • NIfTI-1/2      (.nii / .nii.gz) – neuroimaging standard [stub]
//   • MetaImage      (.mhd + .raw)    – ITK/VTK interchange format [stub]
//
// Persistence pair output:
//   • CSV  (.csv)  – birth, death, dimension (human-readable)
//   • Bin  (.bin)  – compact binary: uint64_t n, then n×{Scalar×2, int32_t}
//
// Usage example:
//   auto [grid, meta] = glsf::io::read_volume<float>("scan.nii");
//   glsf::io::write_pairs_csv("pairs.csv", filtration.pairs());
// ─────────────────────────────────────────────────────────────────────────────

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "../filtration/cubical_grid.hpp"
#include "../filtration/lower_star.hpp"

namespace glsf::io {

// ── Volume metadata ────────────────────────────────────────────────────────────
struct VolumeMetadata {
    std::array<std::size_t, 3> dims    = {0, 0, 0};      ///< nx, ny, nz (voxels)
    std::array<double, 3>      spacing = {1., 1., 1.};   ///< voxel size (mm)
    std::array<double, 3>      origin  = {0., 0., 0.};   ///< world-space origin
    std::string                source_file;
    std::string                data_type;   ///< "float32", "float64", "int16", …
};

// ── Read – auto-dispatch on file extension ────────────────────────────────────
/// Detect format from extension (.nii, .nii.gz, .mhd) and load as Scalar.
template <typename Scalar = float>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_volume(const std::filesystem::path& path);

// ── Read – explicit format ────────────────────────────────────────────────────
/// Headerless raw binary: caller supplies grid dimensions.
/// @param offset_bytes  Bytes to skip at the start of the file (custom headers).
template <typename Scalar = float>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_raw(const std::filesystem::path& path,
         std::size_t nx, std::size_t ny, std::size_t nz,
         std::size_t offset_bytes = 0);

/// NIfTI-1 or NIfTI-2 (.nii / .nii.gz).
template <typename Scalar = float>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_nifti(const std::filesystem::path& path);

/// MetaImage (.mhd header + companion .raw data file).
template <typename Scalar = float>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_metaimage(const std::filesystem::path& path);

// ── Write – persistence pairs ─────────────────────────────────────────────────
/// CSV: one row per pair → "birth,death,dimension\n"
template <typename Scalar>
void write_pairs_csv(const std::filesystem::path&              path,
                     const std::vector<PersistencePair<Scalar>>& pairs);

/// Binary: uint64_t n, then n × { Scalar birth, Scalar death, int32_t dim }
template <typename Scalar>
void write_pairs_bin(const std::filesystem::path&              path,
                     const std::vector<PersistencePair<Scalar>>& pairs);

// ── Write – scalar volume ─────────────────────────────────────────────────────
/// Write the scalar field as a headerless raw binary file (x-major order).
template <typename Scalar>
void write_raw(const std::filesystem::path& path,
               const CubicalGrid3D<Scalar>& grid);

} // namespace glsf::io
