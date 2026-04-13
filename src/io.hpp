#pragma once
// ****************************************************************************
/// @file io.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief NIfTI-1/2 volume I/O for 3D scalar grids
/// @details
/// Supports NIfTI-1 (sizeof_hdr = 348) and NIfTI-2 (sizeof_hdr = 540),
/// uncompressed .nii files only. Byte-swapping and scl_slope/scl_inter
/// rescaling are applied automatically.
///
/// Supported voxel datatypes:
///   uint8 (2), int16 (4), int32 (8), float32 (16), float64 (64),
///   int8 (256), uint16 (512), uint32 (768)
// ****************************************************************************
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "boundary.hpp"
#include "grid.hpp"

namespace plsf
{

namespace detail
{

template <typename T>
T read_at (const std::vector<char> &buf, std::size_t offset)
{
  T v;
  std::memcpy (&v, buf.data () + offset, sizeof (T));
  return v;
}

template <typename T> T bswap (T v)
{
  char buf[sizeof (T)];
  std::memcpy (buf, &v, sizeof (T));
  std::reverse (buf, buf + sizeof (T));
  std::memcpy (&v, buf, sizeof (T));
  return v;
}

template <typename scalar, typename src_t>
void convert_voxels (std::vector<scalar> &dst, const std::vector<char> &raw,
    std::size_t n, bool swap, bool apply_scale, double scl_slope,
    double scl_inter)
{
  for (std::size_t i = 0; i < n; ++i)
    {
      src_t v;
      std::memcpy (&v, raw.data () + i * sizeof (src_t), sizeof (src_t));
      if (swap)
        v = bswap (v);
      if (apply_scale)
        dst[i] = static_cast<scalar> (
            scl_slope * static_cast<double> (v) + scl_inter);
      else
        dst[i] = static_cast<scalar> (v);
    }
}

} // namespace detail

/// @brief Read a NIfTI-1 or NIfTI-2 (.nii) file into a Grid
/// @tparam scalar  datatype to cast voxels into
/// @param path     path to the .nii file
template <typename scalar>
Grid<scalar> read_nifti (const std::filesystem::path &path)
{
  std::ifstream file (path, std::ios::binary);
  if (!file)
    throw std::runtime_error (
        "read_nifti: cannot open '" + path.string () + "'");

  // Detect NIfTI version and byte order
  int32_t sizeof_hdr_raw;
  if (!file.read (reinterpret_cast<char *> (&sizeof_hdr_raw), 4))
    throw std::runtime_error (
        "read_nifti: failed to read header in '" + path.string () + "'");

  int32_t sizeof_hdr = sizeof_hdr_raw;
  bool    swap = false;

  if (sizeof_hdr != 348 && sizeof_hdr != 540)
    {
      sizeof_hdr = detail::bswap (sizeof_hdr_raw);
      swap = true;
    }

  if (sizeof_hdr != 348 && sizeof_hdr != 540)
    throw std::runtime_error (
        "read_nifti: unrecognised header size in '" + path.string () + "'");

  const int         version = (sizeof_hdr == 348) ? 1 : 2;
  const std::size_t hdr_size = static_cast<std::size_t> (sizeof_hdr);

  // Read full header
  std::vector<char> hdr (hdr_size);
  file.seekg (0);
  if (!file.read (hdr.data (), static_cast<std::streamsize> (hdr_size)))
    throw std::runtime_error (
        "read_nifti: truncated header in '" + path.string () + "'");

  // Parse dimensions, datatype, and scaling
  std::size_t Nx, Ny, Nz;
  int16_t     datatype;
  std::size_t vox_offset;
  double      scl_slope, scl_inter;

  if (version == 1)
    {
      auto ndim = detail::read_at<int16_t> (hdr, 40);
      auto d1 = detail::read_at<int16_t> (hdr, 42);
      auto d2 = detail::read_at<int16_t> (hdr, 44);
      auto d3 = detail::read_at<int16_t> (hdr, 46);
      datatype = detail::read_at<int16_t> (hdr, 70);
      auto voff = detail::read_at<float> (hdr, 108);
      auto slop = detail::read_at<float> (hdr, 112);
      auto intr = detail::read_at<float> (hdr, 116);

      if (swap)
        {
          ndim = detail::bswap (ndim);
          d1 = detail::bswap (d1);
          d2 = detail::bswap (d2);
          d3 = detail::bswap (d3);
          datatype = detail::bswap (datatype);
          voff = detail::bswap (voff);
          slop = detail::bswap (slop);
          intr = detail::bswap (intr);
        }

      if (ndim < 3)
        throw std::runtime_error (
            "read_nifti: expected a 3D volume in '" + path.string () + "'");

      Nx = static_cast<std::size_t> (d1);
      Ny = static_cast<std::size_t> (d2);
      Nz = static_cast<std::size_t> (d3);
      vox_offset = std::max (static_cast<std::size_t> (voff), hdr_size + 4);
      scl_slope = static_cast<double> (slop);
      scl_inter = static_cast<double> (intr);
    }
  else // version == 2
    {
      datatype = detail::read_at<int16_t> (hdr, 12);
      auto ndim = detail::read_at<int64_t> (hdr, 16);
      auto d1 = detail::read_at<int64_t> (hdr, 24);
      auto d2 = detail::read_at<int64_t> (hdr, 32);
      auto d3 = detail::read_at<int64_t> (hdr, 40);
      auto voff = detail::read_at<int64_t> (hdr, 168);
      auto slop = detail::read_at<double> (hdr, 176);
      auto intr = detail::read_at<double> (hdr, 184);

      if (swap)
        {
          datatype = detail::bswap (datatype);
          ndim = detail::bswap (ndim);
          d1 = detail::bswap (d1);
          d2 = detail::bswap (d2);
          d3 = detail::bswap (d3);
          voff = detail::bswap (voff);
          slop = detail::bswap (slop);
          intr = detail::bswap (intr);
        }

      if (ndim < 3)
        throw std::runtime_error (
            "read_nifti: expected a 3D volume in '" + path.string () + "'");

      Nx = static_cast<std::size_t> (d1);
      Ny = static_cast<std::size_t> (d2);
      Nz = static_cast<std::size_t> (d3);
      vox_offset = std::max (static_cast<std::size_t> (voff), hdr_size + 4);
      scl_slope = slop;
      scl_inter = intr;
    }

  // Read and convert voxel data
  const bool        apply_scale = (scl_slope != 0.0);
  const std::size_t n = Nx * Ny * Nz;

  file.seekg (static_cast<std::streamoff> (vox_offset));

  // Map NIfTI datatype code to element size for raw read
  static const struct
  {
    int16_t     code;
    std::size_t bytes;
  } dtype_sizes[] = {
    {   2, 1 },
    {   4, 2 },
    {   8, 4 },
    {  16, 4 },
    {  64, 8 },
    { 256, 1 },
    { 512, 2 },
    { 768, 4 }
  };
  std::size_t elem_bytes = 0;
  for (const auto &e : dtype_sizes)
    if (e.code == datatype)
      {
        elem_bytes = e.bytes;
        break;
      }

  if (elem_bytes == 0)
    throw std::runtime_error ("read_nifti: unsupported datatype "
                              + std::to_string (datatype) + " in '"
                              + path.string () + "'");

  std::vector<char> raw (n * elem_bytes);
  if (!file.read (raw.data (), static_cast<std::streamsize> (n * elem_bytes)))
    throw std::runtime_error (
        "read_nifti: unexpected EOF in '" + path.string () + "'");

  std::vector<scalar> data (n);

  switch (datatype)
    {
    case 2:
      detail::convert_voxels<scalar, uint8_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 4:
      detail::convert_voxels<scalar, int16_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 8:
      detail::convert_voxels<scalar, int32_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 16:
      detail::convert_voxels<scalar, float> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 64:
      detail::convert_voxels<scalar, double> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 256:
      detail::convert_voxels<scalar, int8_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 512:
      detail::convert_voxels<scalar, uint16_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    case 768:
      detail::convert_voxels<scalar, uint32_t> (
          data, raw, n, swap, apply_scale, scl_slope, scl_inter);
      break;
    default:
      break; // unreachable; handled above
    }

  return Grid<scalar>{ Nx, Ny, Nz, std::move (data) };
}

/// @brief Serialise a BoundaryMatrix to PHAT binary format
/// @param bm    boundary matrix to write
/// @param stem  output path without extension
/// @details
/// Produces two files:
///   <stem>.bin  - int64 stream consumed by `phat --binary`
///                 Layout: num_cells, then per cell: dim, bnd_size,
///                 bnd_indices...
///   <stem>.vals - flat binary array of num_cells doubles
///                 Use to map PHAT's (birth_col, death_col) pairs to
///                 birth/death values
template <typename scalar>
void write_phat_binary (
    BoundaryMatrix<scalar> const &bm, std::string const &stem)
{
  {
    std::ofstream f (stem + ".bin", std::ios::binary);
    if (!f)
      throw std::runtime_error ("cannot open " + stem + ".bin for writing");

    auto write_i64 = [&] (int64_t v) {
      f.write (reinterpret_cast<const char *> (&v), sizeof (v));
    };

    write_i64 (static_cast<int64_t> (bm.num_cells));
    uint64_t offset = 0;
    for (uint64_t p = 0; p < bm.num_cells; ++p)
      {
        const int64_t bnd_size = static_cast<int64_t> (2 * bm.dims[p]);
        write_i64 (static_cast<int64_t> (bm.dims[p]));
        write_i64 (bnd_size);
        for (int64_t b = 0; b < bnd_size; ++b)
          write_i64 (static_cast<int64_t> (
              bm.boundaries[offset + static_cast<uint64_t> (b)]));
        offset += static_cast<uint64_t> (bnd_size);
      }
  }

  {
    std::ofstream f (stem + ".vals", std::ios::binary);
    if (!f)
      throw std::runtime_error ("cannot open " + stem + ".vals for writing");

    f.write (reinterpret_cast<const char *> (bm.filt_values.data ()),
        static_cast<std::streamsize> (bm.num_cells * sizeof (double)));
  }
}

} // namespace plsf
// ****************************************************************************
