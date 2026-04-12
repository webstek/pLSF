// volume_io.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementations of the volume I/O functions declared in volume_io.hpp.
//
// Implementation status:
//   read_raw          – complete (headerless binary)
//   write_raw         – complete
//   write_pairs_csv   – complete
//   write_pairs_bin   – complete
//   read_nifti        – stub (see TODO below)
//   read_metaimage    – stub (see TODO below)
//   read_volume       – dispatches to the above; complete once stubs are done
// ─────────────────────────────────────────────────────────────────────────────

#include "volume_io.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace glsf::io {

// ═════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═════════════════════════════════════════════════════════════════════════════
namespace {

// Detect native byte order at compile time.
inline bool host_is_big_endian() noexcept {
    const uint32_t probe = 0x01020304u;
    uint8_t bytes[4];
    std::memcpy(bytes, &probe, 4);
    return bytes[0] == 0x01;
}

// In-place byte reversal for an array of `count` elements of size `elem_size`.
void byteswap_array(char* data, std::size_t count, std::size_t elem_size) {
    for (std::size_t i = 0; i < count; ++i) {
        char* p = data + i * elem_size;
        std::reverse(p, p + elem_size);
    }
}

} // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// read_raw
// ─────────────────────────────────────────────────────────────────────────────
// Reads a headerless flat-binary file of `nx*ny*nz` Scalar values.
// Assumes native byte order; swap if your data is big-endian on a
// little-endian host (or vice-versa) by calling byteswap_array() after reading.
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_raw(const std::filesystem::path& path,
         std::size_t nx, std::size_t ny, std::size_t nz,
         std::size_t offset_bytes)
{
    const std::size_t n = nx * ny * nz;

    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("read_raw: cannot open '" + path.string() + "'");

    if (offset_bytes > 0)
        file.seekg(static_cast<std::streamoff>(offset_bytes));

    std::vector<Scalar> data(n);
    if (!file.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(n * sizeof(Scalar))))
        throw std::runtime_error("read_raw: unexpected EOF in '" + path.string() + "'");

    VolumeMetadata meta;
    meta.dims        = {nx, ny, nz};
    meta.source_file = path.string();
    meta.data_type   = (sizeof(Scalar) == 2) ? "int16"
                     : (sizeof(Scalar) == 4) ? "float32"
                     :                         "float64";

    return {CubicalGrid3D<Scalar>(nx, ny, nz, std::move(data)), meta};
}

// ═════════════════════════════════════════════════════════════════════════════
// read_nifti
// ─────────────────────────────────────────────────────────────────────────────
// TODO: implement NIfTI-1 / NIfTI-2 reader.
//
// Steps:
//   1. Open the file and read the first 4 bytes to determine header format:
//        348 (little-endian) → NIfTI-1   (.nii or paired .hdr/.img)
//        540 (little-endian) → NIfTI-2
//   2. Parse the fixed-size header struct (nifti_1_header / nifti_2_header).
//        dim[1..3]   → grid dimensions nx, ny, nz
//        pixdim[1..3]→ voxel spacing
//        datatype    → NIFTI_TYPE_FLOAT32 (16), INT16 (4), …
//        vox_offset  → byte offset to image data within the .nii file
//        scl_slope, scl_inter → affine rescaling; apply if slope ≠ 0
//   3. For .nii.gz: transparently decompress via zlib or pipe through gzip.
//      Tip: use a custom std::streambuf wrapping zlib's inflate() API.
//   4. Read the voxel array, cast/rescale to Scalar, construct CubicalGrid3D.
//
// Optional third-party library:
//   nifticlib  https://github.com/NIFTI-Imaging/nifti_clib
//     #include <nifti/nifti1_io.h>
//     nifti_image* img = nifti_image_read(path.c_str(), 1);
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_nifti(const std::filesystem::path& /*path*/)
{
    throw std::runtime_error("read_nifti: not yet implemented");
}

// ═════════════════════════════════════════════════════════════════════════════
// read_metaimage
// ─────────────────────────────────────────────────────────────────────────────
// TODO: implement MetaImage (.mhd + .raw) reader.
//
// Steps:
//   1. Parse the ASCII .mhd header line-by-line (format: "Key = Value\n").
//      Required fields:
//        NDims            – must be 3
//        DimSize          – three integers (nx ny nz)
//        ElementType      – "MET_FLOAT", "MET_SHORT", "MET_UCHAR", …
//        ElementSpacing   – three doubles (mm per voxel)
//        ElementDataFile  – relative path to the companion .raw file, or "LOCAL"
//        BinaryDataByteOrderMSB – "True" or "False"
//   2. Resolve ElementDataFile relative to the .mhd directory.
//   3. Byte-swap if BinaryDataByteOrderMSB differs from host endianness
//      (use byteswap_array() above).
//   4. Cast each element to Scalar, construct CubicalGrid3D.
//
// Optional third-party library:
//   ITK SimpleITK  https://simpleitk.org  (reads MetaImage natively)
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_metaimage(const std::filesystem::path& /*path*/)
{
    throw std::runtime_error("read_metaimage: not yet implemented");
}

// ═════════════════════════════════════════════════════════════════════════════
// read_volume  (extension-based dispatch)
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
std::pair<CubicalGrid3D<Scalar>, VolumeMetadata>
read_volume(const std::filesystem::path& path)
{
    const auto ext  = path.extension().string();
    const auto stem = path.stem().extension().string();   // for .nii.gz

    if (ext == ".nii" || (ext == ".gz" && stem == ".nii"))
        return read_nifti<Scalar>(path);

    if (ext == ".mhd")
        return read_metaimage<Scalar>(path);

    if (ext == ".raw" || ext == ".bin")
        throw std::runtime_error(
            "read_volume: raw files require explicit dimensions. "
            "Call read_raw(path, nx, ny, nz) directly.");

    throw std::runtime_error(
        "read_volume: unsupported file extension '" + ext + "'");
}

// ═════════════════════════════════════════════════════════════════════════════
// write_pairs_csv
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void write_pairs_csv(const std::filesystem::path&               path,
                     const std::vector<PersistencePair<Scalar>>& pairs)
{
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("write_pairs_csv: cannot open '" + path.string() + "'");

    file << "birth,death,dimension\n";
    for (const auto& p : pairs)
        file << p.birth << ',' << p.death << ',' << p.dimension << '\n';
}

// ═════════════════════════════════════════════════════════════════════════════
// write_pairs_bin
// ─────────────────────────────────────────────────────────────────────────────
// Binary layout (all values in host byte order):
//   uint64_t  n                          – number of pairs
//   for each pair:
//     Scalar   birth
//     Scalar   death
//     int32_t  dimension
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void write_pairs_bin(const std::filesystem::path&               path,
                     const std::vector<PersistencePair<Scalar>>& pairs)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("write_pairs_bin: cannot open '" + path.string() + "'");

    const uint64_t n = static_cast<uint64_t>(pairs.size());
    file.write(reinterpret_cast<const char*>(&n), sizeof n);

    for (const auto& p : pairs) {
        file.write(reinterpret_cast<const char*>(&p.birth),     sizeof p.birth);
        file.write(reinterpret_cast<const char*>(&p.death),     sizeof p.death);
        const int32_t dim = static_cast<int32_t>(p.dimension);
        file.write(reinterpret_cast<const char*>(&dim),         sizeof dim);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// write_raw
// ═════════════════════════════════════════════════════════════════════════════
template <typename Scalar>
void write_raw(const std::filesystem::path& path,
               const CubicalGrid3D<Scalar>& grid)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("write_raw: cannot open '" + path.string() + "'");

    const auto& data = grid.data();
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(Scalar)));
}

// ── Explicit template instantiations ─────────────────────────────────────────
#define GLSF_INSTANTIATE_IO(T)                                                     \
    template std::pair<CubicalGrid3D<T>, VolumeMetadata>                           \
        read_raw<T>      (const std::filesystem::path&,                            \
                          std::size_t, std::size_t, std::size_t, std::size_t);     \
    template std::pair<CubicalGrid3D<T>, VolumeMetadata>                           \
        read_nifti<T>    (const std::filesystem::path&);                           \
    template std::pair<CubicalGrid3D<T>, VolumeMetadata>                           \
        read_metaimage<T>(const std::filesystem::path&);                           \
    template std::pair<CubicalGrid3D<T>, VolumeMetadata>                           \
        read_volume<T>   (const std::filesystem::path&);                           \
    template void write_pairs_csv<T>(const std::filesystem::path&,                 \
                                     const std::vector<PersistencePair<T>>&);      \
    template void write_pairs_bin<T>(const std::filesystem::path&,                 \
                                     const std::vector<PersistencePair<T>>&);      \
    template void write_raw<T>      (const std::filesystem::path&,                 \
                                     const CubicalGrid3D<T>&);

GLSF_INSTANTIATE_IO(float)
GLSF_INSTANTIATE_IO(double)
GLSF_INSTANTIATE_IO(int16_t)

#undef GLSF_INSTANTIATE_IO

} // namespace glsf::io
