#pragma once
// sycl_utils.hpp
// ─────────────────────────────────────────────────────────────────────────────
// SYCL device selection helpers and diagnostic utilities.
//
// Usage:
//   sycl::queue q = glsf::make_queue(glsf::DevicePreference::GPU);
//   glsf::print_device_info(q);
// ─────────────────────────────────────────────────────────────────────────────

#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

namespace glsf {

// ── Device preference ──────────────────────────────────────────────────────────
enum class DevicePreference { Default, CPU, GPU };

// ── Create a SYCL queue for the requested device class ────────────────────────
// Falls back to the CPU if a GPU is requested but unavailable.
inline sycl::queue make_queue(DevicePreference pref            = DevicePreference::GPU,
                               bool             enable_profiling = false)
{
    const sycl::property_list props = enable_profiling
        ? sycl::property_list{sycl::property::queue::enable_profiling{}}
        : sycl::property_list{};

    sycl::device dev;
    try {
        switch (pref) {
        case DevicePreference::GPU:     dev = sycl::device{sycl::gpu_selector_v};     break;
        case DevicePreference::CPU:     dev = sycl::device{sycl::cpu_selector_v};     break;
        case DevicePreference::Default: dev = sycl::device{sycl::default_selector_v}; break;
        }
    } catch (const sycl::exception&) {
        if (pref == DevicePreference::GPU) {
            std::cerr << "[gLSF] Warning: no GPU found, falling back to CPU device.\n";
            dev = sycl::device{sycl::cpu_selector_v};
        } else {
            throw;
        }
    }

    return sycl::queue{dev, props};
}

// ── Print a human-readable summary of the device backing a queue ──────────────
inline void print_device_info(const sycl::queue& q,
                               std::ostream&      os = std::cout)
{
    const auto& dev = q.get_device();
    os << "SYCL device : " << dev.get_info<sycl::info::device::name>()           << '\n'
       << "  Vendor    : " << dev.get_info<sycl::info::device::vendor>()          << '\n'
       << "  Driver    : " << dev.get_info<sycl::info::device::driver_version>()  << '\n'
       << "  Max CUs   : " << dev.get_info<sycl::info::device::max_compute_units>()<< '\n'
       << "  Global mem: "
       << dev.get_info<sycl::info::device::global_mem_size>() / (1024ULL * 1024ULL)
       << " MiB\n";
}

} // namespace glsf
