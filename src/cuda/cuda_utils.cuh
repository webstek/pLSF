#pragma once
// ****************************************************************************
/// @file cuda/cuda_utils.cuh
/// @author Kyle Webster
/// @version 0.1
/// @date 13 Apr 2026
/// @brief CUDA device selection helpers and diagnostic utilities
// ****************************************************************************

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace plsf
{

//  Error handling 

#define CUDA_CHECK(call)                                                      \
  do                                                                          \
    {                                                                         \
      cudaError_t err = (call);                                               \
      if (err != cudaSuccess)                                                 \
        throw std::runtime_error (                                            \
            std::string ("CUDA error at ") + __FILE__ + ":"                   \
            + std::to_string (__LINE__) + " — "                               \
            + cudaGetErrorString (err));                                       \
    }                                                                         \
  while (0)

//  Device preference 

enum class DevicePreference
{
  Default,
  CPU, ///< not supported for CUDA; throws
  GPU
};

/// @brief Select and initialise a CUDA device
/// @param pref  Device preference (only GPU and Default are valid)
/// @return      Selected device ID
inline int select_device (DevicePreference pref = DevicePreference::GPU)
{
  if (pref == DevicePreference::CPU)
    throw std::runtime_error (
        "CUDA backend does not support CPU device selection");

  int device_count = 0;
  CUDA_CHECK (cudaGetDeviceCount (&device_count));
  if (device_count == 0)
    throw std::runtime_error ("No CUDA-capable devices found");

  int dev = 0; // default: use device 0
  CUDA_CHECK (cudaSetDevice (dev));
  return dev;
}

/// @brief Print a human-readable summary of the active CUDA device
inline void print_device_info (int dev = 0, std::ostream &os = std::cout)
{
  cudaDeviceProp prop;
  CUDA_CHECK (cudaGetDeviceProperties (&prop, dev));

  os << "CUDA device : " << prop.name << '\n'
     << "  Compute   : " << prop.major << "." << prop.minor << '\n'
     << "  SMs       : " << prop.multiProcessorCount << '\n'
     << "  Global mem: " << prop.totalGlobalMem / (1024ULL * 1024ULL)
     << " MiB\n";
}

} // namespace plsf
