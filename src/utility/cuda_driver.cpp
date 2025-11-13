/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/utility/cuda_driver.hpp>

#include <gqe/utility/error.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>

#include <rmm/cuda_stream_view.hpp>

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

/**
 * @brief Check CUDA function error code
 *
 * Cannot use GQE_CUDA_TRY, because it's for the runtime API. This macro
 * supports the driver API.
 */
#define CHECK_CUDA(statement, message)                    \
  {                                                       \
    assertCuda((statement), message, __FILE__, __LINE__); \
  }

namespace {

/**
 * @brief Helper for `CHECK_CUDA` macro
 */
void assertCuda(CUresult code, const llvm::StringRef message, const llvm::StringRef file, int line)
{
  if (code != CUDA_SUCCESS) {
    const char* error;
    cuGetErrorString(code, &error);

    auto error_message = llvm::Twine{"CUDA error at "} + file + ":" + llvm::Twine(line) + " with " +
                         error + ": " + message;

    throw gqe::cuda_error{error_message.str()};
  }
}

/**
 * @brief Convert a `CUkernel` type to a `CUfunction` type.
 *
 * Context-independent kernels can be launched directly, instead of first
 * retrieving a CUfunction (c.f. `cuLaunchKernel` documentation).
 *
 * Reference:
 * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
 */
inline CUfunction convertKernelType(CUkernel kernel) noexcept
{
  return reinterpret_cast<CUfunction>(kernel);
}

}  // namespace

namespace gqe {
namespace utility {

namespace detail {
struct KernelLibrary {
  CUlibrary handle;
};
}  // namespace detail

void safeCuInit(unsigned int flags)
{
  CHECK_CUDA(cuInit(flags), "Failed to initialize CUDA with cuInit().");
}

int32_t detectDeviceArchitecture(int32_t deviceId)
{
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, deviceId), "Failed to get CUDA device");

  int32_t arch_major = 0;
  CHECK_CUDA(
    cuDeviceGetAttribute(&arch_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
    "Failed to detect device architecture");

  int32_t arch_minor = 0;
  CHECK_CUDA(
    cuDeviceGetAttribute(&arch_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
    "Failed to detect device architecture");

  return arch_major * 10 + arch_minor;
}

KernelLauncher::KernelLauncher(llvm::ArrayRef<char> cubin)
  : _library(std::make_unique<detail::KernelLibrary>())
{
  CHECK_CUDA(
    cuLibraryLoadData(&_library->handle, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
    "Failed to load library");

  uint32_t num_kernels;
  CHECK_CUDA(cuLibraryGetKernelCount(&num_kernels, _library->handle), "Failed to get num kernels");
  assert(num_kernels > 0 && "expected the cubin to contain at least one kernel");
}

KernelLauncher::~KernelLauncher()
{
  CHECK_CUDA(cuLibraryUnload(_library->handle), "Failed to unload library");
}

LaunchConfiguration KernelLauncher::detectLaunchConfiguration(std::string const& kernelName,
                                                              std::optional<int32_t> blockSizeLimit)
{
  LaunchConfiguration config = {0};

  CUkernel kernel;
  CHECK_CUDA(cuLibraryGetKernel(&kernel, _library->handle, kernelName.c_str()),
             "Failed to load kernel");

  CHECK_CUDA(cuOccupancyMaxPotentialBlockSize(&config.gridSize,
                                              &config.blockSize,
                                              convertKernelType(kernel),
                                              /* blockSizeToDynamicSMemSize = */ nullptr,
                                              /* dynamicSMemSize = */ 0,
                                              blockSizeLimit.value_or(0)),
             "Failed to detect launch configuration.");

  return config;
}

void KernelLauncher::launch(std::string const& kernelName,
                            LaunchConfiguration const& config,
                            llvm::MutableArrayRef<void*> arguments,
                            rmm::cuda_stream_view stream)
{
  CUkernel kernel;
  CHECK_CUDA(cuLibraryGetKernel(&kernel, _library->handle, kernelName.c_str()),
             "Failed to load kernel");

  // 2nd and 3rd dimensions of grid size and block size are all set to `1`.
  //
  // `CUstream` (driver API) and `cudaStream_t` (runtime API) are the same type.
  //
  // Kernel parameters are provided.
  //
  // No extra options are provided.
  //
  // References:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html
  CHECK_CUDA(cuLaunchKernel(convertKernelType(kernel),
                            /* gridDimX = */ config.gridSize,
                            /* gridDimY = */ 1,
                            /* gridDimZ = */ 1,
                            /* blockDimX = */ config.blockSize,
                            /* blockDimY = */ 1,
                            /* blockDimZ = */ 1,
                            /* sharedMemBytes = */ config.sharedMemoryBytes,
                            /* hStream = */ stream.value(),
                            /* kernelParams = */ arguments.data(),
                            /* extra = */ nullptr),
             "Failed to launch kernel");
}

}  // namespace utility
}  // namespace gqe
