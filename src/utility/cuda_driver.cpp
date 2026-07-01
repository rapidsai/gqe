/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

namespace {

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

namespace gqe::utility {

namespace detail {
struct KernelLibrary {
  CUlibrary handle;
};
}  // namespace detail

void safeCuInit(unsigned int flags)
{
  GQE_CU_TRY(cuInit(flags), "Failed to initialize CUDA with cuInit().");
}

int32_t detectDeviceArchitecture(int32_t deviceId)
{
  CUdevice device;
  GQE_CU_TRY(cuDeviceGet(&device, deviceId), "Failed to get CUDA device");

  int32_t arch_major = 0;
  GQE_CU_TRY(
    cuDeviceGetAttribute(&arch_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
    "Failed to detect device architecture");

  int32_t arch_minor = 0;
  GQE_CU_TRY(
    cuDeviceGetAttribute(&arch_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
    "Failed to detect device architecture");

  return arch_major * 10 + arch_minor;
}

KernelLauncher::KernelLauncher(llvm::ArrayRef<char> cubin)
  : _library(std::make_unique<detail::KernelLibrary>())
{
  GQE_CU_TRY(
    cuLibraryLoadData(&_library->handle, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
    "Failed to load library");

  uint32_t num_kernels;
  GQE_CU_TRY(cuLibraryGetKernelCount(&num_kernels, _library->handle), "Failed to get num kernels");
  assert(num_kernels > 0 && "expected the cubin to contain at least one kernel");
}

KernelLauncher::~KernelLauncher()
{
  GQE_CU_TRY_NO_THROW(cuLibraryUnload(_library->handle), "Failed to unload library");
}

LaunchConfiguration KernelLauncher::detectLaunchConfiguration(std::string const& kernelName,
                                                              std::optional<int32_t> blockSizeLimit)
{
  LaunchConfiguration config = {0};

  CUkernel kernel;
  GQE_CU_TRY(cuLibraryGetKernel(&kernel, _library->handle, kernelName.c_str()),
             "Failed to load kernel");

  GQE_CU_TRY(cuOccupancyMaxPotentialBlockSize(&config.gridSize,
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
  GQE_CU_TRY(cuLibraryGetKernel(&kernel, _library->handle, kernelName.c_str()),
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
  GQE_CU_TRY(cuLaunchKernel(convertKernelType(kernel),
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

}  // namespace gqe::utility
