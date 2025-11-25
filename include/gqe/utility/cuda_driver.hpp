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

#pragma once

#include <llvm/ADT/ArrayRef.h>

#include <optional>
#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>
#include <string>

/**
 * @brief CUDA driver API utilities for launching JIT-compiled kernels.
 *
 * These utilties are not in `cuda.hpp` to separate the CUDA driver and runtime
 * API utilities. Further reasons are:
 *
 *  - All compiler-specific functionality is behind a CMake flag to optionally
 *    enable/disable the query compiler.
 *  - Allow the compiler utilities to be compiled separately from the GQE
 *    library for unit tests.
 *
 * Note that this header file takes care to _not_ include `cuda.h`, because that
 * is a large, 26k LOC C-style header.
 */
namespace gqe {
namespace utility {

namespace detail {

/**
 * @brief An opaque struct that wraps `CUlibrary`.
 */
struct KernelLibrary;

};  // namespace detail

/**
 * @brief Initialize the CUDA driver.
 *
 * Error checking wrapper of `cuInit()`.
 *
 * @param flags The flags of `cuInit`.
 */
void safeCuInit(unsigned int flags = 0);

/**
 * Detect the hardware architecture of the CUDA device.
 *
 * For example, `sm_80` for Ampere.
 *
 * @param deviceId The CUDA device ID to get.
 *
 * @return The architecture ordinal number.
 */
[[nodiscard]] int32_t detectDeviceArchitecture(int32_t deviceId);

/**
 * @brief A kernel launch configuration.
 */
struct LaunchConfiguration {
  int32_t gridSize;           ///< The launch grid size.
  int32_t blockSize;          ///< The launch block size.
  int32_t sharedMemoryBytes;  ///< The dynamic shared memory size.
};

/**
 * @brief Utility to launch kernels with the CUDA Library API.
 *
 * The launcher uses the CUDA context-independent module loading feature. It
 * takes a `cubin` (represented as a char array) and loads the CUDA library from
 * it. The launcher manages the CUDA library object according to RAII.
 *
 * References:
 * https://developer.nvidia.com/blog/cuda-context-independent-module-loading
 */
class KernelLauncher final {
 public:
  /**
   * @brief Create a kernel launcher.
   *
   * @param[in] cubin The device binary. Passing PTX is also possible, in which
   * case it will be compiled while loading the library.
   */
  explicit KernelLauncher(llvm::ArrayRef<char> cubin);
  ~KernelLauncher();

  KernelLauncher(KernelLauncher&)            = delete;
  KernelLauncher& operator=(KernelLauncher&) = delete;

  KernelLauncher(KernelLauncher&&)            = default;
  KernelLauncher& operator=(KernelLauncher&&) = default;

  /**
   * @brief Detect a "reasonable" kernel launch configuration.
   *
   * Calculates a launch configuration that achieves a "reasonable" occupancy
   * using the `cuOccupancyMaxPotentialBlockSize` function.
   *
   * Reference:
   * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html
   *
   * @param kernelName The kernel to configure.
   * @param blockSizeLimit The maximum block size that the kernel is designed
   * to handle.
   */
  [[nodiscard]] LaunchConfiguration detectLaunchConfiguration(
    std::string const& kernelName, std::optional<int32_t> blockSizeLimit = std::nullopt);

  /**
   * @brief Launch a kernel with the CUDA Library API.
   *
   * @param[in] kernelName The name of the kernel to call.
   * @param[in] config The kernel launch configuration.
   * @param[in,out] arguments The arguments provided to the kernel.
   * @param[in] stream The CUDA stream to launch the kernel on.
   */
  void launch(std::string const& kernelName,
              LaunchConfiguration const& config,
              llvm::MutableArrayRef<void*> arguments,
              rmm::cuda_stream_view stream = cudf::get_default_stream());

 private:
  std::unique_ptr<detail::KernelLibrary> _library;
};

}  // namespace utility
}  // namespace gqe
