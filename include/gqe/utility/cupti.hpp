/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <rmm/cuda_device.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {
namespace utility {

namespace detail {
class user_range_profiler_impl;
}  // namespace detail

/**
 * @brief CUPTI user range profiler.
 *
 * The profiler measures CUPTI metrics for a user-defined range. The range is defined by starting
 * and stopping the profiler. All kernels, cudaMemcpys, etc. contained within the range are
 * profiled.
 *
 * The implementation is opaque as not to include low-level CUPTI and CUDA driver headers in this
 * header file.
 *
 * @pre To ensure low overhead, the profiler performs only a single profiling pass. The metrics must
 * not incur multiple profiling passes.
 *
 * References:
 * CUDA v.13.0 `extras/CUPTI/samples/range_profiling/range_profiling.cu`
 * https://docs.nvidia.com/cupti/api/group__CUPTI__RANGE__PROFILER__API.html
 * https://docs.nvidia.com/cupti/tutorial/tutorial.html#id6
 */
class user_range_profiler {
 public:
  /**
   * @brief A measured profile.
   */
  struct profile {
    std::unordered_map<std::string, double> metric_values;  /// The metric values measured.
  };

  /**
   * @brief A profiler configuration.
   */
  struct configuration {
    rmm::cuda_device_id device_id;     /// The device to profile.
    std::vector<std::string> metrics;  /// The metrics to profile.
  };

  /**
   * @brief Construct a new profiler.
   *
   * Performs any heavy-weight initialization necessary for profiling.
   *
   * @pre The CUDA context must be initialized before constructing this object.
   *
   * @throws `gqe::cuda_error` for any errors encountered during CUPTI API calls.
   * @throws std::invalid_argument Thrown if multiple passes are required to profile the list of
   * metrics.
   */
  user_range_profiler(configuration config);

  /**
   * @brief Destruct the profiler.
   *
   * @pre The CUDA context must be the same as in the constructor.
   *
   * @throws `gqe::cuda_error` when CUPTI returns an error code.
   * @throws `std::logic_error` if `teardown` is called on a profiler without previous `setup` call
   * or on a profiler that is running.
   */
  ~user_range_profiler() noexcept(false);

  user_range_profiler(user_range_profiler&)            = delete;
  user_range_profiler& operator=(user_range_profiler&) = delete;

  user_range_profiler(user_range_profiler&&);
  user_range_profiler& operator=(user_range_profiler&&);

  /**
   * @brief Start profiling.
   *
   * Creates a profiling range.
   *
   * Performs only light-weight initialization.
   *
   * @throws `gqe::cuda_error` when CUPTI returns an error code.
   * @throws `std::logic_error` if the profiler is not setup, or already running.
   */
  void start();

  /**
   * @brief Stop profiling.
   *
   * @throws `gqe::cuda_error` when CUPTI returns an error code.
   * @throws `std::logic_error` if the profiler is not running.
   *
   ** @return The measured profile.
   */
  [[nodiscard]] profile stop();

 private:
  std::unique_ptr<detail::user_range_profiler_impl> _impl;
};

}  // namespace utility
}  // namespace gqe
