/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <nvtx3/nvtx3.hpp>

#include <cstddef>

namespace gqe {
// Forward declaration to escape include cycle.
//
// The struct is defined in `<gqe/device_properties.hpp>`.
struct device_properties;
}  // namespace gqe

namespace gqe::utility {

namespace detail {
/**
 * @brief Implementation of `detect_launch_grid_size()`.
 *
 * The function is wrapped by a template function that takes a non-void
 * `kernel` argument type.
 */
int detect_launch_grid_size(gqe::device_properties const& device_properties,
                            void const* const kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes);
}  // namespace detail

/**
 * @brief Return the current CUDA device ID.
 */
rmm::cuda_device_id current_cuda_device_id();

/**
 * @brief Detect a "reasonable" grid size for the kernel.
 *
 * Calculates a "reasonable" grid size based on the theoretical occupancy using
 * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
 *
 * Reference:
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
 *
 * @param[in] device_properties The GQE device properties cache.
 * @param[in] kernel The kernel function.
 * @param[in] block_size The block size used for the kernel launch.
 * @param[in] dynamic_shared_memory_bytes The dynamic shared memory size in bytes used for the
 * kernel launch.
 *
 * @return The grid size.
 */
template <typename KernelType>
int detect_launch_grid_size(const gqe::device_properties& device_properties,
                            const KernelType kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes = 0)
{
  // Cast the kernel function pointer to a void data pointer. g++ emits warning #167-D when passing
  // kernel directly to a `void*` argument.
  return detail::detect_launch_grid_size(device_properties,
                                         reinterpret_cast<void const*>(kernel),
                                         block_size,
                                         dynamic_shared_memory_bytes);
}

/**
 * @brief NVTX domain which should be used across the GQE project
 */
struct gqe_nvtx_domain {
  static constexpr char const* name{"GQE"};
};

/**
 * @brief A RAII object for creating a NVTX range local to a thread within the GQE domain
 */
using nvtx_scoped_range = nvtx3::scoped_range_in<gqe_nvtx_domain>;

/**
 * @brief Create a NVTX marker within the GQE domain.
 */
template <typename... Args>
inline void nvtx_mark(Args&&... args)
{
  nvtx3::mark_in<gqe_nvtx_domain>(std::forward<Args>(args)...);
}

}  // namespace gqe::utility
