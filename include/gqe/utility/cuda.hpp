/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

#include <memory>

namespace gqe::utility {

/**
 * @brief Return the current CUDA device ID.
 */
rmm::cuda_device_id current_cuda_device_id();

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
