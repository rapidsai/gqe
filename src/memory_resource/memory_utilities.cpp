/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/memory_resource/memory_utilities.hpp>

namespace gqe {

namespace memory_resource {

std::unique_ptr<rmm::mr::device_memory_resource> create_static_memory_pool()
{
  using upstream_mr_type       = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
  using mr_type                = rmm::mr::pool_memory_resource<upstream_mr_type>;
  static auto upstream_cuda_mr = rmm::mr::cuda_memory_resource();
  static auto pool_size        = gqe::utility::default_device_memory_pool_size();
  static auto upstream_mr =
    std::make_shared<upstream_mr_type>(upstream_cuda_mr, pool_size, pool_size);
  return std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(
    upstream_mr, pool_size, pool_size);
}

}  // namespace memory_resource

}  // namespace gqe
