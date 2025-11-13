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
