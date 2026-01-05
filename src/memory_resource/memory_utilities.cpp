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

#include <gqe/memory_resource/memory_utilities.hpp>

#include <gqe/executor/optimization_parameters.hpp>

#include <rmm/aligned.hpp>

namespace gqe {

namespace memory_resource {

std::size_t percent_of_memory(std::size_t memory_bytes, int percent)
{
  return rmm::align_down((memory_bytes * percent) / 100, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

std::unique_ptr<rmm::mr::device_memory_resource> create_static_memory_pool()
{
  using upstream_mr_type        = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
  using mr_type                 = rmm::mr::pool_memory_resource<upstream_mr_type>;
  static auto upstream_cuda_mr  = rmm::mr::cuda_memory_resource();
  static auto initial_pool_size = optimization_parameters{}.initial_query_memory;
  static auto max_pool_size     = optimization_parameters{}.max_query_memory;
  static auto upstream_mr =
    std::make_shared<upstream_mr_type>(upstream_cuda_mr, initial_pool_size, max_pool_size);
  return std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(
    upstream_mr, initial_pool_size, max_pool_size);
}

}  // namespace memory_resource

}  // namespace gqe
