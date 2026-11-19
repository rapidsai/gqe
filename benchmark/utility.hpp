/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/utility/error.hpp>

#include <cuda/memory_resource>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <optional>

namespace gqe::benchmark {

inline std::size_t get_memory_pool_size()
{
  std::size_t free_memory, total_memory;
  GQE_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  auto const free_based_pool_size =
    free_memory / 284 * 256;  // ~90% of currently free memory with 256-byte alignment
  return free_based_pool_size;
}

/**
 * @brief Build an owning, type-erased CUDA pool resource sized for benchmarking.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> make_pool_resource(
  std::size_t pool_size = get_memory_pool_size())
{
  return cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::pool_memory_resource{
    cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    pool_size,
    std::make_optional(pool_size)}};
}

}  // namespace gqe::benchmark
