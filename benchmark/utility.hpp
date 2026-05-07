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

#include <gqe/utility/error.hpp>

namespace gqe::benchmark {

inline std::size_t get_memory_pool_size()
{
  std::size_t free_memory, total_memory;
  GQE_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  auto pool_size =
    total_memory / 284 * 256;  // ~90% of the total memory and have 256-byte alignment
  return pool_size;
}

}  // namespace gqe::benchmark
