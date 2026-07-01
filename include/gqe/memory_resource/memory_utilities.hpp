/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/helpers.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace gqe {

namespace memory_resource {

/**
 * @brief Create a shared, statically allocated memory pool.
 *
 * Test classes should use this memory resource when they create a `gqe::task_manager_context`
 * to speed up test execution.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible> create_static_memory_pool();

/**
 * @brief Returns the approximate specified percent of the given memory, aligned (down) to the
 * nearest CUDA allocation size.
 *
 * @param memory_bytes The total memory in bytes.
 * @param percent The percent of memory to return.
 * @return The calculated memory size in bytes, aligned down to CUDA allocation boundaries.
 */
std::size_t percent_of_memory(std::size_t memory_bytes, int percent);

}  // namespace memory_resource

}  // namespace gqe
