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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief System memory resource for pageable memory.
 *
 * Calls `new` and `delete` to allocate and deallocate memory resources.
 */
class system_memory_resource : public rmm::mr::device_memory_resource {
 public:
  system_memory_resource()           = default;
  ~system_memory_resource() override = default;

 private:
  static constexpr auto _allocation_alignment =
    std::align_val_t{256}; /**< `rmm::mr::device_memory_resource` specifies that all allocations are
                            256-byte aligned. */

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;

  void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override;
};

}  // namespace memory_resource

}  // namespace gqe
