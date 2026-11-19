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

#include <gqe/memory_resource/shared_memory_resource.hpp>

#include <cassert>
#include <exception>
#include <stdexcept>

namespace gqe {

namespace memory_resource {

shared_memory_resource::shared_memory_resource(shared_segment* non_owning_segment)
  : _non_owning_segment(non_owning_segment)
{
  if (_non_owning_segment == nullptr) {
    throw std::invalid_argument("Shared memory resource requires a non-null segment");
  }
}

void* shared_memory_resource::allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
{
  if (0 == bytes) { return nullptr; }

  return _non_owning_segment->allocate_aligned(bytes, alignment);
}

void shared_memory_resource::deallocate(cuda::stream_ref,
                                        void* ptr,
                                        std::size_t,
                                        std::size_t alignment) noexcept
{
  if (nullptr == ptr) { return; }
  assert(rmm::is_pointer_aligned(ptr, alignment));

  // Deallocation cannot report failure, so an unrecoverable allocator lock poisons the segment
  // instead of terminating the process. The block is leaked, and the segment is discarded when the
  // node tears down.
  try {
    _non_owning_segment->deallocate(ptr);
  } catch (...) {
    if (!record_if_shared_memory_unrecoverable(std::current_exception(),
                                               "shared_memory_resource::deallocate")) {
      // Any other failure also leaves the allocator's bookkeeping in an unknown state.
      mark_shared_memory_poisoned(
        "shared_memory_resource::deallocate: the arena allocator failed to free a block");
    }
  }
}

}  // namespace memory_resource

}  // namespace gqe
