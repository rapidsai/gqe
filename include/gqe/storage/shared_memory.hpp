/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/storage/shared_memory_poison.hpp>
#include <gqe/storage/shared_segment.hpp>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/version_type.hpp>

#include <cstddef>

namespace gqe {

/// @brief Numeric handle for objects stored in the shared memory segment.
using shared_memory_handle_t = shared_segment::handle_t;

/**
 * @brief Share the poisoned flag through @p segment.
 *
 * Uses the flag at a fixed place in the segment (see @ref shared_segment_manager::poisoned_flag)
 * and sets it if this process is already poisoned. From then on,
 * @ref mark_shared_memory_poisoned also sets the segment's flag, and
 * @ref is_shared_memory_poisoned also reads it. A process attaches at most one segment at a time.
 *
 * Takes no lock in the segment, so it works even when one of the segment's locks is
 * unrecoverable. Task managers attach when they open the segment.
 *
 * @param segment The segment to attach. It must stay mapped until
 * @ref detach_shared_memory_poison_flag is called.
 */
void attach_shared_memory_poison_flag(shared_segment& segment) noexcept;

/**
 * @brief Stop sharing the poisoned flag through @p segment, before it is unmapped.
 *
 * Does nothing if @p segment is not the attached segment. The process-local flag is kept.
 */
void detach_shared_memory_poison_flag(shared_segment const& segment) noexcept;

// ----- Shared-memory allocator -----

/**
 * @brief Allocator for containers stored in the shared-memory segment.
 *
 * Wraps `boost::interprocess::allocator`, which takes an arena's allocator lock on every allocation
 * and deallocation. Once that lock is unrecoverable, @ref shared_segment throws
 * @ref shared_memory_unrecoverable_error from both. A container's destructor deallocates while
 * being `noexcept`, so with a plain Boost allocator the error would call `std::terminate`. This
 * allocator instead treats it like `memory_resource::shared_memory_resource` does: `deallocate`
 * sets the poisoned flag and leaks the block.
 *
 * It declares allocator version 1, so Boost containers free memory only through `deallocate`, never
 * through version-2 calls such as `deallocate_one` that would throw. It adds no data members, so
 * containers keep the layout they had with the Boost allocator.
 *
 * @tparam T Type of the allocated objects.
 */
template <typename T>
class shared_memory_container_allocator
  : public boost::interprocess::allocator<T, shared_segment::segment_manager> {
  using base_type = boost::interprocess::allocator<T, shared_segment::segment_manager>;

 public:
  using typename base_type::pointer;
  using typename base_type::segment_manager;
  using typename base_type::size_type;
  using version = boost::interprocess::version_type<shared_memory_container_allocator, 1>;

  template <typename U>
  struct rebind {
    using other = shared_memory_container_allocator<U>;
  };

  /// @brief Construct from the segment manager of the segment to allocate from.
  shared_memory_container_allocator(segment_manager* manager) : base_type(manager) {}

  /// @brief Construct from an allocator for another type, sharing its segment.
  template <typename U>
  shared_memory_container_allocator(shared_memory_container_allocator<U> const& other)
    : base_type(other.get_segment_manager())
  {
  }

  /// @brief Return a block to the segment, or leak it and set the poisoned flag.
  void deallocate(pointer const& ptr, size_type count) noexcept
  {
    try {
      base_type::deallocate(ptr, count);
    } catch (...) {
      if (!record_if_shared_memory_unrecoverable(std::current_exception(),
                                                 "shared-memory container deallocation")) {
        mark_shared_memory_poisoned(
          "shared-memory container deallocation: the arena allocator failed to free a block");
      }
    }
  }

  template <typename U>
  [[nodiscard]] bool operator==(shared_memory_container_allocator<U> const& other) const noexcept
  {
    return this->get_segment_manager() == other.get_segment_manager();
  }
};

/// @brief Allocator for metadata stored in the shared memory segment.
using shared_memory_allocator = shared_memory_container_allocator<std::byte>;

}  // namespace gqe
