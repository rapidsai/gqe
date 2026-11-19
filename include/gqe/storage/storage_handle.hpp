/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/shared_memory.hpp>
#include <gqe/utility/error.hpp>

#include <boost/interprocess/allocators/allocator.hpp>
#include <cuda/stream>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace gqe {
namespace storage {

using shared_memory_allocator = gqe::shared_memory_allocator;

/**
 * @brief Owns local storage by value for a process-local wrapper.
 *
 * @tparam Storage Concrete non-polymorphic storage type.
 */
template <class Storage>
class local_storage_handle {
 public:
  using storage_type = Storage;

  /**
   * @brief Construct the owned storage in place.
   */
  template <class... Args>
  explicit local_storage_handle(Args&&... args) : _storage(std::forward<Args>(args)...)
  {
  }

  /**
   * @brief Return the owned storage.
   */
  [[nodiscard]] storage_type const& get() const { return _storage; }

  /**
   * @brief Return the owned storage.
   */
  [[nodiscard]] storage_type& get() { return _storage; }

  /**
   * @brief Explicitly release payload buffers owned by the local storage.
   */
  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    _storage.deallocate(stream, mr);
  }

 private:
  storage_type _storage;
};

/**
 * @brief References storage that lives in the shared memory segment.
 *
 * @tparam Storage Concrete non-polymorphic shared-memory storage type.
 */
template <class Storage>
class shared_storage_handle {
 public:
  using storage_type = Storage;
  using handle_type  = shared_segment::handle_t;

  /**
   * @brief Construct a handle for an existing storage object in @p segment.
   */
  shared_storage_handle(shared_segment& segment, handle_type handle)
    : _segment(&segment), _handle(handle)
  {
  }

  shared_storage_handle(shared_storage_handle const&)            = delete;
  shared_storage_handle& operator=(shared_storage_handle const&) = delete;

  /**
   * @brief Resolve and return the shared storage object.
   *
   * @throw std::logic_error if this handle has already been invalidated.
   */
  [[nodiscard]] storage_type const& get() const
  {
    if (_segment == nullptr || _handle == handle_type{}) {
      throw std::logic_error("shared storage handle is invalid");
    }
    auto const* addr = _segment->get_address_from_handle(_handle);
    if (addr == nullptr) {
      throw std::logic_error("shared storage handle does not resolve to a storage object");
    }
    return *static_cast<storage_type const*>(addr);
  }

  /**
   * @brief Resolve and return the shared storage object.
   *
   * @throw std::logic_error if this handle has already been invalidated.
   */
  [[nodiscard]] storage_type& get()
  {
    if (_segment == nullptr || _handle == handle_type{}) {
      throw std::logic_error("shared storage handle is invalid");
    }
    auto* addr = _segment->get_address_from_handle(_handle);
    if (addr == nullptr) {
      throw std::logic_error("shared storage handle does not resolve to a storage object");
    }
    return *static_cast<storage_type*>(addr);
  }

  /**
   * @brief Release payload buffers, destroy the storage object in the segment, and invalidate this
   * handle.
   */
  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    if (_segment == nullptr || _handle == handle_type{}) {
      _segment = nullptr;
      _handle  = handle_type{};
      return;
    }

    auto* storage_addr = _segment->get_address_from_handle(_handle);
    GQE_EXPECTS(storage_addr != nullptr,
                "shared storage handle does not resolve to a storage object");
    auto* storage_ptr = static_cast<storage_type*>(storage_addr);
    storage_ptr->deallocate(stream, mr);
    _segment->destroy_ptr(storage_ptr);
    _segment = nullptr;
    _handle  = handle_type{};
  }

 private:
  shared_segment* _segment;
  handle_type _handle;
};

}  // namespace storage
}  // namespace gqe
