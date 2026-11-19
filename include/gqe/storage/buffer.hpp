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

#include <boost/interprocess/offset_ptr.hpp>
#include <cuda/stream>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <cstddef>

namespace gqe {
namespace storage {

/**
 * @brief Explicit-allocation byte buffer stored as an offset pointer.
 *
 * The buffer intentionally does not deallocate on destruction. Callers that own the backing memory
 * must call @ref deallocate with the same memory resource used for allocation.
 */
class buffer {
 public:
  buffer()  = default;
  ~buffer() = default;

  buffer(buffer const&)            = delete;
  buffer& operator=(buffer const&) = delete;

  buffer(buffer&& other) noexcept : _data(other._data), _size(other._size)
  {
    other._data = nullptr;
    other._size = 0;
  }

  buffer& operator=(buffer&& other) noexcept
  {
    if (this != &other) {
      assert(_data == nullptr && _size == 0);
      _data       = other._data;
      _size       = other._size;
      other._data = nullptr;
      other._size = 0;
    }
    return *this;
  }

  /**
   * @brief Allocate @p bytes from @p mr.
   */
  void allocate(std::size_t bytes, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Allocate storage from @p mr and enqueue a copy from @p source on @p stream.
   */
  explicit buffer(rmm::device_buffer const& source,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr);

  /**
   * @brief Deallocate this buffer from the same memory resource used to allocate it.
   */
  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  [[nodiscard]] void* data() noexcept { return _data.get(); }
  [[nodiscard]] void const* data() const noexcept { return _data.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }

 private:
  boost::interprocess::offset_ptr<std::byte> _data{nullptr};
  std::size_t _size{0};
};

}  // namespace storage
}  // namespace gqe
