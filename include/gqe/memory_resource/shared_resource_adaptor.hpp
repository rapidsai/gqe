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

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <rmm/aligned.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief Type-erasure-ready wrapper that holds a non-copyable memory resource via
 * `std::shared_ptr`.
 *
 * `cuda::mr::any_resource<Properties...>` requires the wrapped resource to be **copyable**.
 * Several gqe memory resources own state that is fundamentally non-copyable -- a `cudaMemPool_t`,
 * an NVSHMEM symmetric allocation, a `boost::interprocess::managed_shared_memory` segment
 * registered with CUDA, etc. This adaptor wraps such a resource in a `std::shared_ptr` (which
 * **is** copyable) and forwards the `cuda::mr::async_resource` concept's surface to the
 * underlying object so that `cuda::mr::any_resource{adaptor}` works.
 *
 * Example:
 * @code
 * auto pgas = std::make_shared<gqe::pgas_memory_resource>(num_bytes);
 * cuda::mr::any_resource<cuda::mr::device_accessible> mr{
 *   gqe::memory_resource::shared_resource_adaptor{pgas}};
 * @endcode
 *
 * The adaptor does not need to enumerate which CCCL access properties (`device_accessible`,
 * `host_accessible`, etc.) the wrapped resource exposes -- the `get_property` overload below is
 * constrained on `Resource` itself supporting that property, so the adaptor advertises **exactly
 * the same set** of properties as the wrapped resource.
 *
 * @tparam Resource A type modelling `cuda::mr::async_resource` (i.e. one of gqe's ported MR
 * classes, or any RMM/cudf resource).
 */
template <typename Resource>
class shared_resource_adaptor {
 public:
  /**
   * @brief Construct the adaptor from a `shared_ptr` to the underlying resource.
   *
   * The pointer must be non-null.
   */
  explicit shared_resource_adaptor(std::shared_ptr<Resource> resource)
  {
    if (!resource) {
      throw std::invalid_argument(
        "shared_resource_adaptor requires a non-null std::shared_ptr<Resource>.");
    }
    _resource = std::move(resource);
  }

  /**
   * @brief Convenience constructor that takes ownership of an rvalue resource.
   *
   * Equivalent to wrapping with `std::make_shared<Resource>(std::move(resource))`.
   */
  explicit shared_resource_adaptor(Resource&& resource)
    : _resource{std::make_shared<Resource>(std::move(resource))}
  {
  }

  /**
   * @brief Access the wrapped resource.
   */
  [[nodiscard]] std::shared_ptr<Resource> const& resource() const noexcept { return _resource; }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return _resource->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    _resource->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return _resource->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    _resource->deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief Forward accessibility tags from the wrapped resource.
   *
   * If `Resource` supports a tag property such as `host_accessible` or
   * `device_accessible`, this adaptor advertises the same tag.
   *
   * This overload intentionally applies only to tag-only properties
   * (properties without a `value_type`). Value-bearing properties are left to
   * CCCL's default `get_property` logic.
   */
  template <typename Property>
    requires(requires(Resource const& r, Property p) { get_property(r, p); } &&
             !cuda::property_with_value<Property>)
  friend void get_property(shared_resource_adaptor const&, Property) noexcept
  {
  }

  /**
   * @brief Two adaptors compare equal iff they share ownership of the same underlying resource.
   */
  [[nodiscard]] bool operator==(shared_resource_adaptor const& other) const noexcept
  {
    return _resource == other._resource;
  }
  [[nodiscard]] bool operator!=(shared_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  std::shared_ptr<Resource> _resource;
};

/**
 * @brief Class template argument deduction guide for `shared_resource_adaptor`.
 */
template <typename Resource>
shared_resource_adaptor(std::shared_ptr<Resource>) -> shared_resource_adaptor<Resource>;

}  // namespace memory_resource

}  // namespace gqe
