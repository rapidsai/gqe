/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/memory_resource/shared_numa_pool_memory_resource.hpp>

#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/utilities/default_stream.hpp>

namespace gqe {

namespace memory_resource {

shared_numa_pool_resource::shared_numa_pool_resource(int numa_node,
                                                     std::size_t initial_size,
                                                     std::optional<std::size_t> max_size)
  : _state(export_state::unexported),
    _owner_pool(
      std::make_unique<numa_pool_memory_resource>(numa_node, initial_size, max_size.value_or(0)))
{
}

shared_numa_pool_resource::export_state shared_numa_pool_resource::state() const
{
  std::lock_guard<std::mutex> latch_guard(_state_latch);
  return _state;
}

remote_numa_pool_handle shared_numa_pool_resource::export_pool()
{
  std::lock_guard<std::mutex> latch_guard(_state_latch);

  GQE_EXPECTS(_owner_pool != nullptr,
              "Invalid shared_numa_pool_resource state: missing owner pool");

  auto handle = _owner_pool->pool_handle().export_pool();
  _state      = export_state::exported;
  return handle;
}

remote_pool_pointer shared_numa_pool_resource::export_pointer(void* ptr) const
{
  GQE_EXPECTS(ptr != nullptr, "Cannot export a nullptr");

  std::lock_guard<std::mutex> latch_guard(_state_latch);
  GQE_EXPECTS(_owner_pool != nullptr,
              "Invalid shared_numa_pool_resource state: missing owner pool");
  return _owner_pool->pool_handle().export_pointer(ptr);
}

void shared_numa_pool_resource::finalize()
{
  std::lock_guard<std::mutex> latch_guard(_state_latch);
  GQE_EXPECTS(_owner_pool != nullptr,
              "Invalid shared_numa_pool_resource state: missing owner pool");

  // Ensure all prior GPU work is complete before enqueuing deferred frees.
  GQE_CUDA_TRY(cudaDeviceSynchronize());
  auto const default_stream = cudf::get_default_stream();

  for (auto it = _deferred_deallocations.begin(); it != _deferred_deallocations.end();) {
    _owner_pool->deallocate(it->first, it->second, default_stream);
    it = _deferred_deallocations.erase(it);
  }

  // Ensure deferred frees complete before finalize() returns.
  GQE_CUDA_TRY(cudaStreamSynchronize(default_stream.value()));
}

imported_numa_pool_resource shared_numa_pool_resource::import_pool(
  remote_numa_pool_handle const& remote_handle)
{
  return imported_numa_pool_resource(remote_handle);
}

void* shared_numa_pool_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
{
  if (bytes == 0) { return nullptr; }

  std::lock_guard<std::mutex> latch_guard(_state_latch);
  GQE_EXPECTS(_owner_pool != nullptr,
              "Invalid shared_numa_pool_resource state: missing owner pool");

  return _owner_pool->allocate(bytes, stream);
}

void shared_numa_pool_resource::do_deallocate(void* ptr,
                                              std::size_t bytes,
                                              rmm::cuda_stream_view stream)
{
  if (ptr == nullptr) { return; }

  std::lock_guard<std::mutex> latch_guard(_state_latch);
  GQE_EXPECTS(_owner_pool != nullptr,
              "Invalid shared_numa_pool_resource state: missing owner pool");

  if (_state == export_state::exported) {
    auto [_, inserted] = _deferred_deallocations.emplace(ptr, bytes);
    GQE_EXPECTS(inserted, "Pointer is already scheduled for deferred deallocation");
    return;
  }

  _owner_pool->deallocate(ptr, bytes, stream);
}

imported_numa_pool_resource::imported_numa_pool_resource(
  remote_numa_pool_handle const& remote_handle)
  : _imported_pool(remote_handle)
{
}

owning_pointer imported_numa_pool_resource::import_pointer(
  remote_pool_pointer const& remote_pointer, rmm::cuda_stream_view stream)
{
  auto imported_ptr = _imported_pool.import_pointer(remote_pointer);
  return owning_pointer(imported_ptr, [stream](void* ptr) noexcept {
    // Release on the stream selected when importing this pointer.
    if (ptr != nullptr) {
      auto const status = cudaFreeAsync(ptr, stream.value());
      if (status != cudaSuccess) {
        GQE_LOG_ERROR("Failed to release imported pointer {} with cudaFreeAsync: {} {}",
                      ptr,
                      cudaGetErrorName(status),
                      cudaGetErrorString(status));
      }
    }
  });
}

}  // namespace memory_resource

}  // namespace gqe
