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

#include <gqe/memory_resource/numa_pool_memory_resource.hpp>

#include <gqe/device_properties.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_async_view_memory_resource.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>

namespace {

// Ref:
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#query-for-support
constexpr int k_min_driver_version_for_hw_decompress_support = 12080;

void validate_pool_capabilities(rmm::cuda_device_id device)
{
  auto const& device_props = gqe::device_properties::instance();

  GQE_EXPECTS(device_props.get<gqe::device_properties::property::memoryPoolsSupported>(device),
              "CUDA memory pools are not supported on the current device");

  auto pool_supported_handle_types =
    device_props.get<gqe::device_properties::property::memoryPoolSupportedHandleTypes>(device);
  GQE_EXPECTS((pool_supported_handle_types & cudaMemHandleTypeFabric) != 0,
              "CUDA memory pools do not support fabric handles on the current device");
}

cudaMemPool_t create_numa_pool(int numa_node, std::size_t initial_size, std::size_t max_size)
{
  GQE_EXPECTS(numa_node >= 0, "NUMA node must be non-negative");

  auto device_id = rmm::get_current_cuda_device();
  validate_pool_capabilities(device_id);

  int driver_version = 0;
  GQE_CUDA_TRY(cudaDriverGetVersion(&driver_version));

  cudaMemPoolProps pool_props{};
  pool_props.allocType = cudaMemAllocationTypePinned;
  // TODO: Add support for POSIX handle types on platforms which dont support Fabric handles.
  pool_props.handleTypes   = cudaMemHandleTypeFabric;
  pool_props.location.type = cudaMemLocationTypeHostNuma;
  pool_props.location.id   = numa_node;
  pool_props.maxSize       = max_size;  // Setting to 0 uses a system dependent value
  auto const& device_props = gqe::device_properties::instance();
  bool mem_decompress_support =
    device_props.get<gqe::device_properties::property::memDecompressSupport>(device_id);

  // TODO: This check should be done in device_properties class.
  bool hw_decompress_supported =
    driver_version >= k_min_driver_version_for_hw_decompress_support && mem_decompress_support;
  if (hw_decompress_supported) {
    pool_props.usage = cudaMemPoolCreateUsageHwDecompress;
  } else {
    GQE_LOG_WARN(
      "Hardware decompression not supported (driver={}, hw_support={}); "
      "continuing without it.",
      driver_version,
      mem_decompress_support);
  }

  cudaMemPool_t pool{nullptr};
  GQE_CUDA_TRY(cudaMemPoolCreate(&pool, &pool_props));

  // Allow device kernels to access memory
  cudaMemAccessDesc access_desc{};
  access_desc.location.type = cudaMemLocationTypeDevice;
  access_desc.location.id   = device_id.value();
  access_desc.flags         = cudaMemAccessFlagsProtReadWrite;
  GQE_CUDA_TRY(cudaMemPoolSetAccess(pool, &access_desc, 1));

  // Set release threshold to max pool size so we dont eagerly release memory back to the OS.
  cuuint64_t release_threshold = static_cast<cuuint64_t>(max_size);
  GQE_CUDA_TRY(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));

  // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
  // specified size (only if initial_pool_size is provided)
  if (initial_size > 0) {
    auto stream = cudf::get_default_stream();
    void* ptr   = nullptr;
    GQE_CUDA_TRY(cudaMallocFromPoolAsync(&ptr, initial_size, pool, stream.value()));
    GQE_CUDA_TRY(cudaFreeAsync(ptr, stream.value()));
    GQE_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    GQE_LOG_INFO("Pre-allocated {} bytes in NUMA pool", initial_size);
  }

  return pool;
}

}  // namespace

namespace gqe {

namespace memory_resource {

struct remote_numa_pool_handle_impl {
  explicit remote_numa_pool_handle_impl(CUmemFabricHandle handle) : _handle(handle) {}

  [[nodiscard]] CUmemFabricHandle handle() const noexcept { return _handle; }

 private:
  CUmemFabricHandle _handle{};
};

struct remote_pool_pointer_impl {
  explicit remote_pool_pointer_impl(cudaMemPoolPtrExportData handle) : _handle(handle) {}

  [[nodiscard]] cudaMemPoolPtrExportData handle() const noexcept { return _handle; }

 private:
  cudaMemPoolPtrExportData _handle{};
};

remote_numa_pool_handle::remote_numa_pool_handle(std::unique_ptr<remote_numa_pool_handle_impl> impl)
  : _impl(std::move(impl))
{
}

remote_numa_pool_handle::~remote_numa_pool_handle() = default;

remote_numa_pool_handle::remote_numa_pool_handle(remote_numa_pool_handle&&) noexcept = default;

remote_numa_pool_handle& remote_numa_pool_handle::operator=(remote_numa_pool_handle&&) noexcept =
  default;

remote_pool_pointer::remote_pool_pointer(std::unique_ptr<remote_pool_pointer_impl> impl)
  : _impl(std::move(impl))
{
}

remote_pool_pointer::~remote_pool_pointer() = default;

remote_pool_pointer::remote_pool_pointer(remote_pool_pointer&&) noexcept = default;

remote_pool_pointer& remote_pool_pointer::operator=(remote_pool_pointer&&) noexcept = default;

/**
 * @brief Implementation class for numa_pool_handle.
 */
struct numa_pool_handle_impl {
  explicit numa_pool_handle_impl(int numa_node, std::size_t initial_size, std::size_t max_size)
    : _pool(create_numa_pool(numa_node, initial_size, max_size)),
      _mr(std::make_unique<rmm::mr::cuda_async_view_memory_resource>(_pool))
  {
  }

  ~numa_pool_handle_impl()
  {
    _mr.reset();
    if (_pool != nullptr) {
      cuuint64_t used_mem = 0;
      cudaError_t err = cudaMemPoolGetAttribute(_pool, cudaMemPoolAttrUsedMemCurrent, &used_mem);
      if (err == cudaSuccess && used_mem > 0) {
        GQE_LOG_WARN("Memory leak detected: {} bytes still allocated from NUMA pool at destruction",
                     used_mem);
      }

      // We call terminate() here since exceptions cannot be thrown in a destructor.
      if (auto status = cudaMemPoolDestroy(_pool); status != cudaSuccess) {
        GQE_LOG_ERROR("cudaMemPoolDestroy failed with: {}", cudaGetErrorString(status));
        std::terminate();
      }
    }
  }

  numa_pool_handle_impl(numa_pool_handle_impl const&)            = delete;
  numa_pool_handle_impl& operator=(numa_pool_handle_impl const&) = delete;
  numa_pool_handle_impl(numa_pool_handle_impl&&)                 = delete;
  numa_pool_handle_impl& operator=(numa_pool_handle_impl&&)      = delete;

  [[nodiscard]] cudaMemPool_t pool() const noexcept { return _pool; }

  [[nodiscard]] rmm::mr::cuda_async_view_memory_resource& mr() { return *_mr; }

 private:
  cudaMemPool_t _pool{nullptr};

  std::unique_ptr<rmm::mr::cuda_async_view_memory_resource> _mr;
};

/**
 * @brief Implementation class for imported_numa_pool_handle.
 */
struct imported_numa_pool_handle_impl {
  explicit imported_numa_pool_handle_impl(cudaMemPool_t imported_pool) : _pool(imported_pool) {}

  ~imported_numa_pool_handle_impl()
  {
    if (_pool != nullptr) {
      // We call terminate() here since exceptions cannot be thrown in a destructor.
      if (auto status = cudaMemPoolDestroy(_pool); status != cudaSuccess) {
        GQE_LOG_ERROR("cudaMemPoolDestroy failed with: {}", cudaGetErrorString(status));
        std::terminate();
      }
    }
  }

  imported_numa_pool_handle_impl(imported_numa_pool_handle_impl const&)            = delete;
  imported_numa_pool_handle_impl& operator=(imported_numa_pool_handle_impl const&) = delete;
  imported_numa_pool_handle_impl(imported_numa_pool_handle_impl&&)                 = delete;
  imported_numa_pool_handle_impl& operator=(imported_numa_pool_handle_impl&&)      = delete;

  [[nodiscard]] cudaMemPool_t pool() const noexcept { return _pool; }

 private:
  cudaMemPool_t _pool{nullptr};
};

// numa_pool_handle implementation
numa_pool_handle::numa_pool_handle() = default;

numa_pool_handle::numa_pool_handle(std::unique_ptr<numa_pool_handle_impl> impl)
  : _impl(std::move(impl))
{
}

numa_pool_handle::~numa_pool_handle() = default;

numa_pool_handle::numa_pool_handle(numa_pool_handle&&) noexcept = default;

numa_pool_handle& numa_pool_handle::operator=(numa_pool_handle&&) noexcept = default;

remote_numa_pool_handle numa_pool_handle::export_pool() const
{
  GQE_EXPECTS(_impl != nullptr, "Attempted to export an invalid numa_pool_handle");

  CUmemFabricHandle remote_handle{};
  GQE_CUDA_TRY(
    cudaMemPoolExportToShareableHandle(&remote_handle, _impl->pool(), cudaMemHandleTypeFabric, 0));
  return remote_numa_pool_handle(std::make_unique<remote_numa_pool_handle_impl>(remote_handle));
}

remote_pool_pointer numa_pool_handle::export_pointer(void* ptr) const
{
  GQE_EXPECTS(_impl != nullptr, "Attempted to export pointer from an invalid numa_pool_handle");
  GQE_EXPECTS(ptr != nullptr, "Cannot export a nullptr");

  cudaMemPoolPtrExportData export_data{};
  GQE_CUDA_TRY(cudaMemPoolExportPointer(&export_data, ptr));
  return remote_pool_pointer(std::make_unique<remote_pool_pointer_impl>(export_data));
}

// imported_numa_pool_handle implementation
imported_numa_pool_handle::imported_numa_pool_handle(remote_numa_pool_handle const& remote_handle)
{
  GQE_EXPECTS(remote_handle._impl != nullptr,
              "Attempted to import pool from an invalid remote_numa_pool_handle");

  cudaMemPool_t imported_pool{nullptr};
  auto mutable_handle = remote_handle._impl->handle();
  GQE_CUDA_TRY(cudaMemPoolImportFromShareableHandle(
    &imported_pool, &mutable_handle, cudaMemHandleTypeFabric, 0));
  _impl = std::make_unique<imported_numa_pool_handle_impl>(imported_pool);
}

imported_numa_pool_handle::~imported_numa_pool_handle() = default;

imported_numa_pool_handle::imported_numa_pool_handle(imported_numa_pool_handle&&) noexcept =
  default;

imported_numa_pool_handle& imported_numa_pool_handle::operator=(
  imported_numa_pool_handle&&) noexcept = default;

void* imported_numa_pool_handle::import_pointer(remote_pool_pointer const& remote_pointer) const
{
  GQE_EXPECTS(_impl != nullptr,
              "Attempted to import pointer with an invalid imported_numa_pool_handle");
  GQE_EXPECTS(remote_pointer._impl != nullptr,
              "Attempted to import pointer with invalid remote_pool_pointer");

  auto mutable_export_data = remote_pointer._impl->handle();
  void* imported_ptr       = nullptr;
  GQE_CUDA_TRY(cudaMemPoolImportPointer(&imported_ptr, _impl->pool(), &mutable_export_data));
  return imported_ptr;
}

// numa_pool_memory_resource implementation
numa_pool_memory_resource::numa_pool_memory_resource(int numa_node,
                                                     std::size_t initial_size,
                                                     std::optional<std::size_t> max_size)
{
  _pool_handle._impl =
    std::make_unique<numa_pool_handle_impl>(numa_node, initial_size, max_size.value_or(0));
}

numa_pool_memory_resource::~numa_pool_memory_resource() = default;

numa_pool_handle const& numa_pool_memory_resource::pool_handle() const noexcept
{
  return _pool_handle;
}

void* numa_pool_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
{
  if (bytes == 0) { return nullptr; }
  return _pool_handle._impl->mr().allocate(bytes, stream);
}

void numa_pool_memory_resource::do_deallocate(void* ptr,
                                              std::size_t bytes,
                                              rmm::cuda_stream_view stream)
{
  if (ptr == nullptr) { return; }
  _pool_handle._impl->mr().deallocate(ptr, bytes, stream);
}

}  // namespace memory_resource

}  // namespace gqe
