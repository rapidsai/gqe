/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/cuda.hpp>

#include <gqe/device_properties.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/utilities/pinned_memory.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

struct extra_dummy_copy_config {
  size_t base_copied_bytes{};
  size_t target_dummy_bytes{};
  size_t full_dummy_copy_passes{};
  size_t partial_pass_full_buffer_count{};
  size_t partial_pass_last_buffer_bytes{};
};

// Compute how many extra dummy memcpy operations are needed to match the configured multiplier.
extra_dummy_copy_config compute_extra_dummy_copy_config(
  const rmm::device_uvector<size_t>& sizes_buffer, double total_copy_multiplier)
{
  extra_dummy_copy_config config{};
  if (total_copy_multiplier < 1.0) {
    throw std::invalid_argument("in_memory_dummy_copy_multiplier must be >= 1.0");
  }
  if (total_copy_multiplier == 1.0 || sizes_buffer.size() == 0) { return config; }

  config.base_copied_bytes = std::reduce(sizes_buffer.cbegin(), sizes_buffer.cend(), size_t{0});
  if (config.base_copied_bytes == 0) { return config; }

  config.target_dummy_bytes = static_cast<size_t>(
    std::ceil(static_cast<double>(config.base_copied_bytes) * (total_copy_multiplier - 1.0)));
  config.full_dummy_copy_passes    = config.target_dummy_bytes / config.base_copied_bytes;
  const auto remaining_dummy_bytes = config.target_dummy_bytes % config.base_copied_bytes;
  if (remaining_dummy_bytes == 0) { return config; }

  size_t remaining_bytes_in_partial_pass = remaining_dummy_bytes;
  for (const auto buffer_size : sizes_buffer) {
    if (remaining_bytes_in_partial_pass > buffer_size) {
      remaining_bytes_in_partial_pass -= buffer_size;
      ++config.partial_pass_full_buffer_count;
      continue;
    }
    if (remaining_bytes_in_partial_pass == buffer_size) {
      ++config.partial_pass_full_buffer_count;
    } else {
      config.partial_pass_last_buffer_bytes = remaining_bytes_in_partial_pass;
    }
    break;
  }
  return config;
}

void execute_extra_dummy_copy(void** dst_ptrs,
                              void** src_ptrs,
                              size_t* sizes,
                              size_t num_copied_buffers,
                              rmm::cuda_stream_view stream,
                              const extra_dummy_copy_config& config)
{
  for (size_t i = 0; i < config.full_dummy_copy_passes; ++i) {
    gqe::utility::do_batched_memcpy(dst_ptrs, src_ptrs, sizes, num_copied_buffers, stream);
  }
  if (config.partial_pass_full_buffer_count > 0) {
    gqe::utility::do_batched_memcpy(
      dst_ptrs, src_ptrs, sizes, config.partial_pass_full_buffer_count, stream);
  }
  if (config.partial_pass_last_buffer_bytes > 0) {
    const auto partial_last_idx = config.partial_pass_full_buffer_count;
    GQE_CUDA_TRY(cudaMemcpyAsync(dst_ptrs[partial_last_idx],
                                 src_ptrs[partial_last_idx],
                                 config.partial_pass_last_buffer_bytes,
                                 cudaMemcpyDefault,
                                 stream.value()));
  }
}

}  // namespace

namespace gqe {

namespace utility {

namespace detail {
int detect_launch_grid_size(void const* const kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes)
{
  auto device_id = current_cuda_device_id();
  auto num_sms =
    device_properties::instance().get<device_properties::multiProcessorCount>(device_id);

  int max_active_blocks = 0;
  GQE_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, kernel, block_size, dynamic_shared_memory_bytes));

  return max_active_blocks * num_sms;
}
}  // namespace detail

int get_device_count()
{
  int count{};
  GQE_CUDA_TRY(cudaGetDeviceCount(&count));
  return count;
}

rmm::cuda_device_id current_cuda_device_id()
{
  int id{};
  GQE_CUDA_TRY(cudaGetDevice(&id));
  return rmm::cuda_device_id{id};
}

void do_batched_memcpy(
  void** dst_ptrs, void** src_ptrs, size_t* sizes, size_t num_buffers, cudaStream_t stream)
{
  assert(num_buffers > 0 && "Must at least copy a single buffer");
  std::vector<cudaMemcpyAttributes> attrs(1);
  attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
  attrs[0].flags                = 0;
  std::vector<size_t> attrsIdxs = {0};
  size_t numAttrs               = attrs.size();
  size_t fail_idx;
#ifndef NDEBUG
  for (size_t i = 0; i < num_buffers; ++i) {
    GQE_LOG_DEBUG("i = {}, dst_ptrs[i] = {}, src_ptrs[i] = {}, sizes[i] = {}",
                  i,
                  (void*)dst_ptrs[i],
                  (void*)src_ptrs[i],
                  sizes[i]);
  }
#endif
  GQE_CUDA_TRY(cudaMemcpyBatchAsync((void**)dst_ptrs,
                                    (void**)src_ptrs,
                                    sizes,
                                    num_buffers,
                                    attrs.data(),
                                    attrsIdxs.data(),
                                    numAttrs,
                                    &fail_idx,
                                    stream));
}

void copy_batch::add(std::byte* dst_ptr, std::byte const* src_ptr, size_t size_in_bytes)
{
  if (size_in_bytes == 0) { return; }
  _requests.push_back({dst_ptr, src_ptr, size_in_bytes});
}

void copy_batch::reserve(size_t num_requests) { _requests.reserve(num_requests); }

bool copy_batch::empty() const { return _requests.empty(); }

size_t copy_batch::size() const { return _requests.size(); }

void copy_batch::execute(rmm::cuda_stream_view stream, double total_copy_multiplier) const
{
  if (empty()) { return; }

  nvtx_scoped_range nvtx_range("Filling cudaMemcpyBatchAsync arrays");

  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  rmm::device_uvector<size_t> sizes_buffer(size(), stream, cudf_pinned_resource);
  auto* sizes = sizes_buffer.data();
  rmm::device_uvector<std::byte*> dst_ptrs_buffer(size(), stream, cudf_pinned_resource);
  auto** dst_ptrs = dst_ptrs_buffer.data();
  rmm::device_uvector<std::byte const*> src_ptrs_buffer(size(), stream, cudf_pinned_resource);
  auto** src_ptrs = src_ptrs_buffer.data();
  stream.synchronize();

  GQE_LOG_DEBUG(
    "Created pointer arrays for batched memcpy; src_ptrs = {}, dst_ptrs = {}, sizes = {}",
    static_cast<void*>(src_ptrs),
    static_cast<void*>(dst_ptrs),
    static_cast<void*>(sizes));

  GQE_LOG_DEBUG("Filling pointer arrays for batched memcpy");
  for (size_t copy_idx = 0; copy_idx < _requests.size(); ++copy_idx) {
    const auto& request = _requests[copy_idx];
    dst_ptrs[copy_idx]  = request.dst_ptr;
    src_ptrs[copy_idx]  = request.src_ptr;
    sizes[copy_idx]     = request.size_in_bytes;
#ifndef NDEBUG
    GQE_LOG_DEBUG("copy_idx = {}, dst_ptrs[copy_idx] = {}, src_ptrs[copy_idx] = {}, size = {}",
                  copy_idx,
                  static_cast<void*>(dst_ptrs[copy_idx]),
                  static_cast<void const*>(src_ptrs[copy_idx]),
                  sizes[copy_idx]);
#endif
  }

  const auto dummy_copy_config =
    compute_extra_dummy_copy_config(sizes_buffer, total_copy_multiplier);
  if (dummy_copy_config.target_dummy_bytes > 0) {
    GQE_LOG_DEBUG(
      "Extra dummy memcpy enabled; multiplier = {}, base_copied_bytes = {}, dummy_bytes = {}",
      total_copy_multiplier,
      dummy_copy_config.base_copied_bytes,
      dummy_copy_config.target_dummy_bytes);
  }

  auto** dst_void_ptrs = reinterpret_cast<void**>(dst_ptrs);
  auto** src_void_ptrs = reinterpret_cast<void**>(const_cast<std::byte**>(src_ptrs));

  GQE_LOG_DEBUG("Batched memcpy");
  do_batched_memcpy(dst_void_ptrs, src_void_ptrs, sizes, size(), stream);
  if (total_copy_multiplier > 1.0) {
    execute_extra_dummy_copy(
      dst_void_ptrs, src_void_ptrs, sizes, size(), stream, dummy_copy_config);
  }
  // Keep this sync in this function because cudaMemcpyBatchAsync consumes the pointer/size arrays
  // asynchronously, and those arrays are backed by local buffers destroyed when we return.
  stream.synchronize();
}

}  // namespace utility

}  // namespace gqe
