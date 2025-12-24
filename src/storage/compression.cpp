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

#include <cstdio>
#include <cuda.h>
#include <cudf/utilities/pinned_memory.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <mutex>

// Helper function to convert compression format enum to string for logging
std::string compression_format_to_string(gqe::compression_format comp_format)
{
  switch (comp_format) {
    case gqe::compression_format::none: return "NONE";
    case gqe::compression_format::lz4: return "LZ4";
    case gqe::compression_format::snappy: return "SNAPPY";
    case gqe::compression_format::ans: return "ANS";
    case gqe::compression_format::cascaded: return "CASCADED";
    case gqe::compression_format::gdeflate: return "GDEFLATE";
    case gqe::compression_format::deflate: return "DEFLATE";
    case gqe::compression_format::zstd: return "ZSTD";
    case gqe::compression_format::bitcomp: return "BITCOMP";
    case gqe::compression_format::best_compression_ratio: return "BEST_COMPRESSION_RATIO";
    case gqe::compression_format::best_decompression_speed: return "BEST_DECOMPRESSION_SPEED";
    default: return "UNKNOWN";
  }
}

// Helper function to check if a compression algorithm is supported on the CPU
bool is_algorthm_cpu_supported(gqe::compression_format comp_format)
{
  return comp_format == gqe::compression_format::lz4;
}

compression_manager::compression_manager(gqe::compression_format comp_format,
                                         gqe::compression_format secondary_compression_format,
                                         int compression_chunk_size,
                                         rmm::cuda_stream_view supplied_stream,
                                         rmm::device_async_resource_ref mr,
                                         double compression_ratio_threshold,
                                         double secondary_compression_ratio_threshold,
                                         double secondary_compression_multiplier_threshold,
                                         bool use_cpu_compression,
                                         int compression_level,
                                         std::string column_name,
                                         cudf::data_type cudf_type)
  : _comp_format(comp_format),
    _secondary_comp_format(secondary_compression_format),
    _compression_chunk_size(compression_chunk_size),
    _column_name(column_name),
    _cudf_type(cudf_type),
    _compression_ratio_threshold(compression_ratio_threshold),
    _secondary_compression_ratio_threshold(secondary_compression_ratio_threshold),
    _secondary_compression_multiplier_threshold(secondary_compression_multiplier_threshold),
    _use_cpu_compression(use_cpu_compression),
    _compression_level(compression_level)
{
  // Try a test allocation to see if we can access on the host

  int device_id = 0;
  GQE_CUDA_TRY(cudaGetDevice(&device_id));

  int decompressSupportMask = 0;
  CUresult result           = cuDeviceGetAttribute(
    &decompressSupportMask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device_id);
  if (result != CUDA_SUCCESS) {
    const char* error_string = nullptr;
    cuGetErrorString(result, &error_string);
    GQE_LOG_ERROR("Failed to get device attribute MEM_DECOMPRESS_ALGORITHM_MASK for device {}: {}",
                  device_id,
                  error_string);
    throw std::runtime_error("Failed to get device attribute MEM_DECOMPRESS_ALGORITHM_MASK");
  }
  const bool device_supports_decomp = static_cast<bool>(decompressSupportMask);

  _decompress_backend =
    device_supports_decomp ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE : NVCOMP_DECOMPRESS_BACKEND_DEFAULT;

  GQE_LOG_TRACE(
    "Created compression manager for column '{}': format={}, chunk_size={}, "
    "cudf_type={}, use_cpu_compression={}, compression_level={}",
    _column_name,
    compression_format_to_string(_comp_format),
    _compression_chunk_size,
    _cudf_type,
    _use_cpu_compression,
    _compression_level);
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_compress(
  rmm::device_buffer const* uncompressed,
  bool& is_compressed,
  int64_t& compressed_size,
  int64_t& uncompressed_size,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  is_compressed = true;

  nvcompType_t data_type = get_optimal_nvcomp_data_type(_cudf_type.id());
  GQE_LOG_TRACE(
    "Starting compression for column '{}': input_size={}, compression_algorithm={}, "
    "chunk_size={}, data_type={}, cudf_type={}",
    _column_name,
    uncompressed->size(),
    compression_format_to_string(_comp_format),
    _compression_chunk_size,
    static_cast<int>(data_type),
    _cudf_type);

  auto compression_manager = create_manager(_comp_format, data_type, supplied_stream, mr);
  auto comp_config         = compression_manager->configure_compression(uncompressed->size());
  auto compressed_buffer   = std::make_unique<rmm::device_buffer>(
    comp_config.max_compressed_buffer_size, supplied_stream, mr);

  compression_manager->compress(static_cast<uint8_t const*>(uncompressed->data()),
                                static_cast<uint8_t*>(compressed_buffer->data()),
                                comp_config);

  auto const comp_size = compression_manager->get_compressed_output_size(
    static_cast<uint8_t*>(compressed_buffer->data()));

  auto const compression_ratio = static_cast<double>(uncompressed->size()) / comp_size;

  compressed_buffer->resize(comp_size, supplied_stream);
  compressed_buffer->shrink_to_fit(supplied_stream);

  if (compression_ratio < _compression_ratio_threshold) {
    is_compressed = false;
    GQE_LOG_TRACE(
      "Compression ineffective for column '{}' using compression algorithm {}: compressed_size={} "
      "> uncompressed_size={}, compression_ratio={:.2f}",
      _column_name,
      compression_format_to_string(_comp_format),
      comp_size,
      uncompressed->size(),
      compression_ratio);
    return std::make_unique<rmm::device_buffer>(*uncompressed, supplied_stream, mr);
  }

  compressed_size   = comp_size;
  uncompressed_size = uncompressed->size();

  GQE_LOG_TRACE(
    "Compression successful for column '{}' using compression algorithm {}: uncompressed_size={}, "
    "compressed_size={}, compression_ratio={:.2f}",
    _column_name,
    compression_format_to_string(_comp_format),
    uncompressed->size(),
    comp_size,
    compression_ratio);

  return compressed_buffer;
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_decompress(
  cudf::device_span<uint8_t const> compressed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  nvcompType_t data_type = get_optimal_nvcomp_data_type(_cudf_type.id());
  GQE_LOG_TRACE(
    "Starting decompression for column '{}': compressed_size={}, compression_algorithm={}, "
    "data_type={}, chunk_size={}, cudf_type={}",
    _column_name,
    compressed.size(),
    compression_format_to_string(_comp_format),
    static_cast<int>(data_type),
    _compression_chunk_size,
    _cudf_type);

  auto decompression_manager = create_manager(_comp_format, data_type, stream, mr);

  DecompressionConfig decomp_config =
    decompression_manager->configure_decompression(compressed.data());

  std::unique_ptr<rmm::device_buffer> decompressed_data =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, stream, mr);

  decompression_manager->decompress(
    (uint8_t*)decompressed_data->data(), compressed.data(), decomp_config);

  GQE_LOG_TRACE(
    "Decompression completed for column '{}' using compression algorithm {}: compressed_size={}, "
    "decompressed_size={}",
    _column_name,
    compression_format_to_string(_comp_format),
    compressed.size(),
    decompressed_data->size());

  return decompressed_data;
}

void compression_manager::decompress_batch(uint8_t* const device_decompressed_primary_ptr,
                                           const uint8_t* const* device_compressed,
                                           const uint8_t* const* host_compressed,
                                           const size_t batch_count,
                                           const bool is_secondary_compressed,
                                           cudf::data_type cudf_type,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
{
  // variable not needed for decomp

  constexpr const size_t* compression_sizes = nullptr;
  compression_format comp_format = is_secondary_compressed ? _secondary_comp_format : _comp_format;
  nvcompType_t data_type         = get_optimal_nvcomp_data_type(cudf_type.id());
  auto decompression_manager     = create_manager(comp_format, data_type, stream, mr);
  auto cudf_pinned_resource      = cudf::get_pinned_memory_resource();
  rmm::device_buffer device_decompressed_buffer(
    sizeof(uint8_t*) * batch_count, stream, cudf_pinned_resource);
  uint8_t** device_decompressed = reinterpret_cast<uint8_t**>(device_decompressed_buffer.data());
  GQE_CUDA_TRY(cudaStreamSynchronize(stream));

  // This function assumes all pointers were allocated the same way and share the attributes of the
  // first buffer. Otherwise, behavior is undefined.
  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, host_compressed[0]);

  // for host api we still pass in an empty vector, otherwise populate
  std::vector<nvcomp::DecompressionConfig> decompression_configs;
  // if call succeeded and we have non-nullptr on hostPointer, this should be host-accessible
  if (status == cudaSuccess && attrs.hostPointer) {
    GQE_LOG_TRACE("decompress_batch: Using no-kernel batched decompression api");
    size_t decompressed_offset = 0;
    for (size_t i = 0; i < batch_count; ++i) {
      device_decompressed[i] = device_decompressed_primary_ptr + decompressed_offset;
      decompressed_offset +=
        decompression_manager->get_decompressed_output_size_host(host_compressed[i]);
    }
    // uses zero-copy no-kernel decompression api
    decompression_manager->decompress(device_decompressed,
                                      device_compressed,
                                      decompression_configs,
                                      batch_count,
                                      compression_sizes,
                                      host_compressed);
  } else {
    // fall back to decompression api that uses kernels
    GQE_LOG_TRACE(
      "decompress_batch: Using fallback kernel-based batched decompression api due to host "
      "compression buffer accessibility");
    size_t decompressed_offset = 0;
    for (size_t i = 0; i < batch_count; ++i) {
      decompression_configs.push_back(
        decompression_manager->configure_decompression(device_compressed[i]));
      device_decompressed[i] = device_decompressed_primary_ptr + decompressed_offset;
      decompressed_offset += decompression_configs[i].decomp_data_size;
    }
    decompression_manager->decompress(device_decompressed,
                                      device_compressed,
                                      decompression_configs,
                                      batch_count,
                                      compression_sizes);
  }
}

std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::try_compression_algorithm(
  const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
  const size_t total_uncompressed_size,
  uint8_t** uncompressed_ptrs,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  cudf::data_type cudf_type,
  size_t num_buffers,
  memory_kind::type memory_kind,
  uint8_t** compressed_ptrs,
  double& compression_ratio,
  size_t& total_compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  bool compression_mr_is_host_accessible = memory_kind::is_cpu_accessible(memory_kind);

  compressed_sizes.reserve(num_buffers);

  if ((_use_cpu_compression) && (is_algorthm_cpu_supported(comp_format))) {
    return try_cpu_compression(device_uncompressed,
                               total_uncompressed_size,
                               uncompressed_ptrs,
                               comp_format,
                               data_type,
                               cudf_type,
                               num_buffers,
                               memory_kind,
                               compressed_ptrs,
                               compression_ratio,
                               total_compressed_size,
                               compressed_sizes,
                               stream,
                               mr);
  }

  std::vector<nvcomp::CompressionConfig> compression_configs;
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;
  std::unique_ptr<nvcomp::nvcompManagerBase> compression_manager =
    create_manager(comp_format, data_type, stream, mr);
  compression_configs.reserve(num_buffers);

  for (size_t ix = 0; ix < num_buffers; ix++) {
    auto& uncompressed = device_uncompressed[ix];
    auto config        = compression_manager->configure_compression(uncompressed->size());
    compression_configs.push_back(config);
    compressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(config.max_compressed_buffer_size, stream, mr));
    compressed_ptrs[ix] = static_cast<uint8_t*>(compressed_data_buffers.back()->data());
  }
  compression_manager->compress(uncompressed_ptrs, compressed_ptrs, compression_configs);
  if (compression_mr_is_host_accessible) {
    // Only necessary if we're going to grab the compressed sizes from the host
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  total_compressed_size = 0;
  for (size_t ix = 0; ix < num_buffers; ix++) {
    size_t comp_size = 0;
    if (compression_mr_is_host_accessible) {
      comp_size = compression_manager->get_compressed_output_size_host(compressed_ptrs[ix]);

    } else {
      comp_size = compression_manager->get_compressed_output_size(compressed_ptrs[ix]);
    }

    assert(comp_size <= compression_configs[ix].max_compressed_buffer_size);

    compressed_sizes.push_back(comp_size);
    total_compressed_size += comp_size;
  }
  compression_ratio = static_cast<double>(total_uncompressed_size) / total_compressed_size;
  return compressed_data_buffers;
}

std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::try_cpu_compression(
  const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
  const size_t total_uncompressed_size,
  uint8_t** uncompressed_ptrs,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  cudf::data_type cudf_type,
  size_t num_buffers,
  memory_kind::type memory_kind,
  uint8_t** compressed_ptrs,
  double& compression_ratio,
  size_t& compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  bool compression_mr_is_host_accessible = memory_kind::is_cpu_accessible(memory_kind);

  std::unique_ptr<CPUHLIFManager> cpu_compression_manager;
  std::vector<std::vector<uint8_t>> cpu_comp_buffers;

  cpu_compression_manager = create_cpu_manager(_compression_level);

  std::vector<size_t> max_comp_sizes;
  std::vector<std::vector<uint8_t>> cpu_uncomp_buffers;
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;

  for (size_t ix = 0; ix < device_uncompressed.size(); ix++) {
    auto& uncompressed = device_uncompressed[ix];
    size_t max_compressed_size =
      cpu_compression_manager->get_max_compressed_output_size(uncompressed->size());
    max_comp_sizes.push_back(max_compressed_size);
    cpu_comp_buffers.push_back(std::vector<uint8_t>(max_compressed_size));

    compressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(max_compressed_size, stream, mr));
    compressed_ptrs[ix] = static_cast<uint8_t*>(compressed_data_buffers.back()->data());

    // Check if uncompressed data is host-accessible
    if (compression_mr_is_host_accessible) {
      const uint8_t* host_ptr = static_cast<const uint8_t*>(uncompressed->data());
      cpu_uncomp_buffers.push_back(std::vector<uint8_t>(host_ptr, host_ptr + uncompressed->size()));
      // cpu_batch_compress expects const std::vector<std::vector<uint8_t>>&, so we cannot
      // directly use the host pointer
      GQE_LOG_TRACE("compress_batch: Buffer {} is host-accessible, avoiding cudaMemcpy D2H", ix);
    } else {
      GQE_LOG_TRACE("compress_batch: Buffer {} is device-only memory, copying via cudaMemcpy D2H",
                    ix);
      // Data is device-only memory, need to copy via cudaMemcpy
      utility::nvtx_scoped_range nvtx_cpu_compress_dtoh("CPU_Compress_DtoH");
      cpu_uncomp_buffers.push_back(std::vector<uint8_t>(uncompressed->size()));
      GQE_CUDA_TRY(cudaMemcpyAsync(cpu_uncomp_buffers.back().data(),
                                   uncompressed->data(),
                                   uncompressed->size(),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    }
  }

  if (compression_mr_is_host_accessible) {
    // Synchronize the stream to ensure the data is copied before we launch CPU compression (which
    // does not use CUDA stream).
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  size_t total_compressed_size = 0;
  // Use lock_guard in a scope to serialize CPU batch compression
  {
    static std::mutex cpu_batch_compress_mutex;
    std::lock_guard<std::mutex> cpu_lock(cpu_batch_compress_mutex);

    utility::nvtx_scoped_range nvtx_cpu_compress("CPU_Compress");
    const int num_threads = std::thread::hardware_concurrency();
    GQE_LOG_TRACE(
      "Using CPU compression manager for batched compression of column '{}' with compression "
      "level {}",
      _column_name,
      _compression_level);

    cpu_compression_manager->cpu_batch_compress(
      cpu_comp_buffers, cpu_uncomp_buffers, num_threads, max_comp_sizes);
  }

  for (size_t ix = 0; ix < cpu_comp_buffers.size(); ix++) {
    size_t batch_comp_size =
      cpu_compression_manager->get_compressed_output_size(cpu_comp_buffers[ix].data());
    assert(batch_comp_size <= max_comp_sizes[ix]);
    compressed_sizes.push_back(batch_comp_size);
    total_compressed_size += batch_comp_size;

    GQE_CUDA_TRY(cudaMemcpyAsync(compressed_ptrs[ix],
                                 cpu_comp_buffers[ix].data(),
                                 batch_comp_size,
                                 cudaMemcpyHostToDevice,
                                 stream));
  }

  compressed_size   = total_compressed_size;
  compression_ratio = static_cast<double>(total_uncompressed_size) / total_compressed_size;
  return compressed_data_buffers;
}

std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::compact_compressed_buffers(
  const std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
  const std::vector<cudf::size_type>& compressed_sizes,
  const size_t num_buffers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  std::vector<std::unique_ptr<rmm::device_buffer>> new_compressed_data_buffers;
  new_compressed_data_buffers.reserve(num_buffers);
  for (size_t ix = 0; ix < num_buffers; ix++) {
    new_compressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(compressed_sizes[ix], stream, mr));
  }

  std::vector<void*> new_compressed_data_buffers_ptrs(num_buffers);
  std::vector<void*> compressed_data_buffers_ptrs(num_buffers);
  std::vector<size_t> compressed_sizes_vec(num_buffers);
  for (size_t ix = 0; ix < num_buffers; ix++) {
    new_compressed_data_buffers_ptrs[ix] =
      static_cast<void*>(new_compressed_data_buffers[ix]->data());
    compressed_data_buffers_ptrs[ix] = static_cast<void*>(compressed_data_buffers[ix]->data());
    compressed_sizes_vec[ix]         = static_cast<size_t>(compressed_sizes[ix]);
  }

  gqe::utility::do_batched_memcpy((void**)new_compressed_data_buffers_ptrs.data(),
                                  (void**)compressed_data_buffers_ptrs.data(),
                                  compressed_sizes_vec.data(),
                                  num_buffers,
                                  stream);
  return new_compressed_data_buffers;
}

std::pair<bool, bool> compression_manager::determine_best_compression(
  const double primary_compression_ratio, const double secondary_compression_ratio) const
{
  // Decide on the best compression algorithm
  bool is_compressed           = false;
  bool is_secondary_compressed = false;
  bool primary_compression_passes_threshold =
    primary_compression_ratio > _compression_ratio_threshold;
  bool secondary_compression_passes_threshold =
    secondary_compression_ratio > _secondary_compression_ratio_threshold;
  bool secondary_compression_better =
    secondary_compression_ratio >
    (primary_compression_ratio * _secondary_compression_multiplier_threshold);
  bool prefer_secondary =
    secondary_compression_passes_threshold and
    (secondary_compression_better or not primary_compression_passes_threshold);
  if (primary_compression_passes_threshold and not prefer_secondary) {
    is_compressed = true;
  } else if (prefer_secondary) {
    is_compressed           = true;
    is_secondary_compressed = true;
  }

  return std::make_pair(is_compressed, is_secondary_compressed);
}

std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::compress_batch(
  std::vector<std::unique_ptr<rmm::device_buffer>>&& device_uncompressed,
  bool& is_compressed,
  size_t& compressed_size,
  size_t& uncompressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  cudf::data_type cudf_type,
  memory_kind::type memory_kind,
  bool& is_secondary_compressed,
  bool try_secondary_compression,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  nvcompType_t data_type = get_optimal_nvcomp_data_type(cudf_type.id());

  const size_t num_buffers = device_uncompressed.size();

  size_t total_uncompressed_size = 0;
  for (size_t ix = 0; ix < num_buffers; ix++) {
    total_uncompressed_size += device_uncompressed[ix]->size();
  }
  uncompressed_size = total_uncompressed_size;

  size_t pinned_mem_size = num_buffers * sizeof(uint8_t*) +  // uncompressed ptrs
                           num_buffers * sizeof(uint8_t*);   // compressed ptrs

  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  rmm::device_buffer pinned_ptr_alloc_buffer(pinned_mem_size, stream, cudf_pinned_resource);
  uint8_t** pinned_ptr_alloc = reinterpret_cast<uint8_t**>(pinned_ptr_alloc_buffer.data());

  GQE_CUDA_TRY(cudaStreamSynchronize(stream));  // Necessary before filling these on the host
  uint8_t** decompressed_ptrs = pinned_ptr_alloc;
  uint8_t** compressed_ptrs   = decompressed_ptrs + num_buffers;

  // The decompressed ptrs are the same for all compression algorithms
  for (size_t ix = 0; ix < num_buffers; ix++) {
    auto& uncompressed    = device_uncompressed[ix];
    decompressed_ptrs[ix] = static_cast<uint8_t*>(uncompressed->data());
  }

  size_t primary_compressed_size   = 0;
  double primary_compression_ratio = 0.0;
  std::vector<cudf::size_type> primary_compressed_sizes;
  std::vector<std::unique_ptr<rmm::device_buffer>> primary_compressed_data_buffers;

  {
    utility::nvtx_scoped_range compress_primary_range("compress_primary");
    primary_compressed_data_buffers = try_compression_algorithm(device_uncompressed,
                                                                total_uncompressed_size,
                                                                decompressed_ptrs,
                                                                _comp_format,
                                                                data_type,
                                                                cudf_type,
                                                                num_buffers,
                                                                memory_kind,
                                                                compressed_ptrs,
                                                                primary_compression_ratio,
                                                                primary_compressed_size,
                                                                primary_compressed_sizes,
                                                                stream,
                                                                mr);
  }

  size_t secondary_compressed_size   = 0;
  double secondary_compression_ratio = 0.0;
  std::vector<cudf::size_type> secondary_compressed_sizes;
  std::vector<std::unique_ptr<rmm::device_buffer>> secondary_compressed_data_buffers;

  try_secondary_compression =
    try_secondary_compression and (_secondary_comp_format != compression_format::none);
  if (try_secondary_compression) {
    utility::nvtx_scoped_range compress_secondary_range("compress_secondary");
    secondary_compressed_data_buffers = try_compression_algorithm(device_uncompressed,
                                                                  total_uncompressed_size,
                                                                  decompressed_ptrs,
                                                                  _secondary_comp_format,
                                                                  data_type,
                                                                  cudf_type,
                                                                  num_buffers,
                                                                  memory_kind,
                                                                  compressed_ptrs,
                                                                  secondary_compression_ratio,
                                                                  secondary_compressed_size,
                                                                  secondary_compressed_sizes,
                                                                  stream,
                                                                  mr);
  }

  std::tie(is_compressed, is_secondary_compressed) =
    determine_best_compression(primary_compression_ratio, secondary_compression_ratio);

  // Finish the setup of the compressed data buffers
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;
  if (is_compressed) {
    if (is_secondary_compressed) {
      compressed_data_buffers = std::move(secondary_compressed_data_buffers);
      compressed_sizes        = std::move(secondary_compressed_sizes);
      compressed_size         = secondary_compressed_size;
      primary_compressed_data_buffers.clear();
    } else {
      compressed_data_buffers = std::move(primary_compressed_data_buffers);
      compressed_sizes        = std::move(primary_compressed_sizes);
      compressed_size         = primary_compressed_size;
      secondary_compressed_data_buffers.clear();
    }
  }

  const char* success_msg = is_compressed ? "successful" : "unsuccessful";
  GQE_LOG_TRACE(
    "Compression {} for column '{}' using compression algorithm {}: "
    "uncompressed_size={}, "
    "compressed_size={}, "
    "use_secondary_compression={}, "
    "try secondary compression={}, "
    "compression ratio threshold={}, "
    "secondary compression ratio threshold={}, "
    "secondary compression multiplier threshold={}, "
    "base compression ratio={}, "
    "secondary compression ratio={}, "
    "data type={}, "
    "cudf data type={}",
    success_msg,
    _column_name,
    compression_format_to_string(_comp_format),
    total_uncompressed_size,
    compressed_size,
    is_secondary_compressed,
    try_secondary_compression,
    _compression_ratio_threshold,
    _secondary_compression_ratio_threshold,
    _secondary_compression_multiplier_threshold,
    primary_compression_ratio,
    secondary_compression_ratio,
    static_cast<int>(data_type),
    static_cast<int>(cudf_type.id()));

  if (is_compressed) {
    compressed_data_buffers = compact_compressed_buffers(
      compressed_data_buffers, compressed_sizes, num_buffers, stream, mr);
  } else {
    utility::nvtx_scoped_range compress_failed_range("compress_failed");
    compressed_data_buffers = std::move(device_uncompressed);
    compressed_sizes.clear();
    for (size_t ix = 0; ix < num_buffers; ix++) {
      compressed_sizes.push_back(compressed_data_buffers[ix]->size());
    }
    compressed_size = total_uncompressed_size;
  }

  return compressed_data_buffers;
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_decompress(
  void const* compressed,
  size_t compressed_size,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  nvcompType_t data_type = get_optimal_nvcomp_data_type(_cudf_type.id());
  GQE_LOG_TRACE(
    "Starting decompression for column '{}': compressed_size={}, compression_algorithm={}, "
    "data_type={}, chunk_size={}, cudf_type={}",
    _column_name,
    compressed_size,
    compression_format_to_string(_comp_format),
    static_cast<int>(data_type),
    _compression_chunk_size,
    _cudf_type);

  rmm::device_buffer device_memory_compressed{
    static_cast<uint8_t const*>(compressed), compressed_size, supplied_stream, mr};
  auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

  auto manager       = create_manager(_comp_format, data_type, supplied_stream, mr);
  auto decomp_config = manager->configure_decompression(comp_buffer);
  auto decompressed_buffer =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, supplied_stream, mr);

  manager->decompress(
    static_cast<uint8_t*>(decompressed_buffer->data()), comp_buffer, decomp_config);

  GQE_LOG_TRACE(
    "Decompression completed for column '{}' using compression algorithm {}: compressed_size={}, "
    "decompressed_size={}",
    _column_name,
    compression_format_to_string(_comp_format),
    compressed_size,
    decompressed_buffer->size());

  return decompressed_buffer;
}

void compression_manager::print_usage() const
{
  GQE_LOG_ERROR(
    "Incorrect object initialization\nUsage: nvcompgqe object_name ([stream], "
    "[compression_format], [data_format], [chunk_size]) \n");
  GQE_LOG_ERROR("  %-60s One of < ans / cascaded / gdeflate / deflate / lz4 / snappy / zstd >\n",
                "[compression_format]");
  GQE_LOG_ERROR(
    "  %-60s Data format Options are < char / short / int / longlong / bits > (default value is "
    "'char')\n",
    "[data_format]");
  GQE_LOG_ERROR("  %-60s Chunk size (default value is 64 kB).\n", "[chunk_size]");
}

std::unique_ptr<nvcompManagerBase> compression_manager::create_manager(
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  std::unique_ptr<nvcompManagerBase> manager;

  bool use_de_sort = false;  // This flag determins whether or not to sort the chunks before HW
                             // decompression for load balancing purposes. Should be disabled when
                             // chunks are approximately the same size.
  int algorithm = 0;
  switch (comp_format) {
    case gqe::compression_format::lz4:
      GQE_LOG_TRACE("Creating LZ4 compression manager for column '{}'", _column_name);
      if (data_type == NVCOMP_TYPE_LONGLONG) {
        // LZ4 doesn't support LONGLONG, downcast to INT
        data_type = NVCOMP_TYPE_INT;
      }
      manager = std::make_unique<LZ4Manager>(
        _compression_chunk_size,
        nvcompBatchedLZ4CompressOpts_t{data_type, {0}},
        nvcompBatchedLZ4DecompressOpts_t{_decompress_backend, use_de_sort ? 1 : 0, {0}},
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::snappy:
      GQE_LOG_TRACE("Creating Snappy compression manager for column '{}'", _column_name);
      manager = std::make_unique<SnappyManager>(
        _compression_chunk_size,
        nvcompBatchedSnappyCompressOpts_t{{0}},
        nvcompBatchedSnappyDecompressOpts_t{_decompress_backend, use_de_sort ? 1 : 0, {0}},
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::ans:
      GQE_LOG_TRACE("Creating ANS compression manager for column '{}'", _column_name);
      manager = std::make_unique<ANSManager>(
        _compression_chunk_size,
        nvcompBatchedANSCompressOpts_t{nvcomp_rANS, nvcompType_t::NVCOMP_TYPE_CHAR, {0}},
        nvcompBatchedANSDecompressDefaultOpts,
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::cascaded: {
      GQE_LOG_TRACE("Creating Cascaded compression manager for column '{}'", _column_name);
      nvcompBatchedCascadedCompressOpts_t cascaded_opts = nvcompBatchedCascadedCompressDefaultOpts;
      cascaded_opts.type                                = data_type;
      manager = std::make_unique<CascadedManager>(_compression_chunk_size,
                                                  cascaded_opts,
                                                  nvcompBatchedCascadedDecompressDefaultOpts,
                                                  stream,
                                                  NoComputeNoVerify);
      break;
    }
    case gqe::compression_format::gdeflate:
      GQE_LOG_TRACE("Creating Gdeflate compression manager for column '{}'", _column_name);
      manager =
        std::make_unique<GdeflateManager>(_compression_chunk_size,
                                          nvcompBatchedGdeflateCompressOpts_t{algorithm, {0}},
                                          nvcompBatchedGdeflateDecompressDefaultOpts,
                                          stream,
                                          NoComputeNoVerify);
      break;
    case gqe::compression_format::deflate:
      GQE_LOG_TRACE("Creating Deflate compression manager for column '{}'", _column_name);
      manager = std::make_unique<DeflateManager>(
        _compression_chunk_size,
        nvcompBatchedDeflateCompressOpts_t{algorithm, {0}},
        nvcompBatchedDeflateDecompressOpts_t{_decompress_backend, use_de_sort ? 1 : 0, {0}},
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::zstd:
      GQE_LOG_TRACE("Creating Zstd compression manager for column '{}'", _column_name);
      manager = std::make_unique<ZstdManager>(static_cast<size_t>(_compression_chunk_size),
                                              nvcompBatchedZstdCompressDefaultOpts,
                                              nvcompBatchedZstdDecompressDefaultOpts,
                                              stream,
                                              NoComputeNoVerify);
      break;
    case gqe::compression_format::bitcomp:
      GQE_LOG_TRACE("Creating Bitcomp compression manager for column '{}'", _column_name);
      manager = std::make_unique<BitcompManager>(
        _compression_chunk_size,
        nvcompBatchedBitcompCompressOpts_t{algorithm, data_type, {0}},
        nvcompBatchedBitcompDecompressDefaultOpts,
        stream,
        NoComputeNoVerify);
      break;
    default:
      GQE_LOG_ERROR("Unrecognized Compression Format '{}' for column '{}'",
                    compression_format_to_string(comp_format),
                    _column_name);
      break;
  }

  auto alloc_fn = [mr, stream](std::size_t bytes) mutable {
    auto ptr = mr.allocate_async(bytes, alignof(std::max_align_t), stream);
    return ptr;
  };

  auto dealloc_fn = [mr, stream](void* ptr, std::size_t bytes) mutable {
    mr.deallocate_async(ptr, bytes, alignof(std::max_align_t), stream);
  };

  manager->set_scratch_allocators(alloc_fn, dealloc_fn);

  return manager;
}

std::unique_ptr<CPUHLIFManager> compression_manager::create_cpu_manager(int compression_level) const
{
  std::unique_ptr<CPUHLIFManager> cpu_manager;
  switch (_comp_format) {
    case gqe::compression_format::lz4:
      cpu_manager = std::make_unique<LZ4CPUHLIFManager>(_compression_chunk_size);
      cpu_manager->set_compression_level(compression_level);
      break;
    case gqe::compression_format::none:
    case gqe::compression_format::ans:
    case gqe::compression_format::snappy:
    case gqe::compression_format::gdeflate:
    case gqe::compression_format::deflate:
    case gqe::compression_format::cascaded:
    case gqe::compression_format::zstd:
    case gqe::compression_format::gzip:
    case gqe::compression_format::bitcomp:
    case gqe::compression_format::best_compression_ratio:
    case gqe::compression_format::best_decompression_speed:
    default: throw std::runtime_error("Unsupported compression format for CPU compression manager");
  }
  return cpu_manager;
}

gqe::compression_format compression_manager::get_comp_format() const { return _comp_format; }

int compression_manager::get_compression_chunk_size() const { return _compression_chunk_size; }

std::string compression_manager::get_column_name() const { return _column_name; }

cudf::data_type compression_manager::get_cudf_type() const { return _cudf_type; }

double compression_manager::get_compression_ratio_threshold() const
{
  return _compression_ratio_threshold;
}

bool compression_manager::get_use_cpu_compression() const { return _use_cpu_compression; }

int compression_manager::get_compression_level() const { return _compression_level; }
