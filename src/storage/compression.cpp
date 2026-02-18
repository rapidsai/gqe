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

#include <gqe/storage/compression.hpp>

#include "nvcomp_adapter.hpp"

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/utilities/pinned_memory.hpp>

#include <cuda.h>

#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace {

// Helper function to check if a compression algorithm is supported on the CPU
bool is_algorithm_cpu_supported(gqe::compression_format comp_format)
{
  return comp_format == gqe::compression_format::lz4;
}

// Helper function to dispatch compression to the appropriate adapter
template <typename nvcomp_manager_adapter_type>
std::vector<std::unique_ptr<rmm::device_buffer>> dispatch_compression_adapter(
  gqe::storage::compression_manager const& comp_manager,
  const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
  const size_t total_uncompressed_size,
  uint8_t** uncompressed_ptrs,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  cudf::data_type cudf_type,
  size_t num_buffers,
  gqe::memory_kind::type memory_kind,
  uint8_t** compressed_ptrs,
  double& compression_ratio,
  size_t& total_compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return nvcomp_manager_adapter_type::compress_batch(comp_manager,
                                                     device_uncompressed,
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

}  // namespace

namespace gqe {
namespace storage {
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
    gqe::compression_format_to_string(_comp_format),
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
    gqe::compression_format_to_string(_comp_format),
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
      gqe::compression_format_to_string(_comp_format),
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
    gqe::compression_format_to_string(_comp_format),
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
    gqe::compression_format_to_string(_comp_format),
    static_cast<int>(data_type),
    _compression_chunk_size,
    _cudf_type);

  auto decompression_manager = create_manager(_comp_format, data_type, stream, mr);

  nvcomp::DecompressionConfig decomp_config =
    decompression_manager->configure_decompression(compressed.data());

  std::unique_ptr<rmm::device_buffer> decompressed_data =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, stream, mr);

  decompression_manager->decompress(
    (uint8_t*)decompressed_data->data(), compressed.data(), decomp_config);

  GQE_LOG_TRACE(
    "Decompression completed for column '{}' using compression algorithm {}: compressed_size={}, "
    "decompressed_size={}",
    _column_name,
    gqe::compression_format_to_string(_comp_format),
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
  size_t& primary_compressed_size,
  size_t& secondary_compressed_size,
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

  double primary_compression_ratio = 0.0;
  std::vector<cudf::size_type> primary_compressed_sizes;
  std::vector<std::unique_ptr<rmm::device_buffer>> primary_compressed_data_buffers;

  {
    // Check if CPU compression is requested and supported for primary compression format
    const bool use_cpu_compression =
      _use_cpu_compression && is_algorithm_cpu_supported(_comp_format);
    auto compression_impl = use_cpu_compression
                              ? dispatch_compression_adapter<nvcomp_cpu_manager_adapter>
                              : dispatch_compression_adapter<nvcomp_manager_adapter>;
    utility::nvtx_scoped_range compress_primary_range("compress_primary");
    primary_compressed_data_buffers = compression_impl(*this,
                                                       device_uncompressed,
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

  double secondary_compression_ratio = 0.0;
  std::vector<cudf::size_type> secondary_compressed_sizes;
  std::vector<std::unique_ptr<rmm::device_buffer>> secondary_compressed_data_buffers;

  try_secondary_compression =
    try_secondary_compression and (_secondary_comp_format != compression_format::none);
  if (try_secondary_compression) {
    // Check if CPU compression is requested and supported for secondary compression format
    const bool use_cpu_compression =
      _use_cpu_compression && is_algorithm_cpu_supported(_secondary_comp_format);
    auto compression_impl = use_cpu_compression
                              ? dispatch_compression_adapter<nvcomp_cpu_manager_adapter>
                              : dispatch_compression_adapter<nvcomp_manager_adapter>;
    utility::nvtx_scoped_range compress_secondary_range("compress_secondary");
    secondary_compressed_data_buffers = compression_impl(*this,
                                                         device_uncompressed,
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
  GQE_LOG_DEBUG(
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
    gqe::compression_format_to_string(_comp_format),
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
    gqe::compression_format_to_string(_comp_format),
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
    gqe::compression_format_to_string(_comp_format),
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

std::unique_ptr<gqe::storage::nvcomp_manager_adapter> compression_manager::create_manager(
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return gqe::storage::nvcomp_manager_adapter::create_manager(
    *this, comp_format, data_type, stream, mr);
}

std::unique_ptr<gqe::storage::nvcomp_cpu_manager_adapter> compression_manager::create_cpu_manager(
  int compression_level) const
{
  return gqe::storage::nvcomp_cpu_manager_adapter::create_cpu_manager(*this, compression_level);
}

gqe::compression_format compression_manager::get_comp_format() const { return _comp_format; }

int compression_manager::get_compression_chunk_size() const { return _compression_chunk_size; }

std::string compression_manager::get_column_name() const { return _column_name; }

cudf::data_type compression_manager::get_cudf_type() const { return _cudf_type; }

double compression_manager::get_compression_ratio_threshold() const
{
  return _compression_ratio_threshold;
}

nvcompDecompressBackend_t compression_manager::get_decompress_backend() const
{
  return _decompress_backend;
}

bool compression_manager::get_use_cpu_compression() const { return _use_cpu_compression; }

int compression_manager::get_compression_level() const { return _compression_level; }

}  // namespace storage
}  // namespace gqe
