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

#include "nvcomp_adapter.hpp"

#include <gqe/device_properties.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/cuda.hpp>

#include <cudf/utilities/pinned_memory.hpp>

#include <stdexcept>
#include <utility>

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

compression_manager::compression_manager(compression_configuration compression_config)
  : _config(std::move(compression_config))
{
  if (_config.secondary_compression_format != gqe::compression_format::none &&
      _config.secondary_compression_format == _config.primary_compression_format) {
    throw std::invalid_argument("Secondary compression format must differ from primary format");
  }

  if (_config.decompress_backend == gqe::decompression_backend::de) {
    auto const device_id = rmm::get_current_cuda_device();
    auto const device_supports_decomp =
      gqe::device_properties::instance()
        .get<gqe::device_properties::property::memDecompressSupport>(device_id);
    if (!device_supports_decomp) {
      throw std::invalid_argument(
        "Requested decompress_backend=DE on device without hardware decompression support");
    }
  }

  GQE_LOG_TRACE(
    "Created compression manager: format={}, decompress_backend={}, chunk_size={}, cudf_type={}, "
    "use_cpu_compression={}, compression_level={}",
    gqe::to_string(_config.primary_compression_format),
    gqe::to_string(_config.decompress_backend),
    _config.compression_chunk_size,
    _config.cudf_type,
    _config.use_cpu_compression,
    _config.compression_level);
}

void compression_manager::decompress_batch(nvcompType_t data_type,
                                           uint8_t* const device_decompressed_primary_ptr,
                                           const uint8_t* const* device_compressed,
                                           const uint8_t* const* host_compressed,
                                           const size_t batch_count,
                                           gqe::compression_format compression_format,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
{
  // variable not needed for decomp

  constexpr const size_t* compression_sizes = nullptr;
  auto decompression_manager =
    nvcomp_manager_adapter::create_manager(*this, compression_format, data_type, stream, mr);
  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  rmm::device_buffer device_decompressed_buffer(
    sizeof(uint8_t*) * batch_count, stream, cudf_pinned_resource);
  uint8_t** device_decompressed = reinterpret_cast<uint8_t**>(device_decompressed_buffer.data());
  stream.synchronize();

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
    decompression_configs.reserve(batch_count);
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

void decompression_batch::add(compressed_sliced_column::buffer_view host_source_buffer,
                              compression_manager const* manager,
                              std::vector<size_t> partition_indexes,
                              rmm::device_buffer* device_staging_buffer,
                              std::byte* output_ptr)
{
  if (partition_indexes.empty()) { return; }
  _requests.push_back(
    {host_source_buffer, manager, std::move(partition_indexes), device_staging_buffer, output_ptr});
}

void decompression_batch::reserve(size_t num_requests) { _requests.reserve(num_requests); }

bool decompression_batch::empty() const { return _requests.empty(); }

size_t decompression_batch::size() const { return _requests.size(); }

void decompression_batch::execute_async(rmm::cuda_stream_view stream) const
{
  for (const auto& request : _requests) {
    auto const data_type =
      get_optimal_nvcomp_data_type(request.host_source_buffer.element_type().id());

    auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
    rmm::device_buffer staged_ptrs_buffer(
      sizeof(uint8_t const*) * request.partition_indexes.size(), stream, cudf_pinned_resource);
    auto** staged_ptrs = reinterpret_cast<uint8_t const**>(staged_ptrs_buffer.data());
    stream.synchronize();

    std::vector<uint8_t const*> host_buffers;
    host_buffers.reserve(request.partition_indexes.size());
    auto* staged_ptr = reinterpret_cast<std::byte*>(request.device_staging_buffer->data());
    for (size_t copy_idx = 0; copy_idx < request.partition_indexes.size(); ++copy_idx) {
      size_t partition_idx  = request.partition_indexes[copy_idx];
      auto const partition  = request.host_source_buffer.get_partition(partition_idx);
      staged_ptrs[copy_idx] = reinterpret_cast<uint8_t const*>(staged_ptr);
      host_buffers.push_back(partition.data());
      staged_ptr += rmm::align_up(partition.size(), size_t{8});
    }

    request.manager->decompress_batch(data_type,
                                      reinterpret_cast<uint8_t*>(request.output_ptr),
                                      staged_ptrs,
                                      host_buffers.data(),
                                      request.partition_indexes.size(),
                                      request.host_source_buffer.compression_format(),
                                      stream,
                                      rmm::mr::get_current_device_resource_ref());
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
    new_compressed_data_buffers_ptrs[ix] = new_compressed_data_buffers[ix]->data();
    compressed_data_buffers_ptrs[ix]     = compressed_data_buffers[ix]->data();
    compressed_sizes_vec[ix]             = static_cast<size_t>(compressed_sizes[ix]);
  }

  gqe::utility::do_batched_memcpy((void**)new_compressed_data_buffers_ptrs.data(),
                                  (void**)compressed_data_buffers_ptrs.data(),
                                  compressed_sizes_vec.data(),
                                  num_buffers,
                                  stream);
  return new_compressed_data_buffers;
}

gqe::compression_format compression_manager::determine_best_compression(
  const double primary_compression_ratio, const double secondary_compression_ratio) const
{
  // Decide on the best compression algorithm
  bool primary_compression_passes_threshold =
    primary_compression_ratio > _config.compression_ratio_threshold;
  bool secondary_compression_passes_threshold =
    secondary_compression_ratio > _config.secondary_compression_ratio_threshold;
  bool secondary_compression_better =
    secondary_compression_ratio >
    (primary_compression_ratio * _config.secondary_compression_multiplier_threshold);
  bool prefer_secondary = secondary_compression_passes_threshold &&
                          (secondary_compression_better || !primary_compression_passes_threshold);
  if (primary_compression_passes_threshold && !prefer_secondary) {
    return _config.primary_compression_format;
  } else if (prefer_secondary) {
    return _config.secondary_compression_format;
  } else {
    return gqe::compression_format::none;
  }
}

std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::compress_batch(
  nvcompType_t data_type,
  std::vector<std::unique_ptr<rmm::device_buffer>>&& device_uncompressed,
  gqe::compression_format& selected_format,
  size_t& compressed_size,
  size_t& uncompressed_size,
  size_t& primary_compressed_size,
  size_t& secondary_compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  memory_kind::type memory_kind,
  bool try_secondary_compression,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
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

  stream.synchronize();  // Necessary before filling these on the host
  uint8_t** decompressed_ptrs = pinned_ptr_alloc;
  uint8_t** compressed_ptrs   = decompressed_ptrs + num_buffers;

  // The decompressed ptrs are the same for all compression algorithms
  for (size_t ix = 0; ix < num_buffers; ix++) {
    auto& uncompressed    = device_uncompressed[ix];
    decompressed_ptrs[ix] = reinterpret_cast<uint8_t*>(uncompressed->data());
  }

  double primary_compression_ratio = 0.0;
  std::vector<cudf::size_type> primary_compressed_sizes;
  std::vector<std::unique_ptr<rmm::device_buffer>> primary_compressed_data_buffers;

  {
    // Check if CPU compression is requested and supported for primary compression format
    const bool use_cpu_compression =
      _config.use_cpu_compression && is_algorithm_cpu_supported(_config.primary_compression_format);
    auto compression_impl = use_cpu_compression
                              ? dispatch_compression_adapter<nvcomp_cpu_manager_adapter>
                              : dispatch_compression_adapter<nvcomp_manager_adapter>;
    utility::nvtx_scoped_range compress_primary_range("compress_primary");
    primary_compressed_data_buffers = compression_impl(*this,
                                                       device_uncompressed,
                                                       total_uncompressed_size,
                                                       decompressed_ptrs,
                                                       _config.primary_compression_format,
                                                       data_type,
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

  try_secondary_compression = try_secondary_compression && (_config.secondary_compression_format !=
                                                            gqe::compression_format::none);
  if (try_secondary_compression) {
    // Check if CPU compression is requested and supported for secondary compression format
    const bool use_cpu_compression =
      _config.use_cpu_compression &&
      is_algorithm_cpu_supported(_config.secondary_compression_format);
    auto compression_impl = use_cpu_compression
                              ? dispatch_compression_adapter<nvcomp_cpu_manager_adapter>
                              : dispatch_compression_adapter<nvcomp_manager_adapter>;
    utility::nvtx_scoped_range compress_secondary_range("compress_secondary");
    secondary_compressed_data_buffers = compression_impl(*this,
                                                         device_uncompressed,
                                                         total_uncompressed_size,
                                                         decompressed_ptrs,
                                                         _config.secondary_compression_format,
                                                         data_type,
                                                         num_buffers,
                                                         memory_kind,
                                                         compressed_ptrs,
                                                         secondary_compression_ratio,
                                                         secondary_compressed_size,
                                                         secondary_compressed_sizes,
                                                         stream,
                                                         mr);
  }

  selected_format =
    determine_best_compression(primary_compression_ratio, secondary_compression_ratio);

  // Finish the setup of the compressed data buffers
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;
  if (selected_format == _config.primary_compression_format) {
    compressed_data_buffers = std::move(primary_compressed_data_buffers);
    compressed_sizes        = std::move(primary_compressed_sizes);
    compressed_size         = primary_compressed_size;
    secondary_compressed_data_buffers.clear();
  } else if (selected_format == _config.secondary_compression_format) {
    compressed_data_buffers = std::move(secondary_compressed_data_buffers);
    compressed_sizes        = std::move(secondary_compressed_sizes);
    compressed_size         = secondary_compressed_size;
    primary_compressed_data_buffers.clear();
  } else if (selected_format != gqe::compression_format::none) {
    throw std::logic_error("Selected compression format does not match configured formats");
  }

  const bool is_compressed = selected_format != gqe::compression_format::none;
  const bool is_secondary_compressed =
    is_compressed && (selected_format == _config.secondary_compression_format);
  const char* success_msg = is_compressed ? "successful" : "unsuccessful";
  GQE_LOG_DEBUG(
    "Compression {} using compression algorithm {}: "
    "uncompressed_size={}, "
    "compressed_size={}, "
    "use_secondary_compression={}, "
    "try_secondary_compression={}, "
    "compression ratio threshold={}, "
    "secondary compression ratio threshold={}, "
    "secondary compression multiplier threshold={}, "
    "primary compression ratio={}, "
    "secondary compression ratio={}, "
    "data type={}",
    success_msg,
    gqe::to_string(selected_format),
    total_uncompressed_size,
    compressed_size,
    is_secondary_compressed,
    try_secondary_compression,
    _config.compression_ratio_threshold,
    _config.secondary_compression_ratio_threshold,
    _config.secondary_compression_multiplier_threshold,
    primary_compression_ratio,
    secondary_compression_ratio,
    static_cast<int>(data_type));

  if (is_compressed) {
    compressed_data_buffers = compact_compressed_buffers(
      compressed_data_buffers, compressed_sizes, num_buffers, stream, mr);
  } else {
    utility::nvtx_scoped_range compress_failed_range("compress_failed");
    compressed_data_buffers = std::move(device_uncompressed);
    compressed_sizes.clear();
    compressed_sizes.reserve(num_buffers);
    for (size_t ix = 0; ix < num_buffers; ix++) {
      compressed_sizes.push_back(compressed_data_buffers[ix]->size());
    }
    compressed_size = total_uncompressed_size;
  }

  return compressed_data_buffers;
}

gqe::compression_format compression_manager::primary_compression_format() const
{
  return _config.primary_compression_format;
}

gqe::compression_format compression_manager::secondary_compression_format() const
{
  return _config.secondary_compression_format;
}

int compression_manager::get_compression_chunk_size() const
{
  return _config.compression_chunk_size;
}

gqe::decompression_backend compression_manager::get_decompress_backend() const
{
  return _config.decompress_backend;
}

int compression_manager::get_compression_level() const { return _config.compression_level; }

}  // namespace storage
}  // namespace gqe
