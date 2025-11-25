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
#include <gqe/utility/error.hpp>

// Helper function to convert CUDF type ID to string for logging
std::string cudf_type_to_string(cudf::type_id type_id)
{
  switch (type_id) {
    case cudf::type_id::EMPTY: return "EMPTY";
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::BOOL8: return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS: return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS: return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS: return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS: return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32: return "DICTIONARY32";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::LIST: return "LIST";
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    default: return "UNKNOWN";
  }
}

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

compression_manager::compression_manager(gqe::compression_format comp_format,
                                         nvcompType_t data_format,
                                         int compression_chunk_size,
                                         rmm::cuda_stream_view supplied_stream,
                                         rmm::device_async_resource_ref mr,
                                         double compression_ratio_threshold,
                                         std::string column_name,
                                         cudf::data_type cudf_type)
  : _comp_format(comp_format),
    _data_type(data_format),
    _compression_chunk_size(compression_chunk_size),
    _column_name(column_name),
    _cudf_type(cudf_type),
    _compression_ratio_threshold(compression_ratio_threshold)
{
  GQE_LOG_TRACE(
    "Created compression manager for column '{}': format={}, data_type={}, chunk_size={}, "
    "cudf_type={}",
    _column_name,
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _compression_chunk_size,
    cudf_type_to_string(_cudf_type.id()));
}

compression_result compression_manager::do_compress(rmm::device_buffer const* uncompressed,
                                                    float& compression_ratio,
                                                    bool& is_compressed,
                                                    rmm::cuda_stream_view supplied_stream,
                                                    rmm::device_async_resource_ref mr)
{
  is_compressed = true;

  GQE_LOG_TRACE(
    "Starting compression for column '{}': input_size={}, compression_algorithm={}, data_type={}, "
    "chunk_size={}, cudf_type={}",
    _column_name,
    uncompressed->size(),
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _compression_chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  auto compression_manager = create_manager(supplied_stream, mr);
  auto comp_config         = compression_manager->configure_compression(uncompressed->size());
  auto compressed_buffer   = std::make_unique<rmm::device_buffer>(
    comp_config.max_compressed_buffer_size, supplied_stream, mr);

  compression_manager->compress(static_cast<uint8_t const*>(uncompressed->data()),
                                static_cast<uint8_t*>(compressed_buffer->data()),
                                comp_config);

  auto const comp_size = compression_manager->get_compressed_output_size(
    static_cast<uint8_t*>(compressed_buffer->data()));

  compression_ratio = static_cast<double>(uncompressed->size()) / comp_size;

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
    return std::make_pair(std::make_unique<rmm::device_buffer>(*uncompressed, supplied_stream, mr),
                          std::nullopt);
  }

  GQE_LOG_TRACE(
    "Compression successful for column '{}' using compression algorithm {}: uncompressed_size={}, "
    "compressed_size={}, compression_ratio={:.2f}",
    _column_name,
    compression_format_to_string(_comp_format),
    uncompressed->size(),
    comp_size,
    compression_ratio);

  return std::make_pair(std::move(compressed_buffer), std::move(comp_config));
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_decompress(
  cudf::device_span<uint8_t const> compressed,
  nvcomp::CompressionConfig const& compression_config,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  GQE_LOG_TRACE(
    "Starting decompression for column '{}': compressed_size={}, compression_algorithm={}, "
    "data_type={}, chunk_size={}, cudf_type={}",
    _column_name,
    compressed.size(),
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _compression_chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  auto decompression_manager = create_manager(stream, mr);

  DecompressionConfig decomp_config =
    decompression_manager->configure_decompression(compression_config);

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

void compression_manager::decompress_batch(uint8_t* const device_decompressed_base_ptr,
                                           const uint8_t* const* device_compressed,
                                           const uint8_t* const* host_compressed,
                                           const size_t batch_count,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // variable not needed for decomp
  constexpr const size_t* compression_sizes = nullptr;
  auto decompression_manager                = create_manager(stream, mr);
  auto cudf_pinned_resource                 = cudf::get_pinned_memory_resource();
  uint8_t** device_decompressed             = reinterpret_cast<uint8_t**>(
    cudf_pinned_resource.allocate_async(sizeof(uint8_t*) * batch_count, alignof(uint8_t*), stream));
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
      device_decompressed[i] = device_decompressed_base_ptr + decompressed_offset;
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
      device_decompressed[i] = device_decompressed_base_ptr + decompressed_offset;
      decompressed_offset += decompression_configs[i].decomp_data_size;
    }
    decompression_manager->decompress(device_decompressed,
                                      device_compressed,
                                      decompression_configs,
                                      batch_count,
                                      compression_sizes);
  }
  GQE_CUDA_TRY(cudaStreamSynchronize(stream));
  cudf_pinned_resource.deallocate_async(
    device_decompressed, sizeof(uint8_t*) * batch_count, stream);
}

/*
std::tie(compressed_data_buffers, compression_configs) = manager.compress_batch(
    device_uncompressed_data_buffers, _compression_ratio,
    is_compressed, compressed_size, compressed_sizes, stream, mr);
*/
std::vector<std::unique_ptr<rmm::device_buffer>> compression_manager::compress_batch(
  const std::vector<rmm::device_buffer>& device_uncompressed,
  float& compression_ratio,
  bool& is_compressed,
  size_t& compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;
  std::vector<nvcomp::CompressionConfig> compression_configs;
  size_t total_uncompressed_size = 0;

  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  size_t pinned_mem_size    = device_uncompressed.size() * sizeof(uint8_t*) * 2;
  void* pinned_alloc =
    cudf_pinned_resource.allocate_async(pinned_mem_size, alignof(uint8_t*), stream);
  uint8_t** compressed_ptrs = reinterpret_cast<uint8_t**>(pinned_alloc);
  uint8_t const** decompressed_ptrs =
    (uint8_t const**)(compressed_ptrs + device_uncompressed.size());

  auto compression_manager = create_manager(stream, mr);
  for (size_t ix = 0; ix < device_uncompressed.size(); ix++) {
    auto& uncompressed = device_uncompressed[ix];
    auto config        = compression_manager->configure_compression(uncompressed.size());
    compression_configs.push_back(config);
    compressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(config.max_compressed_buffer_size, stream, mr));
    total_uncompressed_size += uncompressed.size();
    decompressed_ptrs[ix] = static_cast<const uint8_t*>(uncompressed.data());
    compressed_ptrs[ix]   = static_cast<uint8_t*>(compressed_data_buffers.back()->data());
  }

  GQE_LOG_TRACE(
    "Starting batched compression for column '{}': input_size={}, compression_algorithm={}, num "
    "partitions={}, data_type={}, "
    "chunk_size={}, cudf_type={}",
    _column_name,
    total_uncompressed_size,
    compression_format_to_string(_comp_format),
    device_uncompressed.size(),
    static_cast<int>(_data_type),
    _compression_chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  compression_manager->compress(decompressed_ptrs, compressed_ptrs, compression_configs);

  size_t total_compressed_size = 0;
  for (size_t ix = 0; ix < compressed_data_buffers.size(); ix++) {
    size_t comp_size = compression_manager->get_compressed_output_size(
      reinterpret_cast<const uint8_t*>(compressed_data_buffers[ix]->data()));
    total_compressed_size += comp_size;
    compressed_sizes.push_back(comp_size);
    compressed_data_buffers[ix]->resize(comp_size, stream);
    compressed_data_buffers[ix]->shrink_to_fit(stream);
  }

  compression_ratio = static_cast<double>(total_uncompressed_size) / total_compressed_size;
  if (_compression_ratio_threshold < compression_ratio) {
    is_compressed = true;
    GQE_LOG_TRACE(
      "Compression successful for column '{}' using compression algorithm {}: "
      "uncompressed_size={}, "
      "compressed_size={}, compression_ratio={:.2f}",
      _column_name,
      compression_format_to_string(_comp_format),
      total_uncompressed_size,
      total_compressed_size,
      compression_ratio);
  } else {
    GQE_LOG_TRACE(
      "Compression unsuccessful for column '{}' using compression algorithm {}: "
      "uncompressed_size={}, "
      "compressed_size={}, compression_ratio={:.2f}",
      _column_name,
      compression_format_to_string(_comp_format),
      total_uncompressed_size,
      total_compressed_size,
      compression_ratio);
    is_compressed = false;
    compressed_data_buffers.clear();
    compressed_sizes.clear();
    compression_configs.clear();
    for (size_t ix = 0; ix < device_uncompressed.size(); ix++) {
      compressed_data_buffers.emplace_back(
        std::make_unique<rmm::device_buffer>(device_uncompressed[ix], stream, mr));
      compressed_sizes.push_back(compressed_data_buffers.back()->size());
    }
    total_compressed_size = total_uncompressed_size;
    compression_ratio     = 1.0f;
  }

  compressed_size = total_compressed_size;

  cudf_pinned_resource.deallocate_async(pinned_alloc, pinned_mem_size, stream);

  // no need to std::move, copy elision
  return compressed_data_buffers;
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_decompress(
  void const* compressed,
  size_t compressed_size,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  GQE_LOG_TRACE(
    "Starting decompression for column '{}': compressed_size={}, compression_algorithm={}, "
    "data_type={}, chunk_size={}, cudf_type={}",
    _column_name,
    compressed_size,
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _compression_chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  rmm::device_buffer device_memory_compressed{
    static_cast<uint8_t const*>(compressed), compressed_size, supplied_stream, mr};
  auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

  auto manager       = create_manager(supplied_stream, mr);
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
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
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
  bool device_supports_decomp = static_cast<bool>(decompressSupportMask);

  std::unique_ptr<nvcompManagerBase> manager;
  nvcompDecompressBackend_t backend =
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT;  // This flag tells the program to use HW decompression if
                                        // available. And if not, fallback to SW decompression.

  if (device_supports_decomp) { backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE; }

  bool use_de_sort = false;  // This flag determins whether or not to sort the chunks before HW
                             // decompression for load balancing purposes. Should be disabled when
                             // chunks are approximately the same size.
  int algorithm = 0;

  switch (_comp_format) {
    case gqe::compression_format::lz4:
      GQE_LOG_TRACE("Creating LZ4 compression manager for column '{}'", _column_name);
      manager = std::make_unique<LZ4Manager>(
        _compression_chunk_size,
        nvcompBatchedLZ4CompressOpts_t{NVCOMP_TYPE_UCHAR, {0}},
        nvcompBatchedLZ4DecompressOpts_t{backend, use_de_sort ? 1 : 0, {0}},
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::snappy:
      GQE_LOG_TRACE("Creating Snappy compression manager for column '{}'", _column_name);
      manager = std::make_unique<SnappyManager>(
        _compression_chunk_size,
        nvcompBatchedSnappyCompressOpts_t{{0}},
        nvcompBatchedSnappyDecompressOpts_t{backend, use_de_sort ? 1 : 0, {0}},
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::ans:
      GQE_LOG_TRACE("Creating ANS compression manager for column '{}'", _column_name);
      manager = std::make_unique<ANSManager>(
        _compression_chunk_size,
        nvcompBatchedANSCompressOpts_t{nvcomp_rANS, NVCOMP_TYPE_UCHAR, {0}},
        nvcompBatchedANSDecompressDefaultOpts,
        stream,
        NoComputeNoVerify);
      break;
    case gqe::compression_format::cascaded: {
      GQE_LOG_TRACE("Creating Cascaded compression manager for column '{}'", _column_name);
      nvcompBatchedCascadedCompressOpts_t cascaded_opts = nvcompBatchedCascadedCompressOpts_t{};
      cascaded_opts.type                                = _data_type;
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
        nvcompBatchedDeflateDecompressOpts_t{backend, use_de_sort ? 1 : 0, {0}},
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
        nvcompBatchedBitcompCompressOpts_t{algorithm, _data_type, {0}},
        nvcompBatchedBitcompDecompressDefaultOpts,
        stream,
        NoComputeNoVerify);
      break;
    default:
      GQE_LOG_ERROR("Unrecognized Compression Format '{}' for column '{}'",
                    compression_format_to_string(_comp_format),
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

gqe::compression_format compression_manager::get_comp_format() const { return _comp_format; }

nvcompType_t compression_manager::get_data_type() const { return _data_type; }

int compression_manager::get_compression_chunk_size() const { return _compression_chunk_size; }

std::string compression_manager::get_column_name() const { return _column_name; }

cudf::data_type compression_manager::get_cudf_type() const { return _cudf_type; }

double compression_manager::get_compression_ratio_threshold() const
{
  return _compression_ratio_threshold;
}
