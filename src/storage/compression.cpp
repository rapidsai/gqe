/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>
#include <gqe/storage/compression.hpp>

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
                                         int explicit_chunk_size,
                                         std::string column_name,
                                         cudf::data_type cudf_type)
  : _comp_format(comp_format),
    _data_type(data_format),
    _chunk_size(explicit_chunk_size),
    _column_name(column_name),
    _cudf_type(cudf_type)
{
  GQE_LOG_INFO(
    "Created compression manager for column '{}': format={}, data_type={}, chunk_size={}, "
    "cudf_type={}",
    _column_name,
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _chunk_size,
    cudf_type_to_string(_cudf_type.id()));
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_compress(
  rmm::device_buffer const* uncompressed,
  float& compression_ratio,
  bool& is_compressed,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  is_compressed = true;

  GQE_LOG_INFO(
    "Starting compression for column '{}': input_size={}, compression_algorithm={}, data_type={}, "
    "chunk_size={}, cudf_type={}",
    _column_name,
    uncompressed->size(),
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  auto manager           = create_manager(supplied_stream, mr);
  auto comp_config       = manager->configure_compression(uncompressed->size());
  auto compressed_buffer = std::make_unique<rmm::device_buffer>(
    comp_config.max_compressed_buffer_size, supplied_stream, mr);

  manager->compress(static_cast<uint8_t const*>(uncompressed->data()),
                    static_cast<uint8_t*>(compressed_buffer->data()),
                    comp_config);

  auto const comp_size =
    manager->get_compressed_output_size(static_cast<uint8_t*>(compressed_buffer->data()));
  compression_ratio = static_cast<float>(uncompressed->size()) / comp_size;

  compressed_buffer->resize(comp_size, supplied_stream);
  compressed_buffer->shrink_to_fit(supplied_stream);
  manager->deallocate_gpu_mem();

  if (comp_size > uncompressed->size()) {
    is_compressed = false;
    GQE_LOG_INFO(
      "Compression ineffective for column '{}' using compression algorithm {}: compressed_size={} "
      "> uncompressed_size={}, compression_ratio={:.2f}",
      _column_name,
      compression_format_to_string(_comp_format),
      comp_size,
      uncompressed->size(),
      compression_ratio);
    return std::make_unique<rmm::device_buffer>(*uncompressed, supplied_stream, mr);
  }

  GQE_LOG_INFO(
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
  rmm::device_buffer const* compressed,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  GQE_LOG_INFO(
    "Starting decompression for column '{}': compressed_size={}, compression_algorithm={}, "
    "data_type={}, chunk_size={}, cudf_type={}",
    _column_name,
    compressed->size(),
    compression_format_to_string(_comp_format),
    static_cast<int>(_data_type),
    _chunk_size,
    cudf_type_to_string(_cudf_type.id()));

  rmm::device_buffer device_memory_compressed{
    compressed->data(), compressed->size(), supplied_stream, mr};
  auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

  auto manager       = create_manager(supplied_stream, mr);
  auto decomp_config = manager->configure_decompression(comp_buffer);
  auto decompressed_buffer =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, supplied_stream, mr);

  manager->decompress(
    static_cast<uint8_t*>(decompressed_buffer->data()), comp_buffer, decomp_config);
  manager->deallocate_gpu_mem();

  GQE_LOG_INFO(
    "Decompression completed for column '{}' using compression algorithm {}: compressed_size={}, "
    "decompressed_size={}",
    _column_name,
    compression_format_to_string(_comp_format),
    compressed->size(),
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
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref& mr) const
{
  std::unique_ptr<nvcompManagerBase> manager;

  switch (_comp_format) {
    case gqe::compression_format::lz4:
      GQE_LOG_INFO("Creating LZ4 compression manager for column '{}'", _column_name);
      manager = std::make_unique<LZ4Manager>(
        _chunk_size, nvcompBatchedLZ4Opts_t{_data_type}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::snappy:
      GQE_LOG_INFO("Creating Snappy compression manager for column '{}'", _column_name);
      manager = std::make_unique<SnappyManager>(
        _chunk_size, nvcompBatchedSnappyOpts_t{}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::ans:
      GQE_LOG_INFO("Creating ANS compression manager for column '{}'", _column_name);
      manager = std::make_unique<ANSManager>(
        _chunk_size, nvcompBatchedANSDefaultOpts, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::cascaded: {
      GQE_LOG_INFO("Creating Cascaded compression manager for column '{}'", _column_name);
      nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;
      cascaded_opts.type                        = _data_type;
      cascaded_opts.internal_chunk_bytes        = _chunk_size;
      manager =
        std::make_unique<CascadedManager>(_chunk_size, cascaded_opts, stream, NoComputeNoVerify);
      break;
    }
    case gqe::compression_format::gdeflate:
      GQE_LOG_INFO("Creating Gdeflate compression manager for column '{}'", _column_name);
      manager = std::make_unique<GdeflateManager>(
        _chunk_size, nvcompBatchedGdeflateOpts_t{0}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::deflate:
      GQE_LOG_INFO("Creating Deflate compression manager for column '{}'", _column_name);
      manager = std::make_unique<DeflateManager>(
        _chunk_size, nvcompBatchedDeflateOpts_t{5}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::zstd:
      GQE_LOG_INFO("Creating Zstd compression manager for column '{}'", _column_name);
      manager = std::make_unique<ZstdManager>(
        static_cast<size_t>(_chunk_size), nvcompBatchedZstdDefaultOpts, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::bitcomp:
      GQE_LOG_INFO("Creating Bitcomp compression manager for column '{}'", _column_name);
      manager = std::make_unique<BitcompManager>(
        _chunk_size, nvcompBatchedBitcompFormatOpts{0, _data_type}, stream, NoComputeNoVerify);
      break;
    default:
      GQE_LOG_ERROR("Unrecognized Compression Format '{}' for column '{}'",
                    compression_format_to_string(_comp_format),
                    _column_name);
      break;
  }

  auto alloc_fn = [&mr, stream](std::size_t bytes) {
    return mr.allocate_async(bytes, alignof(std::max_align_t), stream);
  };

  auto dealloc_fn = [&mr, stream](void* ptr, std::size_t bytes) {
    mr.deallocate_async(ptr, bytes, alignof(std::max_align_t), stream);
  };

  manager->set_scratch_allocators(alloc_fn, dealloc_fn);

  return manager;
}

gqe::compression_format compression_manager::get_comp_format() const { return _comp_format; }

nvcompType_t compression_manager::get_data_type() const { return _data_type; }

int compression_manager::get_chunk_size() const { return _chunk_size; }

std::string compression_manager::get_column_name() const { return _column_name; }

cudf::data_type compression_manager::get_cudf_type() const { return _cudf_type; }