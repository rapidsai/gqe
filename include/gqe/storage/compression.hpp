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

#pragma once

#include <gqe/types.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <nvcomp/shared_types.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

// Forward declaration
class nvcomp_manager_adapter;
class nvcomp_cpu_manager_adapter;

/**
 * @brief Class that contains functions for compression and decompression of columns in GQE.
 */
class compression_manager {
 private:
  gqe::compression_format _comp_format;
  gqe::compression_format _secondary_comp_format;
  int _compression_chunk_size;
  std::string _column_name;
  cudf::data_type _cudf_type;
  double _compression_ratio_threshold;
  double _secondary_compression_ratio_threshold;
  double _secondary_compression_multiplier_threshold;
  nvcompDecompressBackend_t _decompress_backend;
  bool _use_cpu_compression;
  int _compression_level;

  void print_usage() const;
  /**
   * @brief Factory method to create a nvcomp_manager_adapter object based on the compression
   * format.
   *
   * @param[in] supplied_stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator and deallocating scratch memory required for
   * nvcomp
   */
  std::unique_ptr<gqe::storage::nvcomp_manager_adapter> create_manager(
    gqe::compression_format comp_format,
    nvcompType_t data_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Helper function to compact the compressed data buffers
   *
   * @param[in] compressed_data_buffers Array of pointers to compressed buffers
   * @param[in] compressed_sizes Vector of sizes of the compressed buffers
   * @param[in] num_buffers Number of buffers to compact
   * @param[in] stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for compression/decompression
   * @return Vector of pointers to compressed buffers
   */
  std::vector<std::unique_ptr<rmm::device_buffer>> compact_compressed_buffers(
    const std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
    const std::vector<cudf::size_type>& compressed_sizes,
    const size_t num_buffers,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Helper function to determine the best compression algorithm
   *
   * @param[in] base_compression_ratio Compression ratio of the base compression algorithm
   * @param[in] secondary_compression_ratio Compression ratio of the secondary compression algorithm
   * @return a pair of booleans indicating whether compression was viable or not and whether
   secondary compression was used
   */
  std::pair<bool, bool> determine_best_compression(const double base_compression_ratio,
                                                   const double secondary_compression_ratio) const;

  /**
   * @brief Factory method to create a nvcomp_cpu_manager_adapter object for cpu-based compression.
   *
   * @param[in] compression_level Compression level (1-12)
   * @return A unique pointer to the nvcomp_cpu_manager_adapter object
   */
  std::unique_ptr<gqe::storage::nvcomp_cpu_manager_adapter> create_cpu_manager(
    int compression_level) const;

 public:
  /**
   * @brief Constructor for class
   *
   * @param[in] comp_format Compression format to use
   * @param[in] data_format Data format to use for compression configuration
   * @param[in] compression_chunk_size Chunk size to use for nvcomp
   * @param[in] stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator in nvcomp
   * @param[in] compression_ratio_threshold Compression ratio threshold to decide whether to
   * compress the columns or not.
   * @param[in] use_cpu_compression Whether to use CPU-based compression instead of GPU
   * @param[in] compression_level Compression level (1-12)
   * @param[in] column_name Name of the column being compressed
   * @param[in] cudf_type CUDF data type of the column being compressed
   */
  compression_manager(gqe::compression_format comp_format,
                      gqe::compression_format secondary_compression_format,
                      int compression_chunk_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr,
                      double compression_ratio_threshold,
                      double secondary_compression_ratio_threshold,
                      double secondary_compression_multiplier_threshold,
                      bool use_cpu_compression,
                      int compression_level,
                      std::string column_name   = "",
                      cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY});

  /**
   * @brief Function to perform compression on a column. Returns a unique pointer to the compressed
   * column
   * @param[in] uncompressed Data buffer to compress
   * @param[out] is_compressed Boolean flag that stores whether compression was viable or not
   * @param[out] compressed_size Total size of the compressed column in bytes
   * @param[out] uncompressed_size Total size of the uncompressed column in bytes
   * @param[in] supplied_stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   */
  std::unique_ptr<rmm::device_buffer> do_compress(rmm::device_buffer const* uncompressed,
                                                  bool& is_compressed,
                                                  int64_t& compressed_size,
                                                  int64_t& uncompressed_size,
                                                  rmm::cuda_stream_view supplied_stream,
                                                  rmm::device_async_resource_ref mr);

  /**
   * @brief Function to perform dcompression of a compressed column. Returns a unique pointer to the
   * decompressed column.
   *
   * @param[in] compressed Data buffer containing compressed data
   * @param[in] stream Stream to use for decompression
   * @param[in] mr Memory resource to use for decompression
   */
  std::unique_ptr<rmm::device_buffer> do_decompress(cudf::device_span<uint8_t const> compressed,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);

  /**
   * @brief Function to perform decompression of a batch of compressed buffers
   *
   * @param[in] device_decompressed_base_ptr Base decompression output buffer
   * @param[in] device_compressed Array of pointers to compressed buffers
   * @param[in] host_compressed A copy of device_compressed available to host
   * @param[in] batch_count the number of compressed batches to decompress
   * @param[in] is_secondary_compressed Boolean flag that stores whether secondary compression was
   * used
   * @param[in] cudf_type CUDF data type of the column being decompressed
   * @param[in] stream Stream to use for decompression
   * @param[in] mr Memory resource to use for decompression
   */
  void decompress_batch(uint8_t* const device_decompressed_base_ptr,
                        const uint8_t* const* device_compressed,
                        const uint8_t* const* host_compressed,
                        const size_t batch_count,
                        const bool is_secondary_compressed,
                        cudf::data_type cudf_type,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const;

  /**
   * @brief Function to perform compression of a batch of uncompressed buffers
   *
   * @param[in] device_uncompressed Array of pointers to uncompressed buffers
   * @param[out] is_compressed Boolean flag that stores whether compression was viable or not
   * @param[out] compressed_size Total size of the compressed column in bytes
   * @param[out] uncompressed_size Total size of the uncompressed column in bytes
   * @param[out] primary_compressed_size Total size of the primary compressed column in bytes
   * @param[out] secondary_compressed_size Total size of the secondary compressed column in bytes
   * @param[out] compressed_sizes Vector of sizes of the compressed buffers
   * @param[in] cudf_type CUDF data type of the column being compressed
   * @param[in] memory_kind Memory kind of the column being compressed
   * @param[out] is_secondary_compressed Boolean flag that stores whether secondary compression was
   * used
   * @param[in] try_secondary_compression Boolean flag that stores whether to try secondary
   * compression
   * @param[in] stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   */
  std::vector<std::unique_ptr<rmm::device_buffer>> compress_batch(
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
    rmm::device_async_resource_ref mr);

  // FIXME: We have two do_decompress functions. See issue:
  // https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/196
  std::unique_ptr<rmm::device_buffer> do_decompress(void const* compressed,
                                                    size_t compressed_size,
                                                    rmm::cuda_stream_view supplied_stream,
                                                    rmm::device_async_resource_ref mr);

  /**
   * @brief Function to fetch the compression format used by the compression manager.
   *
   * */
  gqe::compression_format get_comp_format() const;

  /**
   * @brief Function to fetch the chunk size used by the compression manager.
   * */
  int get_compression_chunk_size() const;

  /**
   * @brief Function to fetch the column name used by the compression manager.
   * */
  std::string get_column_name() const;

  /**
   * @brief Function to fetch the CUDF data type used by the compression manager.
   * */
  cudf::data_type get_cudf_type() const;

  /**
   * @brief Function to fetch the compression ratio threshold used by the compression manager.
   * */
  double get_compression_ratio_threshold() const;

  /**
   * @brief Function to fetch decompress backend.
   * */
  nvcompDecompressBackend_t get_decompress_backend() const;

  /**
   * @brief Function to fetch whether CPU compression is enabled.
   * */
  bool get_use_cpu_compression() const;

  /**
   * @brief Function to fetch the compression level.
   * */
  int get_compression_level() const;
};

// Function to map cudf::type_id to nvcomp_data_format
inline nvcompType_t get_optimal_nvcomp_data_type(cudf::type_id dtype)
{
  switch (dtype) {
    case cudf::type_id::INT8: return NVCOMP_TYPE_CHAR;
    case cudf::type_id::INT16: return NVCOMP_TYPE_SHORT;
    case cudf::type_id::INT32: return NVCOMP_TYPE_INT;
    case cudf::type_id::INT64: return NVCOMP_TYPE_LONGLONG;
    case cudf::type_id::UINT8: return NVCOMP_TYPE_UCHAR;
    case cudf::type_id::UINT16: return NVCOMP_TYPE_USHORT;
    case cudf::type_id::UINT32: return NVCOMP_TYPE_UINT;
    case cudf::type_id::UINT64: return NVCOMP_TYPE_ULONGLONG;
    case cudf::type_id::BOOL8: return NVCOMP_TYPE_CHAR;
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL32: return NVCOMP_TYPE_INT;
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return NVCOMP_TYPE_LONGLONG;
    default: return NVCOMP_TYPE_CHAR;
  }
}

// Function to map cudf::type_id to ideal compression_format.
// Mode 0: Optimizes for best compression ratio
// Mode 1: Optimizes for best decompression speed
inline void best_compression_config(cudf::type_id dtype,
                                    gqe::compression_format& comp_format,
                                    int mode)
{
  switch (dtype) {
    case cudf::type_id::INT8: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::INT16: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::INT32: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::INT64: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::UINT8: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::UINT16: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::UINT32: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::UINT64: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::LIST:
      if (mode == 0) {
        comp_format = gqe::compression_format::zstd;
      } else {
        comp_format = gqe::compression_format::ans;
      }
      break;
    case cudf::type_id::BOOL8:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL32: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: comp_format = gqe::compression_format::bitcomp; break;
    case cudf::type_id::STRING: comp_format = gqe::compression_format::ans; break;
    default: comp_format = gqe::compression_format::bitcomp; break;
  }
}

}  // namespace storage
}  // namespace gqe

namespace fmt {
/**
 * @brief Helper class to support logging of cudf::type_id.
 */
template <>
struct formatter<cudf::data_type> : formatter<std::string> {
  auto format(cudf::data_type type, format_context& ctx) const
  {
    return formatter<std::string>::format(cudf_type_to_string(type.id()), ctx);
  }

  std::string cudf_type_to_string(cudf::type_id type_id) const
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
      case cudf::type_id::STRUCT: return "STRUCT";
      case cudf::type_id::NUM_TYPE_IDS: return "NUM_TYPE_IDS";
      case cudf::type_id::DECIMAL32: return "DECIMAL32";
      case cudf::type_id::DECIMAL64: return "DECIMAL64";
      case cudf::type_id::DECIMAL128: return "DECIMAL128";
    }
    // Throw an exception so this will get fixed quickly when we add a new type.
    throw std::runtime_error("cudf::type_id type_id not supported for log formatting");
  };
};
}  // namespace fmt
