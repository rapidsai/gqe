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

#pragma once

#include <gqe/types.hpp>
#include <gqe/utility/logger.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace nvcomp;
using namespace gqe;

/**
 * @brief Class that contains functions for compression and decompression of columns in GQE.
 */
class compression_manager {
 private:
  gqe::compression_format _comp_format;
  nvcompType_t _data_type;
  int _compression_chunk_size;
  std::string _column_name;
  cudf::data_type _cudf_type;
  double _compression_ratio_threshold;

  void print_usage() const;
  /**
   * @brief Function that creates a nvcompManagerBase object based on the compression format.
   *
   * @param[in] supplied_stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator and deallocating scratch memory required for
   * nvcomp
   */
  std::unique_ptr<nvcompManagerBase> create_manager(rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr) const;

 public:
  /**
   * @brief Constructor for class
   *
   * @param[in] comp_format Compression format to use
   * @param[in] data_format Data format to use for compression configuration
   * @param[in] explicit_compression_chunk_size Chunk size to use for nvcomp
   * @param[in] stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator in nvcomp
   * @param[in] compression_ratio_threshold Compression ratio threshold to decide whether to
   * compress the columns or not.
   * @param[in] column_name Name of the column being compressed
   * @param[in] cudf_type CUDF data type of the column being compressed
   */
  compression_manager(gqe::compression_format comp_format,
                      nvcompType_t data_format,
                      int explicit_compression_chunk_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr,
                      double compression_ratio_threshold,
                      std::string column_name   = "",
                      cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY});

  /**
   * @brief Function to perform compression on a column. Returns a unique pointer to the compressed
   * column
   * @param[in] uncompressed Data buffer to compress
   * @param[out] compression_ratio Compression ratio of the compressed column
   * @param[out] is_compressed Boolean flag that stores whether compression was viable or not
   * @param[in] supplied_stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   */
  std::unique_ptr<rmm::device_buffer> do_compress(rmm::device_buffer const* uncompressed,
                                                  float& compression_ratio,
                                                  bool& is_compressed,
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
   * @param[in] stream Stream to use for decompression
   * @param[in] mr Memory resource to use for decompression
   */
  void decompress_batch(uint8_t* const device_decompressed_base_ptr,
                        const uint8_t* const* device_compressed,
                        const uint8_t* const* host_compressed,
                        const size_t batch_count,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

  /**
   * @brief Function to perform compression of a batch of uncompressed buffers
   *
   * @param[in] device_uncompressed Array of pointers to uncompressed buffers
   * @param[out] compression_ratio Compression ratio of the compressed column
   * @param[out] is_compressed Boolean flag that stores whether compression was viable or not
   * @param[out] compressed_size Total size of the compressed column
   * @param[out] compressed_sizes Vector of sizes of the compressed buffers
   * @param[in] stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   */
  std::vector<std::unique_ptr<rmm::device_buffer>> compress_batch(
    const std::vector<rmm::device_buffer>& device_uncompressed,
    float& compression_ratio,
    bool& is_compressed,
    size_t& compressed_size,
    std::vector<cudf::size_type>& compressed_sizes,
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
   * @brief Function to fetch the data type used by the compression manager.
   * */
  nvcompType_t get_data_type() const;

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

// Function to map cudf::type_id to ideal compression_format and nvcomp_data_format.
// Mode 0: Optimizes for best compression ratio
// Mode 1: Optimizes for best decompression speed
inline void best_compression_config(cudf::type_id dtype,
                                    gqe::compression_format& comp_format,
                                    nvcompType_t& nvcomp_data_format,
                                    int mode)
{
  switch (dtype) {
    case cudf::type_id::INT8:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_CHAR;
      break;
    case cudf::type_id::INT16:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_SHORT;
      break;
    case cudf::type_id::INT32:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_INT;
      break;
    case cudf::type_id::INT64:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_LONGLONG;
      break;
    case cudf::type_id::UINT8:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_UCHAR;
      break;
    case cudf::type_id::UINT16:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_USHORT;
      break;
    case cudf::type_id::UINT32:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_UINT;
      break;
    case cudf::type_id::UINT64:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_ULONGLONG;
      break;
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::LIST:
      if (mode == 0) {
        comp_format        = gqe::compression_format::zstd;
        nvcomp_data_format = NVCOMP_TYPE_CHAR;
      } else {
        comp_format        = gqe::compression_format::ans;
        nvcomp_data_format = NVCOMP_TYPE_FLOAT16;
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
    case cudf::type_id::DECIMAL32:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_INT;
      break;
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_LONGLONG;
      break;
    case cudf::type_id::STRING:
      comp_format        = gqe::compression_format::ans;
      nvcomp_data_format = NVCOMP_TYPE_CHAR;
      break;
    default:
      comp_format        = gqe::compression_format::bitcomp;
      nvcomp_data_format = NVCOMP_TYPE_INT;
      break;
  }
}
