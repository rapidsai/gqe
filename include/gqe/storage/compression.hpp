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

#pragma once

#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <memory>
#include <nvcomp.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <string>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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
  int _chunk_size;
  std::string _column_name;
  cudf::data_type _cudf_type;
  void print_usage() const;
  /**
   * @brief Function that creates a nvcompManagerBase object based on the compression format.
   *
   * @param[in] supplied_stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator and deallocating scratch memory required for
   * nvcomp
   */
  std::unique_ptr<nvcompManagerBase> create_manager(rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref& mr) const;

 public:
  /**
   * @brief Constructor for class
   *
   * @param[in] comp_format Compression format to use
   * @param[in] data_format Data format to use for compression configuration
   * @param[in] explicit_chunk_size Chunk size to use for nvcomp
   * @param[in] column_name Name of the column being compressed
   * @param[in] cudf_type CUDF data type of the column being compressed
   */
  compression_manager(gqe::compression_format comp_format,
                      nvcompType_t data_format,
                      int explicit_chunk_size,
                      std::string column_name   = "",
                      cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY});

  /**
   * @brief Function to perform compression on a column. Retuns a unique pointer to the compressed
   * column.
   *
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
   * @param[in] supplied_stream Stream to use for decompression
   * @param[in] mr Memory resource to use for compression
   */
  std::unique_ptr<rmm::device_buffer> do_decompress(rmm::device_buffer const* compressed,
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
  int get_chunk_size() const;

  /**
   * @brief Function to fetch the column name used by the compression manager.
   * */
  std::string get_column_name() const;

  /**
   * @brief Function to fetch the CUDF data type used by the compression manager.
   * */
  cudf::data_type get_cudf_type() const;
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