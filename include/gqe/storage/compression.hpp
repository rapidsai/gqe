/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda/stream>
#include <cudf/types.hpp>
#include <nvcomp/shared_types.h>
#include <rmm/device_buffer.hpp>

#include <vector>

namespace gqe {
namespace compression {

/**
 * @brief Configuration used by compression functions.
 */
struct configuration {
  gqe::compression_format primary_compression_format;
  gqe::compression_format secondary_compression_format;
  int compression_chunk_size;
  double compression_ratio_threshold;
  double secondary_compression_ratio_threshold;
  double secondary_compression_multiplier_threshold;
  bool use_cpu_compression;
  int compression_level;
  gqe::decompression_backend decompress_backend{gqe::decompression_backend::auto_};
};

/**
 * @brief Function to perform decompression of a batch of compressed buffers
 *
 * @param[in] config Compression configuration
 * @param[in] data_type nvCOMP data type of the column being decompressed
 * @param[in] device_decompressed_base_ptr Base decompression output buffer
 * @param[in] device_compressed Array of pointers to compressed buffers
 * @param[in] host_compressed A copy of device_compressed available to host
 * @param[in] batch_count the number of compressed batches to decompress
 * @param[in] compression_format The compression format used for the compressed buffers.
 * @param[in] stream Stream to use for decompression
 * @param[in] mr Memory resource to use for decompression
 *
 * This function synchronizes @p stream before returning because nvCOMP may access request-local
 * decompression metadata until the asynchronous decompression completes.
 */
void decompress_batch(configuration const& config,
                      nvcompType_t data_type,
                      uint8_t* const device_decompressed_base_ptr,
                      const uint8_t* const* device_compressed,
                      const uint8_t* const* host_compressed,
                      const size_t batch_count,
                      gqe::compression_format compression_format,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr);

/**
 * @brief Function to perform compression of a batch of uncompressed buffers
 *
 * @param[in] config Compression configuration
 * @param[in] data_type nvCOMP data type of the column being compressed
 * @param[in] device_uncompressed Array of pointers to uncompressed buffers
 * @param[out] selected_format The selected compression format. compression_format::none
 * indicates uncompressed output.
 * @param[out] compressed_size Total size of the compressed column in bytes
 * @param[out] uncompressed_size Total size of the uncompressed column in bytes
 * @param[out] primary_compressed_size Total size of the primary compressed column in bytes
 * @param[out] secondary_compressed_size Total size of the secondary compressed column in bytes
 * @param[out] compressed_sizes Vector of sizes of the compressed buffers
 * @param[in] memory_kind Memory kind of the column being compressed
 * @param[in] try_secondary_compression Boolean flag that stores whether to try secondary
 * compression
 * @param[in] stream Stream to use for compression
 * @param[in] mr Memory resource to use for compression
 */
std::vector<rmm::device_buffer> compress_batch(
  configuration const& config,
  nvcompType_t data_type,
  std::vector<rmm::device_buffer>&& device_uncompressed,
  gqe::compression_format& selected_format,
  size_t& compressed_size,
  size_t& uncompressed_size,
  size_t& primary_compressed_size,
  size_t& secondary_compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  memory_kind::type memory_kind,
  bool try_secondary_compression,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

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

}  // namespace compression
}  // namespace gqe
