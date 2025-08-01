/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/utility/timer.hpp>

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gqe::storage {

struct table_with_metadata {
  std::unique_ptr<cudf::table> table;
  std::vector<cudf::size_type> rows_per_file;
};

/**
 * @brief Read Parquet files using cuDF's Parquet reader.
 *
 * If `column_names` is empty, the loaded table will be empty, but the rows-per-file statistics is
 * still valid.
 *
 * @param[in] file_paths List of Parquet files to read.
 * @param[in] column_names Columns to read.
 */
table_with_metadata read_parquet_cudf(std::vector<std::string> const& file_paths,
                                      std::vector<std::string> const& column_names);

/**
 * @brief Exception thrown when the input files have features not supported by the customized
 * Parquet reader.
 */
struct unsupported_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Read Parquet files using the customized Parquet reader.
 *
 * The reader supports the following Parquet types: INT32, INT64, FLOAT, DOUBLE and
 * FIXED_LEN_BYTE_ARRAY. BOOLEAN, INT96 and BYTE_ARRAY are not supported.
 *
 * The reader supports integers, decimals, and floating points as output types. All other
 * types, including strings, are not supported.
 *
 * The reader only supports flat data types. Nested data types are not supported.
 *
 * The reader only supports RLE/Bit-packing encoding (RLE = 3) for the definition level and plain
 * encoding (PLAIN = 0) for the values. All other encodings, including the dictionary encoding, are
 * not supported.
 *
 * The reader supports Snappy and uncompressed pages. All other compression formats are not
 * supported.
 *
 * If `columns` is empty, the loaded table will be empty, but the rows-per-file statistics is still
 * valid.
 *
 * @param[in] file_paths List of Parquet files to read.
 * @param[in] columns Columns to read.
 * @param[in] bounce_buffer Page-locked CPU memory used as a bounce buffer by the Parquet reader.
 * @param[in] bounce_buffer_size Size of `bounce_buffer`.
 * @param[in] num_auxiliary_threads Number of auxiliary host threads to launch for improving I/O
 * performance.
 * @param[in] block_size Size of the I/O block used by the Parquet reader.
 * @param[in] engine I/O engine to use for the Parquet reader.
 * @param[in] pipelining Enable I/O pipelining for the Parquet reader.
 * @param[in] alignment Alignment in bytes for the I/O buffer used by the Parquet reader.
 * @param[in] disk_timer Timer to record disk I/O bandwidth.
 * @param[in] h2d_timer Timer to record host-to-device memory copy bandwidth.
 * @param[in] decomp_timer Timer to record decompression bandwidth.
 * @param[in] decode_timer Timer to record decoding bandwidth.
 * @param[in] stream CUDA stream to operate on.
 * @param[in] mr Memory resource to use for allocating the result table and temporary device
 * buffers.
 */
table_with_metadata read_parquet_custom(
  std::vector<std::string> file_paths,
  std::vector<std::string> columns,
  void* bounce_buffer,
  int64_t bounce_buffer_size,
  std::size_t num_auxiliary_threads,
  std::size_t block_size,
  io_engine_type engine,
  bool pipelining,
  std::size_t alignment,
  gqe::utility::bandwidth_timer& disk_timer,
  gqe::utility::bandwidth_timer& h2d_timer,
  gqe::utility::bandwidth_timer& decomp_timer,
  gqe::utility::bandwidth_timer& decode_timer,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace gqe::storage
