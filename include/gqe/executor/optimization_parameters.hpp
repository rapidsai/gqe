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

#include <gqe/types.hpp>
#include <nvcomp.hpp>

#include <cstdint>
#include <string>

namespace gqe {

/**
 * @brief Parameters indicating which optimizations are enabled and their settings.
 *
 * Implementation note: Add a comment for each parameter that indicates the task subtype it affects.
 * Each paramter must have a default setting.
 */
struct optimization_parameters {
  /**
   * @brief Build a new instance with default parameters and configured parameters.
   *
   * @note Currently collects environment variables to set the optimization parameters. In future,
   * this will be changed to load from a file.
   *
   * @param[in] only_defaults Disables loading of parameters from files or environment variables.
   * Used for testing.
   */
  explicit optimization_parameters(bool only_defaults = false);

  std::size_t max_num_workers = 1;  ///< Maximum number of worker threads per stage.
  int32_t max_num_partitions =
    8;  ///< The maximum number of read tasks that can be generated for a single table.
  std::string log_level = "info";  ///< Enable log messages for this level or higher.
  bool join_use_hash_map_cache =
    false;  ///< Allow multiple join tasks to reuse the same hash map. Enabling this option may
            ///< increase device-memory usage in some circumstances.
  bool join_use_unique_keys = true;  ///< Allow inner join to be optimized for unique build-side
                                     ///< keys with a hashset instead of a hash multiset. Joins on
                                     ///< non-unique keys will always deactivate this optimization.
  bool join_use_perfect_hash =
    true;  ///< Allow inner join to be optimized for unique build-side keys with perfect hashing.
           ///< Perfect hashing requires that both sides have no nulls.
  bool join_use_mark_join =
    true;  // Allow semi and anti left joins to be optimized with the mark join algorithm when the
           // LHS input is smaller than the RHS. If disabled, falls back to cuDF.
  bool read_zero_copy_enable =
    true;  ///< Enable zero-copy reads for in-memory table. When disabled, read tasks copy input
           ///< data to a temporary output buffer.
  bool use_customized_io = false;  ///< Use the customized Parquet reader if supported.
  int32_t io_bounce_buffer_size =
    4;  ///< Size in GB per worker of the page-locked CPU memory bounce
        ///< buffer used by the customized Parquet reader. Default to 4 (GB).
  std::size_t io_auxiliary_threads =
    8;  ///< Number of auxiliary threads per worker launched by the customized Parquet reader.
  bool use_opt_type_for_single_char_col =
    true;  ///< Use optimized char type instead of string type to store single-char columns
           ///< Char type requires less memory to store and kernels on this type
           ///< are more efficient (ex. gather), but it cannot perform any string
           ///< operations (ex. `like` , `contains`).
  compression_format in_memory_table_compression_format =
    compression_format::none;  ///< Compression format for the in-memory table.
  nvcompType_t in_memory_table_compression_data_type =
    NVCOMP_TYPE_CHAR;                    ///< Determines how input data is viewed as for compression
  int compression_chunk_size = 1 << 16;  ///< Size of each chunk for nvcomp Compression
  std::size_t io_block_size =
    2048;  ///< Size in KiB of the I/O block used by the customized Parquet reader.
  io_engine_type io_engine =
    io_engine_type::io_uring;       ///< I/O engine to use for the customized Parquet reader:
                                    ///< IO_URING or PSYNC or AUTO.
  bool io_pipelining       = true;  ///< Enable I/O pipelining for the customized Parquet reader.
  std::size_t io_alignment = 4096;  ///< Alignment in bytes for the I/O buffer used by the
                                    ///< customized Parquet reader for io_uring.
  bool use_overlap_mtx =
    true;  ///< Enable better overlap and pipelining by using locks in memory read task
  bool use_partition_pruning          = false;  ///< Enable partition pruning for in-memory tables.
  std::size_t zone_map_partition_size = 100'000;  ///< Number of rows per zone map partition.
  bool filter_use_like_shift_and =
    true;  ///< Allow like filter to be optimized for using shift_and in the middle patterns.
           ///< Like_shift_and requires that the max length of middle patterns is <= 64 chars.
  bool aggregation_use_perfect_hash =
    true;                              ///< Allow aggregation to be optimized with perfect hashing.
  int32_t num_shuffle_partitions = 2;  ///< Number of shuffle partitions for shuffle join.
};

}  // namespace gqe
