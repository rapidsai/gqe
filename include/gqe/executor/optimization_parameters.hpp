/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace gqe {

namespace detail {
/**
 * @brief Returns the default device memory pool size.
 *
 * @return 90% of free device memory.
 */
std::size_t default_device_memory_pool_size();
}  // namespace detail

/**
 * @brief Parameters indicating which optimizations are enabled and their settings.
 *
 * Implementation note: Add a comment for each parameter that indicates the task subtype it affects.
 * Each paramter must have a default setting.
 *
 * @note When adding a new field:
 *       - Serialization is automatic via Boost PFR; no changes needed to
 *         @ref gqe::rpc::optimization_parameters_to_session_options or
 *         @ref gqe::rpc::apply_session_option.
 *       - Add a corresponding @c parse_env_variable call in the @c only_defaults==false
 *         branch of @ref gqe::make_optimization_parameters.
 *       - Enum-typed fields additionally require a @c to_string / @c from_string
 *         overload pair in @c gqe/types.hpp.
 */
struct optimization_parameters {
  // IMPORTANT: This struct must remain a C++ aggregate (no user-provided constructors, no private
  // non-static data members, no base classes, no virtual functions). Boost PFR requires aggregate
  // types to reflect fields by name at compile time. Always construct instances via
  // make_optimization_parameters() rather than direct aggregate initialization.

  std::size_t max_num_workers = 1;  ///< Maximum number of worker threads per stage.
  int32_t max_num_partitions =
    8;  ///< The maximum number of read tasks that can be generated for a single table.
  std::string log_level = "info";  ///< Enable log messages for this level or higher.
  std::size_t initial_query_memory =
    10UL * 1024 * 1024 * 1024;  ///< Initial memory pool size for running queries (in bytes).
                                ///< Defaults to 10GB. In multi-GPU mode this parameter is ignored
                                ///< as symmetric memory region is not growable.
  std::optional<std::size_t> max_query_memory =
    std::nullopt;  ///< Maximum memory pool size for queries (in bytes). When nullopt, resolved to
                   ///< 90% of free device memory at allocation time. Can be set explicitly via
                   ///< GQE_MAX_QUERY_MEMORY env var.
  std::size_t initial_task_manager_memory =
    10UL * 1024 * 1024 *
    1024;  ///< Initial memory pool size for task manager memory resources (in bytes). This memory
           ///< is used across queries, e.g., for in-memory tables. Defaults to 10GB.
  std::size_t max_task_manager_memory =
    std::numeric_limits<std::size_t>::max();  ///< Maximum memory pool size for
                                              ///< task manager memory resources (in bytes).
  bool join_use_hash_map_cache =
    false;  ///< Allow multiple join tasks to reuse the same hash map. Enabling this option may
            ///< increase device-memory usage in some circumstances.
  bool join_use_unique_keys = true;  ///< Allow inner join to be optimized for unique build-side
                                     ///< keys with a hashset instead of a hash multiset. Joins on
                                     ///< non-unique keys will always deactivate this optimization.
  // TODO: issue!340 is tracking this, we should enable this only when we know it's good for
  // performance. After adding the new rule, we can set this to true.
  bool join_use_perfect_hash =
    false;  ///< Allow inner join to be optimized for unique build-side keys with perfect hashing.
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

  bool use_in_memory_table_multigpu =
    false;  ///< Use inter-process shared memory for the in-memory table.
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
  bool use_partition_pruning = false;  ///< Enable partition pruning for in-memory tables.
  cudf::size_type zone_map_partition_size = 100'000;  ///< Number of rows per zone map partition.
  bool filter_use_like_shift_and =
    true;  ///< Allow like filter to be optimized for using shift_and in the middle patterns.
           ///< Like_shift_and requires that the max length of middle patterns is <= 64 chars.
  // TODO: issue!325 is tracking this, after adding the new rule, we can set this to true.
  bool aggregation_use_perfect_hash =
    false;                             ///< Allow aggregation to be optimized with perfect hashing.
  int32_t num_shuffle_partitions = 2;  ///< Number of shuffle partitions for shuffle join.
  double in_memory_table_compression_ratio_threshold =
    1.0;  ///< Compression ratio threshold that determines whether to use the primary compression
          ///< algorithm to compress the columns. Below this threshold, the primary compression
          ///< algorithm will not be used. If the secondary compression is not used, the data will
          ///< be stored uncompressed. Higher ratio means better compression.
  compression_format in_memory_table_compression_format =
    compression_format::none;  ///< Compression format for the in-memory table.
  int in_memory_table_compression_chunk_size =
    1 << 16;  ///< Size of each chunk for nvcomp Compression
  compression_format in_memory_table_secondary_compression_format =
    compression_format::none;  ///< Compression format for the secondary compression algorithm for
                               ///< the in-memory table.
  double in_memory_table_secondary_compression_ratio_threshold =
    3.0;  ///< Compression ratio threshold to decide whether to use the secondary compression
          ///< algorithm to compress the columns. Below this threshold, the secondary compression
          ///< algorithm will not be considered. If the secondary compression is above this
          ///< threshold, and is at least secondary_compression_multiplier_threshold times better
          ///< than the primary compression algorithm, then the secondary compression algorithm will
          ///< be used.
  double in_memory_table_secondary_compression_multiplier_threshold =
    1.5;  ///< This threshold is used to determine whether the secondary compression algorithm is
          ///< "better enough" to be used instead of the primary compression algorithm. If the
          ///< secondary compression ratio is at least secondary_compression_multiplier_threshold
          ///< times better than the primary compression ratio, then the secondary compression
          ///< algorithm will be used.
  decompression_backend decompress_backend =
    decompression_backend::default_;  ///< nvCOMP decompress backend. Overridable via
                                      ///< GQE_DECOMPRESS_BACKEND.
  bool use_cpu_compression = false;   ///< Use CPU compression for in-memory table compression.
  int compression_level    = 10;      ///< LZ4 CPU compression level (1-12). Higher values provide
                                      ///< better compression but slower compression speed.
  double in_memory_dummy_copy_multiplier =
    1.0;  ///< Target total copy multiplier for memcpy in in_memory_read_task.
          ///< Emulates slower interconnects or networks, assuming the experiment setup
          ///< uses a high-speed interconnect. Must be >= 1.0. A value of 1.0 disables
          ///< dummy memcpy work.
};

/**
 * @brief Build an @c optimization_parameters instance with in-struct defaults and optional
 *        environment variable overrides.
 *
 * @param[in] only_defaults When true, skips environment variable loading. Useful in tests.
 */
optimization_parameters make_optimization_parameters(bool only_defaults = false);

}  // namespace gqe
