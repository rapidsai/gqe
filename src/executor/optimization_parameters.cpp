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

#include <gqe/executor/optimization_parameters.hpp>

#include <gqe/utility/logger.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

// Return true if two strings compare equal case insensitive
bool strcmp_insensitive(std::string const& first, std::string const& second)
{
  return std::equal(first.begin(),
                    first.end(),
                    second.begin(),
                    second.end(),
                    [](char char_first, char char_second) {
                      return std::tolower(char_first) == std::tolower(char_second);
                    });
}

/**
 * @brief Parse an environment variable.
 *
 * @param[in] env_variable Name of the environment variable.
 * @param[in] default_value Default value if the environment variable is not found.
 *
 * @return The parsed value.
 */
template <typename T>
T parse_env_variable(std::string const& env_variable, const T default_value)
{
  static_assert("No default implementation");
}

/**
 * Acceptable values are on/off, true/false and yes/no.
 */
template <>
bool parse_env_variable<bool>(std::string const& env_variable, const bool default_value)
{
  bool value          = default_value;
  auto const env_char = std::getenv(env_variable.c_str());
  if (env_char) {
    std::string env_string(env_char);
    if (strcmp_insensitive(env_string, "true") || strcmp_insensitive(env_string, "on") ||
        strcmp_insensitive(env_string, "yes")) {
      value = true;
    } else if (strcmp_insensitive(env_string, "false") || strcmp_insensitive(env_string, "off") ||
               strcmp_insensitive(env_string, "no")) {
      value = false;
    } else {
      throw std::runtime_error("Invalid value for environment variable: " + env_variable);
    }
  }

  GQE_LOG_DEBUG(env_variable + " = " + std::to_string(value));
  return value;
}

/**
 * Acceptable values are numbers in the range of the specified integer type.
 */
template <>
int parse_env_variable<int>(std::string const& env_variable, const int default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  int value = (val_str != nullptr) ? std::stoi(val_str, nullptr, 10) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + std::to_string(value));
  return value;
}

/**
 * Acceptable values are numbers in the range of the specified integer type.
 */
template <>
unsigned long parse_env_variable<unsigned long>(std::string const& env_variable,
                                                const unsigned long default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  unsigned long value = (val_str != nullptr) ? std::strtoul(val_str, nullptr, 10) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + std::to_string(value));
  return value;
}

/**
 * Acceptable values are numbers in the range of the specified double type.
 */
template <>
double parse_env_variable<double>(std::string const& env_variable, const double default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  double value = (val_str != nullptr) ? std::strtod(val_str, nullptr) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + std::to_string(value));
  return value;
}

/**
 * @brief Parse an optional value from an environment variable.
 *
 * Delegates to the existing parse_env_variable<T> specialization for actual parsing.
 * Returns the parsed value if the env var is set, otherwise returns the default_value.
 */
template <typename T>
std::optional<T> parse_env_variable(std::string const& env_variable,
                                    const std::optional<T> default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str != nullptr) { return parse_env_variable<T>(env_variable, T{}); }
  return default_value;
}

gqe::compression_format parse_nvcomp_compression_format(std::string const& env_variable,
                                                        gqe::compression_format const default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str) {
    return gqe::from_string<gqe::compression_format>(val_str);
  } else {
    return default_value;
  }
}

/**
 * Acceptable values are "default", "sm", and "de" (case-insensitive).
 */
template <>
gqe::decompression_backend parse_env_variable<gqe::decompression_backend>(
  std::string const& env_variable, const gqe::decompression_backend default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  gqe::decompression_backend value =
    (val_str != nullptr) ? gqe::from_string<gqe::decompression_backend>(val_str) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + gqe::to_string(value));
  return value;
}

int parse_nvcomp_chunk_size(std::string const& env_variable, int const default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str) {
    int n = std::stoi(val_str);
    return 1 << n;
  } else {
    return default_value;
  }
}

}  // namespace

namespace gqe {

namespace detail {

std::size_t default_device_memory_pool_size() { return rmm::percent_of_free_device_memory(90); }

}  // namespace detail

optimization_parameters make_optimization_parameters(bool only_defaults)
{
  optimization_parameters p;
  if (!only_defaults) {
    p.max_num_workers = parse_env_variable("MAX_NUM_WORKERS", p.max_num_workers);

    p.max_num_partitions = parse_env_variable("MAX_NUM_PARTITIONS", p.max_num_partitions);

    p.initial_query_memory = parse_env_variable("GQE_INITIAL_QUERY_MEMORY", p.initial_query_memory);
    p.max_query_memory     = parse_env_variable("GQE_MAX_QUERY_MEMORY", p.max_query_memory);
    p.initial_task_manager_memory =
      parse_env_variable("GQE_INITIAL_TASK_MANAGER_MEMORY", p.initial_task_manager_memory);
    p.max_task_manager_memory =
      parse_env_variable("GQE_MAX_TASK_MANAGER_MEMORY", p.max_task_manager_memory);

    p.join_use_hash_map_cache =
      parse_env_variable("GQE_JOIN_USE_HASH_MAP_CACHE", p.join_use_hash_map_cache);

    p.join_use_unique_keys = parse_env_variable("GQE_JOIN_USE_UNIQUE_KEYS", p.join_use_unique_keys);

    p.join_use_perfect_hash =
      parse_env_variable("GQE_JOIN_USE_PERFECT_HASH", p.join_use_perfect_hash);

    p.aggregation_use_perfect_hash =
      parse_env_variable("GQE_AGGREGATION_USE_PERFECT_HASH", p.aggregation_use_perfect_hash);

    p.join_use_mark_join = parse_env_variable("GQE_JOIN_USE_MARK_JOIN", p.join_use_mark_join);

    p.read_zero_copy_enable = parse_env_variable("GQE_READ_USE_ZERO_COPY", p.read_zero_copy_enable);

    p.use_customized_io = parse_env_variable("GQE_USE_CUSTOMIZED_IO", p.use_customized_io);

    p.io_bounce_buffer_size =
      parse_env_variable("GQE_IO_BOUNCE_BUFFER_SIZE", p.io_bounce_buffer_size);

    p.io_auxiliary_threads = parse_env_variable("GQE_IO_AUXILIARY_THREADS", p.io_auxiliary_threads);

    p.use_opt_type_for_single_char_col = parse_env_variable("GQE_USE_OPT_TYPE_FOR_SINGLE_CHAR_COL",
                                                            p.use_opt_type_for_single_char_col);

    p.use_in_memory_table_multigpu =
      parse_env_variable("GQE_IN_MEMORY_TABLE_USE_SHARED_MEMORY", p.use_in_memory_table_multigpu);

    p.io_block_size = parse_env_variable("GQE_IO_BLOCK_SIZE", p.io_block_size);

    auto io_engine_str = std::getenv("GQE_IO_ENGINE");
    if (io_engine_str) {
      if (strcmp_insensitive(io_engine_str, "IO_URING")) {
        p.io_engine = io_engine_type::io_uring;
      } else if (strcmp_insensitive(io_engine_str, "PSYNC")) {
        p.io_engine = io_engine_type::psync;
      } else if (strcmp_insensitive(io_engine_str, "AUTO") ||
                 strcmp_insensitive(io_engine_str, "AUTOMATIC")) {
        p.io_engine = io_engine_type::automatic;
      } else {
        throw std::runtime_error("Invalid value for environment variable: GQE_IO_ENGINE");
      }
    }

    p.io_pipelining = parse_env_variable("GQE_IO_PIPELINING_ENABLE", p.io_pipelining);

    p.io_alignment = parse_env_variable("GQE_IO_ALIGNMENT", p.io_alignment);

    p.use_overlap_mtx = parse_env_variable("GQE_USE_OVERLAP_MTX", p.use_overlap_mtx);

    p.use_partition_pruning =
      parse_env_variable("GQE_USE_PARTITION_PRUNING", p.use_partition_pruning);

    p.zone_map_partition_size =
      parse_env_variable("GQE_ZONE_MAP_PARTITION_SIZE", p.zone_map_partition_size);

    p.filter_use_like_shift_and =
      parse_env_variable("GQE_FILTER_USE_LIKE_SHIFT_AND", p.filter_use_like_shift_and);

    p.num_shuffle_partitions =
      parse_env_variable("GQE_NUM_SHUFFLE_PARTITIONS", p.num_shuffle_partitions);

    p.in_memory_table_compression_ratio_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_COMPRESSION_RATIO_THRESHOLD",
                         p.in_memory_table_compression_ratio_threshold);

    p.in_memory_table_compression_format = parse_nvcomp_compression_format(
      "GQE_IN_MEMORY_TABLE_COMPRESSION_FORMAT", p.in_memory_table_compression_format);

    p.in_memory_table_compression_chunk_size = parse_nvcomp_chunk_size(
      "GQE_IN_MEMORY_TABLE_COMPRESSION_CHUNK_SIZE", p.in_memory_table_compression_chunk_size);

    p.in_memory_table_secondary_compression_format = parse_nvcomp_compression_format(
      "GQE_IN_MEMORY_TABLE_SECONDARY_COMP_FORMAT", p.in_memory_table_secondary_compression_format);

    p.in_memory_table_secondary_compression_ratio_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_SECONDARY_COMPRESSION_RATIO_THRESHOLD",
                         p.in_memory_table_secondary_compression_ratio_threshold);

    p.in_memory_table_secondary_compression_multiplier_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_SECONDARY_COMPRESSION_MULTIPLIER_THRESHOLD",
                         p.in_memory_table_secondary_compression_multiplier_threshold);

    p.decompress_backend = parse_env_variable("GQE_DECOMPRESS_BACKEND", p.decompress_backend);

    p.use_cpu_compression = parse_env_variable("GQE_USE_CPU_COMPRESSION", p.use_cpu_compression);

    p.compression_level = parse_env_variable("GQE_COMPRESSION_LEVEL", p.compression_level);

    p.in_memory_dummy_copy_multiplier =
      parse_env_variable("GQE_IN_MEMORY_DUMMY_COPY_MULTIPLIER", p.in_memory_dummy_copy_multiplier);
  }
  return p;
}

}  // namespace gqe
