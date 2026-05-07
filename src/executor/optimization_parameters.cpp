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

#include <gqe/executor/optimization_parameters.hpp>

#include <gqe/utility/logger.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

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
    return gqe::compression_format_from_string(val_str);
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
    (val_str != nullptr) ? gqe::decompression_backend_from_string(val_str) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + gqe::decompression_backend_to_string(value));
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

optimization_parameters::optimization_parameters(bool only_defaults)
{
  if (!only_defaults) {
    max_num_workers = parse_env_variable("MAX_NUM_WORKERS", max_num_workers);

    max_num_partitions = parse_env_variable("MAX_NUM_PARTITIONS", max_num_partitions);

    initial_query_memory = parse_env_variable("GQE_INITIAL_QUERY_MEMORY", initial_query_memory);
    max_query_memory     = parse_env_variable("GQE_MAX_QUERY_MEMORY", max_query_memory);
    initial_task_manager_memory =
      parse_env_variable("GQE_INITIAL_TASK_MANAGER_MEMORY", initial_task_manager_memory);
    max_task_manager_memory =
      parse_env_variable("GQE_MAX_TASK_MANAGER_MEMORY", max_task_manager_memory);

    join_use_hash_map_cache =
      parse_env_variable("GQE_JOIN_USE_HASH_MAP_CACHE", join_use_hash_map_cache);

    join_use_unique_keys = parse_env_variable("GQE_JOIN_USE_UNIQUE_KEYS", join_use_unique_keys);

    join_use_perfect_hash = parse_env_variable("GQE_JOIN_USE_PERFECT_HASH", join_use_perfect_hash);

    aggregation_use_perfect_hash =
      parse_env_variable("GQE_AGGREGATION_USE_PERFECT_HASH", aggregation_use_perfect_hash);

    join_use_mark_join = parse_env_variable("GQE_JOIN_USE_MARK_JOIN", join_use_mark_join);

    read_zero_copy_enable = parse_env_variable("GQE_READ_USE_ZERO_COPY", read_zero_copy_enable);

    use_customized_io = parse_env_variable("GQE_USE_CUSTOMIZED_IO", use_customized_io);

    io_bounce_buffer_size = parse_env_variable("GQE_IO_BOUNCE_BUFFER_SIZE", io_bounce_buffer_size);

    io_auxiliary_threads = parse_env_variable("GQE_IO_AUXILIARY_THREADS", io_auxiliary_threads);

    use_opt_type_for_single_char_col =
      parse_env_variable("GQE_USE_OPT_TYPE_FOR_SINGLE_CHAR_COL", use_opt_type_for_single_char_col);

    use_in_memory_table_multigpu =
      parse_env_variable("GQE_IN_MEMORY_TABLE_USE_SHARED_MEMORY", use_in_memory_table_multigpu);

    io_block_size = parse_env_variable("GQE_IO_BLOCK_SIZE", io_block_size);

    auto io_engine_str = std::getenv("GQE_IO_ENGINE");
    if (io_engine_str) {
      if (strcmp_insensitive(io_engine_str, "IO_URING")) {
        io_engine = io_engine_type::io_uring;
      } else if (strcmp_insensitive(io_engine_str, "PSYNC")) {
        io_engine = io_engine_type::psync;
      } else if (strcmp_insensitive(io_engine_str, "AUTO") ||
                 strcmp_insensitive(io_engine_str, "AUTOMATIC")) {
        io_engine = io_engine_type::automatic;
      } else {
        throw std::runtime_error("Invalid value for environment variable: GQE_IO_ENGINE");
      }
    }

    io_pipelining = parse_env_variable("GQE_IO_PIPELINING_ENABLE", io_pipelining);

    io_alignment = parse_env_variable("GQE_IO_ALIGNMENT", io_alignment);

    use_overlap_mtx = parse_env_variable("GQE_USE_OVERLAP_MTX", use_overlap_mtx);

    use_partition_pruning = parse_env_variable("GQE_USE_PARTITION_PRUNING", use_partition_pruning);

    zone_map_partition_size =
      parse_env_variable("GQE_ZONE_MAP_PARTITION_SIZE", zone_map_partition_size);

    filter_use_like_shift_and =
      parse_env_variable("GQE_FILTER_USE_LIKE_SHIFT_AND", filter_use_like_shift_and);

    num_shuffle_partitions =
      parse_env_variable("GQE_NUM_SHUFFLE_PARTITIONS", num_shuffle_partitions);

    in_memory_table_compression_ratio_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_COMPRESSION_RATIO_THRESHOLD",
                         in_memory_table_compression_ratio_threshold);

    in_memory_table_compression_format = parse_nvcomp_compression_format(
      "GQE_IN_MEMORY_TABLE_COMPRESSION_FORMAT", in_memory_table_compression_format);

    in_memory_table_compression_chunk_size = parse_nvcomp_chunk_size(
      "GQE_IN_MEMORY_TABLE_COMPRESSION_CHUNK_SIZE", in_memory_table_compression_chunk_size);

    in_memory_table_secondary_compression_format = parse_nvcomp_compression_format(
      "GQE_IN_MEMORY_TABLE_SECONDARY_COMP_FORMAT", in_memory_table_secondary_compression_format);

    in_memory_table_secondary_compression_ratio_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_SECONDARY_COMPRESSION_RATIO_THRESHOLD",
                         in_memory_table_secondary_compression_ratio_threshold);

    in_memory_table_secondary_compression_multiplier_threshold =
      parse_env_variable("GQE_IN_MEMORY_TABLE_SECONDARY_COMPRESSION_MULTIPLIER_THRESHOLD",
                         in_memory_table_secondary_compression_multiplier_threshold);

    decompress_backend = parse_env_variable("GQE_DECOMPRESS_BACKEND", decompress_backend);

    use_cpu_compression = parse_env_variable("GQE_USE_CPU_COMPRESSION", use_cpu_compression);

    compression_level = parse_env_variable("GQE_COMPRESSION_LEVEL", compression_level);
  }
}

}  // namespace gqe
