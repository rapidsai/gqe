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

#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
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
 * Any non-empty value is taken verbatim; an unset variable falls back to the default.
 */
template <>
std::string parse_env_variable<std::string>(std::string const& env_variable,
                                            const std::string default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  std::string value = (val_str != nullptr) ? std::string(val_str) : default_value;

  GQE_LOG_DEBUG(env_variable + " = " + value);
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
 * Acceptable values are "auto", "sm", and "de" (case-insensitive).
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

/**
 * Acceptable values are "double" and "decimal" (case-insensitive).
 */
template <>
gqe::decimal_type parse_env_variable<gqe::decimal_type>(std::string const& env_variable,
                                                        const gqe::decimal_type default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  gqe::decimal_type value =
    (val_str != nullptr) ? gqe::from_string<gqe::decimal_type>(val_str) : default_value;

  GQE_LOG_DEBUG(std::format("{} = {}", env_variable, gqe::to_string(value)));
  return value;
}

/**
 * Acceptable values are "legacy_task_graph" and "qep_task_graph" (case-insensitive).
 */
template <>
gqe::execution_backend_type parse_env_variable<gqe::execution_backend_type>(
  std::string const& env_variable, const gqe::execution_backend_type default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  gqe::execution_backend_type value =
    (val_str != nullptr) ? gqe::from_string<gqe::execution_backend_type>(val_str) : default_value;

  GQE_LOG_DEBUG(std::format("{} = {}", env_variable, gqe::to_string(value)));
  return value;
}

template <>
gqe::source_reader_type parse_env_variable<gqe::source_reader_type>(
  std::string const& env_variable, const gqe::source_reader_type default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());
  auto const value =
    val_str != nullptr ? gqe::from_string<gqe::source_reader_type>(val_str) : default_value;
  GQE_LOG_DEBUG(std::format("{} = {}", env_variable, gqe::to_string(value)));
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
    p.execution_backend = parse_env_variable("GQE_EXECUTION_BACKEND", p.execution_backend);

    p.num_workers = parse_env_variable("GQE_NUM_WORKERS", p.num_workers);

    p.num_partitions = parse_env_variable("GQE_NUM_PARTITIONS", p.num_partitions);

    p.initial_query_memory = parse_env_variable("GQE_INITIAL_QUERY_MEMORY", p.initial_query_memory);
    p.max_query_memory     = parse_env_variable("GQE_MAX_QUERY_MEMORY", p.max_query_memory);
    p.track_query_memory   = parse_env_variable("GQE_TRACK_QUERY_MEMORY", p.track_query_memory);
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

    p.use_late_materialization =
      parse_env_variable("GQE_USE_LATE_MATERIALIZATION", p.use_late_materialization);

    p.join_materialization_threshold =
      parse_env_variable("GQE_JOIN_MATERIALIZATION_THRESHOLD", p.join_materialization_threshold);

    p.read_use_zero_copy = parse_env_variable("GQE_READ_USE_ZERO_COPY", p.read_use_zero_copy);
    p.source_reader      = parse_env_variable("GQE_SOURCE_READER", p.source_reader);

    p.use_customized_io = parse_env_variable("GQE_USE_CUSTOMIZED_IO", p.use_customized_io);

    p.io_bounce_buffer_size =
      parse_env_variable("GQE_IO_BOUNCE_BUFFER_SIZE", p.io_bounce_buffer_size);

    p.io_auxiliary_threads = parse_env_variable("GQE_IO_AUXILIARY_THREADS", p.io_auxiliary_threads);

    p.use_opt_type_for_single_char_col = parse_env_variable("GQE_USE_OPT_TYPE_FOR_SINGLE_CHAR_COL",
                                                            p.use_opt_type_for_single_char_col);

    p.use_in_memory_table_multigpu =
      parse_env_variable("GQE_IN_MEMORY_TABLE_USE_SHARED_MEMORY", p.use_in_memory_table_multigpu);

    p.io_block_size = parse_env_variable("GQE_IO_BLOCK_SIZE", p.io_block_size);

    p.decimal_type = parse_env_variable("GQE_DECIMAL_TYPE", p.decimal_type);

    p.parquet_chunk_read_size =
      parse_env_variable("GQE_PARQUET_CHUNK_READ_SIZE", p.parquet_chunk_read_size);

    p.parquet_pass_read_size =
      parse_env_variable("GQE_PARQUET_PASS_READ_SIZE", p.parquet_pass_read_size);

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

    p.serialize_tables_to_disk =
      parse_env_variable("GQE_SERIALIZE_TABLES_TO_DISK", p.serialize_tables_to_disk);
    p.deserialize_tables_from_disk =
      parse_env_variable("GQE_DESERIALIZE_TABLES_FROM_DISK", p.deserialize_tables_from_disk);
    // `serialized_dataset_location` is intentionally session-only and has no environment variable.
    // It is set via SET by the client that also enables the flags above.

    p.filter_use_like_shift_and =
      parse_env_variable("GQE_FILTER_USE_LIKE_SHIFT_AND", p.filter_use_like_shift_and);

    p.num_shuffle_partitions =
      parse_env_variable("GQE_NUM_SHUFFLE_PARTITIONS", p.num_shuffle_partitions);

    p.use_ast_jit = parse_env_variable("GQE_USE_AST_JIT", p.use_ast_jit);

    p.compression_ratio_threshold =
      parse_env_variable("GQE_COMPRESSION_RATIO_THRESHOLD", p.compression_ratio_threshold);

    p.compression_format =
      parse_nvcomp_compression_format("GQE_COMPRESSION_FORMAT", p.compression_format);

    p.compression_chunk_size =
      parse_nvcomp_chunk_size("GQE_COMPRESSION_CHUNK_SIZE", p.compression_chunk_size);

    p.secondary_compression_format = parse_nvcomp_compression_format(
      "GQE_SECONDARY_COMPRESSION_FORMAT", p.secondary_compression_format);

    p.secondary_compression_ratio_threshold = parse_env_variable(
      "GQE_SECONDARY_COMPRESSION_RATIO_THRESHOLD", p.secondary_compression_ratio_threshold);

    p.secondary_compression_multiplier_threshold =
      parse_env_variable("GQE_SECONDARY_COMPRESSION_MULTIPLIER_THRESHOLD",
                         p.secondary_compression_multiplier_threshold);

    p.decompression_backend =
      parse_env_variable("GQE_DECOMPRESSION_BACKEND", p.decompression_backend);

    p.use_cpu_compression = parse_env_variable("GQE_USE_CPU_COMPRESSION", p.use_cpu_compression);

    p.compression_level = parse_env_variable("GQE_COMPRESSION_LEVEL", p.compression_level);

    p.in_memory_dummy_copy_multiplier =
      parse_env_variable("GQE_IN_MEMORY_DUMMY_COPY_MULTIPLIER", p.in_memory_dummy_copy_multiplier);

    p.sm_monitor_metric = parse_env_variable("GQE_SM_MONITOR_METRIC", p.sm_monitor_metric);

    p.sm_monitor_sample_interval_ns =
      parse_env_variable("GQE_SM_MONITOR_SAMPLE_INTERVAL_NS", p.sm_monitor_sample_interval_ns);

    p.sm_monitor_window_samples =
      parse_env_variable("GQE_SM_MONITOR_WINDOW_SAMPLES", p.sm_monitor_window_samples);

    p.sm_monitor_debug = parse_env_variable("GQE_SM_MONITOR_DEBUG", p.sm_monitor_debug);

    p.sm_monitor_trace_json_file =
      parse_env_variable("GQE_SM_MONITOR_TRACE_JSON_FILE", p.sm_monitor_trace_json_file);

    p.sm_monitor_trace_max_entries =
      parse_env_variable("GQE_SM_MONITOR_TRACE_MAX_ENTRIES", p.sm_monitor_trace_max_entries);

    // Morsel executor specific parameters
    p.morsel_max_rows = parse_env_variable("GQE_MORSEL_MAX_ROWS", p.morsel_max_rows);
  }
  validate_optimization_parameters(p);
  warn_for_ignored_optimization_parameters(p);
  return p;
}

void validate_optimization_parameters(optimization_parameters const& parameters)
{
  // The task-graph adapters can't concatenate references, and their consumers don't take deferred
  // columns.
  GQE_EXPECTS(parameters.execution_backend != execution_backend_type::qep_task_graph ||
                parameters.source_reader != source_reader_type::deferred,
              "optimization_parameters: source_reader=deferred is not supported by qep_task_graph",
              std::invalid_argument);

  GQE_EXPECTS(!parameters.morsel_max_rows.has_value() || *parameters.morsel_max_rows > 0,
              "optimization_parameters: morsel_max_rows must be positive when specified",
              std::invalid_argument);

  GQE_EXPECTS(
    std::isfinite(parameters.join_materialization_threshold) &&
      parameters.join_materialization_threshold >= 0.0 &&
      parameters.join_materialization_threshold <= 1.0,
    "optimization_parameters: join_materialization_threshold must be finite and in [0, 1]",
    std::invalid_argument);

  GQE_EXPECTS(!(parameters.serialize_tables_to_disk && parameters.deserialize_tables_from_disk),
              "optimization_parameters: serialize_tables_to_disk and deserialize_tables_from_disk "
              "are mutually exclusive due to undefined behavior",
              std::invalid_argument);
}

void warn_for_ignored_optimization_parameters(optimization_parameters const& parameters)
{
  if (parameters.execution_backend == execution_backend_type::legacy_task_graph &&
      parameters.source_reader != source_reader_type::oneshot) {
    GQE_LOG_WARN("source_reader={} only applies to QEP lowering; legacy_task_graph ignores it.",
                 to_string(parameters.source_reader));
  }
  if ((parameters.execution_backend == execution_backend_type::qep_task_graph ||
       parameters.execution_backend == execution_backend_type::legacy_task_graph) &&
      parameters.morsel_max_rows) {
    GQE_LOG_WARN(
      "morsel_max_rows is set to {}, but the {} executor does not preserve morsel boundaries; "
      "ignoring this setting.",
      parameters.morsel_max_rows.value(),
      to_string(parameters.execution_backend));
  }
}

}  // namespace gqe
