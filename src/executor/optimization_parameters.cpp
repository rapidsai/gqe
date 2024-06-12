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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/utility/logger.hpp>

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

gqe::compression_format compression_format_from_str(std::string const& format_str)
{
  if (format_str == "none") {
    return gqe::compression_format::none;
  } else if (format_str == "ans") {
    return gqe::compression_format::ans;
  } else {
    throw std::logic_error("Unrecognized compression format");
  }
}

gqe::compression_format parse_compression_format(std::string const& env_variable,
                                                 gqe::compression_format const default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str) {
    return compression_format_from_str(val_str);
  } else {
    return default_value;
  }
}

}  // namespace

namespace gqe {

optimization_parameters::optimization_parameters(bool only_defaults)
{
  if (!only_defaults) {
    max_num_workers = parse_env_variable("MAX_NUM_WORKERS", max_num_workers);

    max_num_partitions = parse_env_variable("MAX_NUM_PARTITIONS", max_num_partitions);

    join_use_hash_map_cache =
      parse_env_variable("GQE_JOIN_USE_HASH_MAP_CACHE", join_use_hash_map_cache);

    read_zero_copy_enable = parse_env_variable("GQE_READ_USE_ZERO_COPY", read_zero_copy_enable);

    use_customized_io = parse_env_variable("GQE_USE_CUSTOMIZED_IO", use_customized_io);

    io_bounce_buffer_size = parse_env_variable("GQE_IO_BOUNCE_BUFFER_SIZE", io_bounce_buffer_size);

    io_auxiliary_threads = parse_env_variable("GQE_IO_AUXILIARY_THREADS", io_auxiliary_threads);

    in_memory_table_compression_format = parse_compression_format(
      "GQE_IN_MEMORY_TABLE_COMP_FORMAT", in_memory_table_compression_format);
  }
}

}  // namespace gqe
