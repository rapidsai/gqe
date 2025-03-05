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
  } else if (format_str == "lz4") {
    return gqe::compression_format::lz4;
  } else if (format_str == "snappy") {
    return gqe::compression_format::snappy;
  } else if (format_str == "gdeflate") {
    return gqe::compression_format::gdeflate;
  } else if (format_str == "deflate") {
    return gqe::compression_format::deflate;
  } else if (format_str == "cascaded") {
    return gqe::compression_format::cascaded;
  } else if (format_str == "zstd") {
    return gqe::compression_format::zstd;
  } else if (format_str == "gzip") {
    return gqe::compression_format::gzip;
  } else if (format_str == "bitcomp") {
    return gqe::compression_format::bitcomp;
  } else {
    throw std::logic_error("Unrecognized compression format");
  }
}

gqe::compression_format parse_nvcomp_compression_format(std::string const& env_variable,
                                                        gqe::compression_format const default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str) {
    return compression_format_from_str(val_str);
  } else {
    return default_value;
  }
}

nvcompType_t nvcomp_data_type_from_str(std::string const& data_format)
{
  if (data_format == "char") {
    return NVCOMP_TYPE_CHAR;
  } else if (data_format == "short") {
    return NVCOMP_TYPE_SHORT;
  } else if (data_format == "int") {
    return NVCOMP_TYPE_INT;
  } else if (data_format == "longlong") {
    return NVCOMP_TYPE_LONGLONG;
  } else if (data_format == "bits") {
    return NVCOMP_TYPE_BITS;
  } else {
    throw std::logic_error("Unrecognized data type format");
  }
}

nvcompType_t parse_nvcomp_data_type(std::string const& env_variable,
                                    nvcompType_t const default_value)
{
  auto const val_str = std::getenv(env_variable.c_str());

  if (val_str) {
    return nvcomp_data_type_from_str(val_str);
  } else {
    return default_value;
  }
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

    use_opt_type_for_single_char_col =
      parse_env_variable("GQE_USE_OPT_TYPE_FOR_SINGLE_CHAR_COL", use_opt_type_for_single_char_col);

    in_memory_table_compression_format = parse_nvcomp_compression_format(
      "GQE_IN_MEMORY_TABLE_COMP_FORMAT", in_memory_table_compression_format);

    in_memory_table_compression_data_type =
      parse_nvcomp_data_type("GQE_COMPRESSION_DATA_TYPE", in_memory_table_compression_data_type);

    compression_chunk_size =
      parse_nvcomp_chunk_size("GQE_COMPRESSION_CHUNK_SIZE", compression_chunk_size);

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
  }
}

}  // namespace gqe
