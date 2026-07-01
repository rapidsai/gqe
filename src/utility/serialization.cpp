/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/serialization.hpp>

#include <gqe/types.hpp>

#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace gqe::utility {

namespace {

[[nodiscard]] std::string to_ascii_upper(std::string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

/** Directory segment for primary compression (see header: LZ4 + CPU adds level suffix). */
[[nodiscard]] std::string compression_format_directory_name(compression_format fmt,
                                                            bool use_cpu_compression,
                                                            int compression_level)
{
  std::string_view basename;
  switch (fmt) {
    case compression_format::none: basename = "none"; break;
    case compression_format::ans: basename = "ANS"; break;
    case compression_format::lz4: basename = "LZ4"; break;
    case compression_format::snappy: basename = "Snappy"; break;
    case compression_format::gdeflate: basename = "GDeflate"; break;
    case compression_format::deflate: basename = "Deflate"; break;
    case compression_format::cascaded: basename = "Cascaded"; break;
    case compression_format::zstd: basename = "ZSTD"; break;
    case compression_format::gzip: basename = "GZIP"; break;
    case compression_format::bitcomp: basename = "Bitcomp"; break;
    default: basename = "UNKNOWN"; break;
  }

  if (use_cpu_compression && fmt == compression_format::lz4) {
    std::string out = to_ascii_upper(std::string{basename});
    out += "_CPU_";
    out += std::to_string(compression_level);
    return out;
  }
  return std::string{basename};
}

[[nodiscard]] std::string chunk_size_directory_name(int chunk_size_bytes)
{
  if (chunk_size_bytes <= 0) { return "0B_chunks"; }

  auto const n = static_cast<std::size_t>(chunk_size_bytes);
  constexpr std::size_t kb{1024};
  constexpr std::size_t mb{kb * kb};

  if (n % mb == 0) { return std::to_string(n / mb) + "MB_chunks"; }
  if (n % kb == 0) { return std::to_string(n / kb) + "KB_chunks"; }
  return std::to_string(n) + "B_chunks";
}

}  // namespace

std::filesystem::path serialized_table_root(std::filesystem::path table_data_directory,
                                            optimization_parameters const& params)
{
  return std::move(table_data_directory) / "serialized_data" /
         ("max_partitions-" + std::to_string(params.max_num_partitions)) /
         ("zmps-" + std::to_string(params.zone_map_partition_size));
}

std::filesystem::path serialized_row_group_column_root(
  std::filesystem::path table_serialized_data_root,
  size_t row_group_index,
  optimization_parameters const& params)
{
  return std::move(table_serialized_data_root) / ("rg-" + std::to_string(row_group_index)) /
         compression_format_directory_name(params.in_memory_table_compression_format,
                                           params.use_cpu_compression,
                                           params.compression_level) /
         chunk_size_directory_name(params.in_memory_table_compression_chunk_size);
}

std::filesystem::path serialized_row_group_zone_maps_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index)
{
  return std::move(table_serialized_data_root) / ("rg-" + std::to_string(row_group_index)) /
         "zone_maps";
}

}  // namespace gqe::utility
