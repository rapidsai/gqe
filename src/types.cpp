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

#include <gqe/types.hpp>

#include <gqe/device_properties.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/linux.hpp>

#include <boost/container_hash/hash.hpp>

#include <unistd.h>  // sysconf

#include <format>

namespace gqe {

std::string to_string(io_engine_type e) noexcept
{
  switch (e) {
    case io_engine_type::io_uring: return "io_uring";
    case io_engine_type::psync: return "psync";
    case io_engine_type::automatic: return "automatic";
    default: return "unknown";
  }
}

std::string to_string(compression_format e) noexcept
{
  switch (e) {
    case compression_format::none: return "none";
    case compression_format::lz4: return "lz4";
    case compression_format::snappy: return "snappy";
    case compression_format::ans: return "ans";
    case compression_format::cascaded: return "cascaded";
    case compression_format::bitpacking: return "bitpacking";
    case compression_format::gdeflate: return "gdeflate";
    case compression_format::deflate: return "deflate";
    case compression_format::zstd: return "zstd";
    case compression_format::gzip: return "gzip";
    case compression_format::bitcomp: return "bitcomp";
    default: return "unknown";
  }
}

std::string to_string(decompression_backend e) noexcept
{
  switch (e) {
    case decompression_backend::auto_: return "auto";
    case decompression_backend::sm: return "sm";
    case decompression_backend::de: return "de";
    default: return "unknown";
  }
}

std::string to_string(decimal_type e) noexcept
{
  switch (e) {
    case decimal_type::double_: return "double";
    case decimal_type::decimal: return "decimal";
    default: return "unknown";
  }
}

std::string to_string(execution_backend_type e) noexcept
{
  switch (e) {
    case execution_backend_type::legacy_task_graph: return "legacy_task_graph";
    case execution_backend_type::qep_task_graph: return "qep_task_graph";
    default: return "unknown";
  }
}

std::string to_string(source_reader_type e) noexcept
{
  switch (e) {
    case source_reader_type::oneshot: return "oneshot";
    case source_reader_type::chunked: return "chunked";
    case source_reader_type::deferred: return "deferred";
    default: return "unknown";
  }
}

template <>
io_engine_type from_string<io_engine_type>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "io_uring") return io_engine_type::io_uring;
  if (n == "psync") return io_engine_type::psync;
  if (n == "automatic" || n == "auto") return io_engine_type::automatic;
  throw std::invalid_argument(std::format("Unrecognized io_engine value: {}", s));
}

template <>
compression_format from_string<compression_format>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "none") return compression_format::none;
  if (n == "ans") return compression_format::ans;
  if (n == "lz4") return compression_format::lz4;
  if (n == "snappy") return compression_format::snappy;
  if (n == "gdeflate") return compression_format::gdeflate;
  if (n == "deflate") return compression_format::deflate;
  if (n == "cascaded") return compression_format::cascaded;
  if (n == "bitpacking") return compression_format::bitpacking;
  if (n == "zstd") return compression_format::zstd;
  if (n == "gzip") return compression_format::gzip;
  if (n == "bitcomp") return compression_format::bitcomp;
  throw std::invalid_argument(std::format("Unrecognized compression format: {}", s));
}

template <>
decompression_backend from_string<decompression_backend>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "auto") return decompression_backend::auto_;
  if (n == "sm") return decompression_backend::sm;
  if (n == "de") return decompression_backend::de;
  throw std::invalid_argument(std::format("Unrecognized decompress backend: {}", s));
}

template <>
decimal_type from_string<decimal_type>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "double") return decimal_type::double_;
  if (n == "decimal") return decimal_type::decimal;
  throw std::invalid_argument(std::format("Unrecognized decimal type: {}", s));
}

template <>
execution_backend_type from_string<execution_backend_type>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "legacy_task_graph") return execution_backend_type::legacy_task_graph;
  if (n == "qep_task_graph") return execution_backend_type::qep_task_graph;
  throw std::invalid_argument(std::format("Unrecognized execution backend: {}", s));
}

template <>
source_reader_type from_string<source_reader_type>(std::string_view s)
{
  auto n = utility::to_lower(s);
  if (n == "oneshot") return source_reader_type::oneshot;
  if (n == "chunked") return source_reader_type::chunked;
  if (n == "deferred") return source_reader_type::deferred;
  throw std::invalid_argument(std::format("Unrecognized source reader: {}", s));
}

scoped_cpu_affinity::scoped_cpu_affinity()
{
  utility::get_thread_affinity(_prev_mask);
  utility::set_thread_affinity_fullmask();
}

scoped_cpu_affinity::~scoped_cpu_affinity() { utility::set_thread_affinity(_prev_mask); }

page_kind::page_kind() {}

page_kind::page_kind(type type) : _type(type) {}

page_kind::type page_kind::value() const noexcept { return _type; };

std::size_t page_kind::size() const noexcept
{
  switch (_type) {
    case type::system_default:
    case type::small:
    case type::transparent_huge: return ::sysconf(_SC_PAGESIZE);
    case type::huge2mb: return std::size_t{1} << 21;
    case type::huge1gb: return std::size_t{1} << 30;
  };

  // Should never be reached
  return 0;
}

bool memory_kind::is_gpu_accessible(memory_kind::type type)
{
  // Check if zero copy is legal
  auto check_pageable_access =
    device_properties::instance().get<device_properties::property::pageableMemoryAccess>();

  return std::visit(
    utility::overloaded{
      [&](memory_kind::system) { return check_pageable_access; },
      [&](memory_kind::numa) { return check_pageable_access; },
      [&](memory_kind::pinned) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [&](memory_kind::numa_pinned) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [](memory_kind::device) { return true; },
      [&](memory_kind::managed) -> bool {
        return device_properties::instance().get<device_properties::property::managedMemory>();
      },
      [&](memory_kind::shared_segment) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [&](memory_kind::numa_pool) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [&](memory_kind::shared_numa_pool) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      }},
    type);
}

bool memory_kind::is_cpu_accessible(memory_kind::type type)
{
  return std::visit(utility::overloaded{
                      [&](memory_kind::system) { return true; },
                      [&](memory_kind::numa) { return true; },
                      [&](memory_kind::pinned) { return true; },
                      [&](memory_kind::numa_pinned) { return true; },
                      [&](memory_kind::managed) { return true; },
                      [&](memory_kind::shared_segment) { return true; },
                      [&](memory_kind::numa_pool) { return true; },
                      [&](memory_kind::shared_numa_pool) { return true; },
                      [&](memory_kind::device) { return false; },
                    },
                    type);
}

bool storage_kind::is_file_backed(storage_kind::type type)
{
  // parquet_file is currently the only file-backed kind; every other alternative is memory-backed.
  return std::holds_alternative<storage_kind::parquet_file>(type);
}

bool memory_kind::system::operator==(system const&) const = default;

memory_kind::numa::numa(cpu_set node_set, gqe::page_kind::type pk)
  : numa_node_set(std::move(node_set)), page_kind(pk)
{
}

memory_kind::numa::numa(gqe::page_kind::type pk)
  : numa_node_set(device_properties::instance().get<device_properties::property::memoryAffinity>()),
    page_kind(pk)
{
}

bool memory_kind::numa::operator==(numa const& other) const
{
  return numa_node_set == other.numa_node_set && page_kind == other.page_kind;
}

bool memory_kind::pinned::operator==(pinned const& other) const = default;

memory_kind::numa_pinned::numa_pinned(cpu_set node_set, gqe::page_kind::type pk)
  : numa_node_set(std::move(node_set)), page_kind(pk)
{
}

memory_kind::numa_pinned::numa_pinned(gqe::page_kind::type pk)
  : numa_node_set(device_properties::instance().get<device_properties::property::memoryAffinity>()),
    page_kind(pk)
{
}

bool memory_kind::numa_pinned::operator==(numa_pinned const& other) const
{
  return numa_node_set == other.numa_node_set && page_kind == other.page_kind;
}

bool memory_kind::device::operator==(device const& other) const
{
  return device_id == other.device_id;
}

bool memory_kind::managed::operator==(managed const& other) const = default;

bool memory_kind::shared_segment::operator==(shared_segment const& other) const = default;

memory_kind::numa_pool::numa_pool(int numa_node_id) : numa_node_id(numa_node_id) {}

memory_kind::numa_pool::numa_pool()
  : numa_node_id(device_properties::instance().get<device_properties::hostNumaId>())
{
}

bool memory_kind::numa_pool::operator==(numa_pool const& other) const
{
  return numa_node_id == other.numa_node_id;
}

memory_kind::shared_numa_pool::shared_numa_pool(int numa_node_id) : numa_node_id(numa_node_id) {}

memory_kind::shared_numa_pool::shared_numa_pool()
  : numa_node_id(device_properties::instance().get<device_properties::hostNumaId>())
{
}

bool memory_kind::shared_numa_pool::operator==(shared_numa_pool const& other) const
{
  return numa_node_id == other.numa_node_id;
}

std::size_t memory_kind::type_hash::operator()(memory_kind::type const& type) const
{
  std::size_t h = boost::hash_value(type.index());
  std::visit(utility::overloaded{
               [](system const&) {},
               [&h](numa const& n) {
                 boost::hash_combine(h, static_cast<int>(n.page_kind));
                 for (int i = 0; i < cpu_set::max_count; ++i) {
                   if (n.numa_node_set.contains(i)) { boost::hash_combine(h, i); }
                 }
               },
               [](pinned const&) {},
               [&h](numa_pinned const& n) {
                 boost::hash_combine(h, static_cast<int>(n.page_kind));
                 for (int i = 0; i < cpu_set::max_count; ++i) {
                   if (n.numa_node_set.contains(i)) { boost::hash_combine(h, i); }
                 }
               },
               [&h](device const& d) { boost::hash_combine(h, d.device_id.value()); },
               [](managed const&) {},
               [](shared_segment const&) {},
               [&h](numa_pool const& n) { boost::hash_combine(h, n.numa_node_id); },
               [&h](shared_numa_pool const& n) { boost::hash_combine(h, n.numa_node_id); }},
             type);
  return h;
}

fixed_width_compression_statistics& fixed_width_compression_statistics::operator+=(
  const fixed_width_compression_statistics& other)
{
  num_compressed_row_groups += other.num_compressed_row_groups;
  compressed_size += other.compressed_size;
  uncompressed_size += other.uncompressed_size;
  primary_compressed_size += other.primary_compressed_size;
  secondary_compressed_size += other.secondary_compressed_size;
  num_primary_compressed_row_groups += other.num_primary_compressed_row_groups;
  num_secondary_compressed_row_groups += other.num_secondary_compressed_row_groups;
  return *this;
}

string_compression_statistics& string_compression_statistics::operator+=(
  const string_compression_statistics& other)
{
  offsets_stats += other.offsets_stats;
  chars_stats += other.chars_stats;
  return *this;
}

table_statistics::table_statistics(int64_t num_rows_, std::span<cudf::data_type const> column_types)
{
  num_rows    = num_rows_;
  num_columns = column_types.size();
  column_stats.reserve(num_columns);
  for (size_t i = 0; i < column_types.size(); ++i) {
    const auto& col_type = column_types[i];
    if (col_type.id() == cudf::type_id::STRING) {
      column_stats.emplace_back(
        column_statistics{.column_id = i, .compression_stats = string_compression_statistics{}});
    } else {
      column_stats.emplace_back(column_statistics{
        .column_id = i, .compression_stats = fixed_width_compression_statistics{}});
    }
  }
}

namespace detail {
void add_compression_stats(column_compression_statistics& lhs,
                           const column_compression_statistics& rhs)
{
  std::visit(
    [](auto& left, const auto& right) {
      using LT = std::decay_t<decltype(left)>;
      using RT = std::decay_t<decltype(right)>;

      if constexpr (std::is_same_v<LT, RT>) {
        left += right;
      } else {
        throw std::logic_error("Type mismatch in compression statistics");
      }
    },
    lhs,
    rhs);
}
}  // namespace detail

void table_statistics::add_column_statistics(const column_statistics& col_stats)
{
  auto& target_stats       = column_stats[col_stats.column_id].compression_stats;
  const auto& source_stats = col_stats.compression_stats;

  detail::add_compression_stats(target_stats, source_stats);
}

void table_statistics::append_table_statistics(const table_statistics& new_stats)
{
  // Statistics carrying zero columns have nothing to merge per column, and an accumulator carrying
  // none adopts the appended columns wholesale. Neither needs the two sides to correspond; only a
  // merge does.
  bool const merges_columns = num_columns > 0 && new_stats.num_columns > 0;

  // Check the invariants before touching this object, so that a rejected append leaves it as it
  // was rather than half merged.
  if (merges_columns) {
    GQE_EXPECTS(new_stats.num_columns == num_columns,
                std::format("Column count mismatch in appended table statistics: this table has "
                            "{} columns, the appended table has {}",
                            num_columns,
                            new_stats.num_columns));
    for (size_t i = 0; i < num_columns; ++i) {
      GQE_EXPECTS(column_stats[i].column_id == new_stats.column_stats[i].column_id,
                  std::format("Column identity mismatch in appended table statistics: position {} "
                              "holds column {} here and column {} in the appended table",
                              i,
                              column_stats[i].column_id,
                              new_stats.column_stats[i].column_id));
    }
  }

  num_rows += new_stats.num_rows;
  // Sum rather than increment. Callers append both a single written row group and an
  // already-aggregated `table_statistics`, so incrementing would discard the aggregate's own
  // count and under-report a multi-row-group table as 1.
  num_row_groups += new_stats.num_row_groups;
  if (num_columns == 0 && new_stats.num_columns > 0) {
    num_columns  = new_stats.num_columns;
    column_stats = new_stats.column_stats;
  } else if (merges_columns) {
    for (size_t i = 0; i < num_columns; ++i) {
      detail::add_compression_stats(column_stats[i].compression_stats,
                                    new_stats.column_stats[i].compression_stats);
    }
  }
}
}  // namespace gqe
