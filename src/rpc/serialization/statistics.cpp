/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/rpc/serialization/statistics.hpp>

#include <gqe/utility/helpers.hpp>

namespace gqe::rpc {

namespace {

proto::FixedWidthCompressionStatistics serialize_fixed_width(
  fixed_width_compression_statistics const& stats)
{
  proto::FixedWidthCompressionStatistics out;
  out.set_num_compressed_row_groups(stats.num_compressed_row_groups);
  out.set_compressed_size(stats.compressed_size);
  out.set_uncompressed_size(stats.uncompressed_size);
  out.set_primary_compressed_size(stats.primary_compressed_size);
  out.set_secondary_compressed_size(stats.secondary_compressed_size);
  out.set_num_primary_compressed_row_groups(stats.num_primary_compressed_row_groups);
  out.set_num_secondary_compressed_row_groups(stats.num_secondary_compressed_row_groups);
  return out;
}

fixed_width_compression_statistics deserialize_fixed_width(
  proto::FixedWidthCompressionStatistics const& pb)
{
  return {pb.num_compressed_row_groups(),
          pb.compressed_size(),
          pb.uncompressed_size(),
          pb.primary_compressed_size(),
          pb.secondary_compressed_size(),
          pb.num_primary_compressed_row_groups(),
          pb.num_secondary_compressed_row_groups()};
}

proto::ColumnCompressionStatistics serialize_column_compression(
  column_compression_statistics const& stats)
{
  proto::ColumnCompressionStatistics out;
  std::visit(
    utility::overloaded{[&](fixed_width_compression_statistics const& fw) {
                          *out.mutable_fixed_width() = serialize_fixed_width(fw);
                        },
                        [&](string_compression_statistics const& str) {
                          auto* msg                     = out.mutable_string();
                          *msg->mutable_offsets_stats() = serialize_fixed_width(str.offsets_stats);
                          *msg->mutable_chars_stats()   = serialize_fixed_width(str.chars_stats);
                        }},
    stats);
  return out;
}

column_compression_statistics deserialize_column_compression(
  proto::ColumnCompressionStatistics const& pb)
{
  switch (pb.stats_case()) {
    case proto::ColumnCompressionStatistics::kFixedWidth:
      return deserialize_fixed_width(pb.fixed_width());
    case proto::ColumnCompressionStatistics::kString:
      return string_compression_statistics{deserialize_fixed_width(pb.string().offsets_stats()),
                                           deserialize_fixed_width(pb.string().chars_stats())};
    default: return fixed_width_compression_statistics{};
  }
}

}  // namespace

proto::TableStatistics serialize_table_statistics(table_statistics const& stats)
{
  proto::TableStatistics out;
  out.set_num_rows(stats.num_rows);
  out.set_num_row_groups(stats.num_row_groups);
  out.set_num_columns(stats.num_columns);
  for (auto const& col : stats.column_stats) {
    auto* pb_col = out.add_column_stats();
    pb_col->set_column_id(col.column_id);
    *pb_col->mutable_compression_stats() = serialize_column_compression(col.compression_stats);
  }
  return out;
}

table_statistics deserialize_table_statistics(proto::TableStatistics const& pb)
{
  table_statistics out;
  out.num_rows       = pb.num_rows();
  out.num_row_groups = pb.num_row_groups();
  out.num_columns    = pb.num_columns();
  out.column_stats.reserve(pb.column_stats_size());
  for (auto const& col : pb.column_stats()) {
    out.column_stats.push_back(
      {col.column_id(), deserialize_column_compression(col.compression_stats())});
  }
  return out;
}

}  // namespace gqe::rpc
