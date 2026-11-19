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

#include <proto/gqe_in_memory_tables.pb.h>

#include <fmt/format.h>
#include <google/protobuf/descriptor.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    case compression_format::none: basename = "columns"; break;
    case compression_format::ans: basename = "ANS"; break;
    case compression_format::lz4: basename = "LZ4"; break;
    case compression_format::snappy: basename = "Snappy"; break;
    case compression_format::gdeflate: basename = "GDeflate"; break;
    case compression_format::deflate: basename = "Deflate"; break;
    case compression_format::cascaded: basename = "Cascaded"; break;
    case compression_format::bitpacking: basename = "Bitpacking"; break;
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

// Framing shared by every serialized metadata header: [magic uint32][length uint64][protobuf
// bytes].
void write_metadata_frame(std::ostream& os,
                          std::uint32_t magic,
                          google::protobuf::Message const& message)
{
  std::string bytes;
  if (!message.SerializeToString(&bytes)) {
    throw std::logic_error("serialize: failed to encode metadata protobuf");
  }
  write_pod(os, magic);
  write_pod(os, static_cast<std::uint64_t>(bytes.size()));
  os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Stamp a header with the format version this build writes. Centralized so no caller can spell
// the version itself and drift from the constants.
void set_format_version(proto::FormatVersion& version)
{
  version.set_major_version(serialization_version_major);
  version.set_minor_version(serialization_version_minor);
}

// Reject a header whose format version is not exactly the one this build writes. The on-disk
// format carries no backward compatibility, so an older or newer file is refused outright rather
// than partially interpreted.
void check_format_version(proto::FormatVersion const& version)
{
  if (version.major_version() != serialization_version_major ||
      version.minor_version() != serialization_version_minor) {
    throw std::invalid_argument(
      fmt::format("deserialize: incompatible metadata version v{}.{}, expected v{}.{}",
                  version.major_version(),
                  version.minor_version(),
                  serialization_version_major,
                  serialization_version_minor));
  }
}

// Report whether @p message or any submessage reachable from it carries an unknown field. The
// recursion is what makes the strict-versioning guarantee hold for nested headers, since
// `GetUnknownFields` only reports fields unknown at the level it is asked about.
[[nodiscard]] bool has_unknown_fields(google::protobuf::Message const& message)
{
  auto const* reflection = message.GetReflection();
  if (!reflection->GetUnknownFields(message).empty()) { return true; }

  // ListFields yields only the fields actually present on the wire, so an absent submessage costs
  // nothing here.
  std::vector<google::protobuf::FieldDescriptor const*> fields;
  reflection->ListFields(message, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) { continue; }
    if (field->is_repeated()) {
      int const count = reflection->FieldSize(message, field);
      for (int i = 0; i < count; ++i) {
        if (has_unknown_fields(reflection->GetRepeatedMessage(message, field, i))) { return true; }
      }
    } else if (has_unknown_fields(reflection->GetMessage(message, field))) {
      return true;
    }
  }
  return false;
}

// Read and validate [magic][length][protobuf], parsing into `message`. Enforces the exact magic and
// rejects any unknown protobuf field, at any nesting depth, because the on-disk format is strictly
// versioned. Leaves `is` positioned immediately after the header, at the raw payloads. Messages
// name no file, because only the caller knows which one it is reading.
void read_metadata_frame(std::istream& is,
                         std::uint32_t expected_magic,
                         google::protobuf::Message& message)
{
  std::uint32_t magic = 0;
  read_pod(is, magic);
  if (is.fail() || magic != expected_magic) {
    throw std::invalid_argument(fmt::format(
      "deserialize: bad or missing magic (got 0x{:X}, expected 0x{:X})", magic, expected_magic));
  }
  std::uint64_t length = 0;
  read_pod(is, length);
  if (is.fail()) { throw std::invalid_argument("deserialize: truncated metadata length"); }
  std::string bytes(static_cast<std::size_t>(length), '\0');
  if (length > 0) { is.read(bytes.data(), static_cast<std::streamsize>(length)); }
  if (is.gcount() != static_cast<std::streamsize>(length)) {
    throw std::invalid_argument("deserialize: truncated metadata");
  }
  if (!message.ParseFromString(bytes)) {
    throw std::invalid_argument("deserialize: malformed metadata protobuf");
  }
  if (has_unknown_fields(message)) {
    throw std::invalid_argument(
      "deserialize: metadata contains unknown protobuf fields, so it was written by a newer, "
      "incompatible serialization version");
  }
}

}  // namespace

void write_serialized_column_metadata(std::ostream& os, proto::ColumnMetadata& metadata)
{
  set_format_version(*metadata.mutable_version());
  write_metadata_frame(os, serialized_column_magic, metadata);
}

proto::ColumnMetadata read_serialized_column_metadata(std::istream& is)
{
  proto::ColumnMetadata metadata;
  read_metadata_frame(is, serialized_column_magic, metadata);
  check_format_version(metadata.version());
  return metadata;
}

void write_serialized_zone_map_metadata(std::ostream& os, proto::ZoneMapMetadata& metadata)
{
  set_format_version(*metadata.mutable_version());
  write_metadata_frame(os, serialized_zone_map_magic, metadata);
}

proto::ZoneMapMetadata read_serialized_zone_map_metadata(std::istream& is)
{
  proto::ZoneMapMetadata metadata;
  read_metadata_frame(is, serialized_zone_map_magic, metadata);
  check_format_version(metadata.version());
  return metadata;
}

std::filesystem::path serialized_table_root(std::filesystem::path table_data_directory,
                                            optimization_parameters const& params)
{
  // decimal_type is part of the layout key: it changes the on-disk column types (DECIMAL columns
  // are stored as FLOAT64 under `double` and as cuDF fixed-point under `decimal`), so snapshots
  // taken under different decimal_type values must not share a directory or a deserialize would
  // read data of the wrong type.
  return std::move(table_data_directory) / "serialized_data" /
         ("max_partitions-" + std::to_string(params.num_partitions)) /
         ("zmps-" + std::to_string(params.zone_map_partition_size)) /
         ("decimal_type-" + gqe::to_string(params.decimal_type)) /
         compression_format_directory_name(
           params.compression_format, params.use_cpu_compression, params.compression_level) /
         chunk_size_directory_name(params.compression_chunk_size);
}

std::string serialized_data_root(optimization_parameters const& params,
                                 std::string const& table_name)
{
  if (params.serialized_dataset_location.empty()) {
    throw std::invalid_argument(
      "serialize/deserialize requires the 'serialized_dataset_location' optimization parameter to "
      "be set (e.g. SET serialized_dataset_location TO '/path/to/dataset')");
  }
  return serialized_table_root(
           std::filesystem::path(params.serialized_dataset_location) / table_name, params)
    .string();
}

std::filesystem::path serialized_row_group_column_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index)
{
  return std::move(table_serialized_data_root) / ("rg-" + std::to_string(row_group_index));
}

std::filesystem::path serialized_row_group_zone_maps_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index)
{
  return std::move(table_serialized_data_root) / ("rg-" + std::to_string(row_group_index)) /
         "zone_maps";
}

std::filesystem::path serialized_snapshot_marker_path(
  std::filesystem::path table_serialized_data_root)
{
  return std::move(table_serialized_data_root) / "_COMPLETE";
}

}  // namespace gqe::utility
