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

#pragma once

#include <gqe/executor/optimization_parameters.hpp>

#include <proto/gqe_in_memory_tables.pb.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>

namespace gqe::utility {

// Serialization format version. The on-disk metadata (Protobuf) is strictly versioned: a reader
// rejects any file whose version differs, so every metadata schema change must bump this.
inline constexpr int serialization_version_major = 2;
inline constexpr int serialization_version_minor = 0;

/// Magic number prefixing every serialized column binary file: ASCII "GQE" (G=0x47, Q=0x51,
/// E=0x45). Used to identify and validate the on-disk format.
inline constexpr std::uint32_t serialized_column_magic = 0x475145u;

/// Magic number prefixing every serialized zone-map binary file: ASCII "GQEZ" (G=0x47, Q=0x51,
/// E=0x45, Z=0x5A). Used to identify and validate the on-disk zone-map format.
inline constexpr std::uint32_t serialized_zone_map_magic = 0x4751455Au;

/**
 * @brief Write a trivially-copyable value to a binary stream in native byte order.
 *
 * Shared by every on-disk serializer (columns, zone maps) so the binary encoding stays identical.
 */
template <typename T>
void write_pod(std::ostream& os, T const& value)
{
  static_assert(std::is_trivially_copyable_v<T>,
                "write_pod serializes T as its raw object bytes, so T must be trivially copyable; "
                "Use a fixed-width type for a stable on-disk layout");
  os.write(reinterpret_cast<char const*>(&value), sizeof(T));
}

/**
 * @brief Read a trivially-copyable value from a binary stream in native byte order.
 */
template <typename T>
void read_pod(std::istream& is, T& value)
{
  static_assert(
    std::is_trivially_copyable_v<T>,
    "read_pod deserializes T from its raw object bytes, so T must be trivially copyable; "
    "use a fixed-width type for a stable on-disk layout");
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
}

/**
 * @brief Write a serialized column's metadata header as `[magic][length][protobuf]`.
 *
 * Sets the format version on @p metadata before encoding, so callers never spell the version
 * themselves. Leaves @p os positioned immediately after the header, ready for the caller to append
 * the raw partition payloads.
 *
 * @param[in] os Binary output stream positioned where the header should start.
 * @param[in,out] metadata Header to encode. Its `version` submessage is overwritten.
 */
void write_serialized_column_metadata(std::ostream& os, proto::ColumnMetadata& metadata);

/**
 * @brief Read and validate a serialized column's metadata header written by @ref
 * write_serialized_column_metadata.
 *
 * Leaves @p is positioned at the raw partition payloads. Throws @c std::invalid_argument if the
 * magic is wrong, the file is truncated, the serialization version does not match exactly, or the
 * protobuf carries an unknown field at any nesting depth. The format is strictly versioned, so a
 * header written by a newer schema is rejected rather than silently accepted.
 *
 * Every populated per-partition array has length equal to the column's partition count, and the
 * string-only arrays are empty for non-string columns. Neither is enforced here, because the
 * partition count is the caller's to derive.
 *
 * @param[in] is Binary input stream positioned at the start of the column file.
 */
[[nodiscard]] proto::ColumnMetadata read_serialized_column_metadata(std::istream& is);

/**
 * @brief Write a serialized zone map's metadata header as `[magic][length][protobuf]`.
 *
 * Sets the format version on @p metadata before encoding. Leaves @p os positioned immediately after
 * the header, ready for the caller to append the raw Arrow IPC min/max record batch.
 *
 * @param[in] os Binary output stream positioned where the header should start.
 * @param[in,out] metadata Header to encode. Its `version` submessage is overwritten.
 */
void write_serialized_zone_map_metadata(std::ostream& os, proto::ZoneMapMetadata& metadata);

/**
 * @brief Read and validate a serialized zone map's metadata header written by @ref
 * write_serialized_zone_map_metadata.
 *
 * Leaves @p is positioned at the raw Arrow IPC record batch. Throws @c std::invalid_argument on a
 * wrong magic, truncation, a version mismatch, or any unknown protobuf field. As with @ref
 * read_serialized_column_metadata, messages name no file.
 *
 * @param[in] is Binary input stream positioned at the start of the zone-map file.
 */
[[nodiscard]] proto::ZoneMapMetadata read_serialized_zone_map_metadata(std::istream& is);

/**
 * @brief Root directory of one serialized in-memory table snapshot (the `{chunk}` folder).
 *
 * Layout under the table Parquet directory (`serialized_dataset_location/table_name/`):
 *
 * @code{.unparsed}
 * <table>/
 * └── serialized_data/
 *     └── max_partitions-{num_partitions}/
 *         └── zmps-{zone_map_rows}/
 *             └── decimal_type-{double|decimal}/
 *                 └── ANS/                         # compression format dir
 *                     └── 128KB_chunks/            # returned by serialized_table_root()
 *                         ├── _COMPLETE            # completion marker, written last
 *                         ├── rg-0/
 *                         │   ├── zone_maps/       # row-group zone map
 *                         │   └── col_a.bin        # serialized column data
 *                         ├── rg-1/
 *                         │   └── ...
 *                         └── rg-<n>/
 *                             └── ...
 * @endcode
 *
 * Every parameter that keys the layout is a directory above the `rg-*` directories. Each snapshot
 * root therefore holds exactly one snapshot, which a serialization can replace as a whole without
 * touching snapshots taken under other settings. A snapshot is loadable only while its @ref
 * serialized_snapshot_marker_path exists.
 *
 * Compression directory names: `ANS`, `LZ4`, `Snappy`, … When `use_cpu_compression` is true,
 * `_CPU` is appended. For `compression_format::lz4` with CPU compression, the level is appended
 * as well (e.g. `LZ4_CPU_10`).
 *
 * @param table_data_directory Absolute or relative path to the folder that holds the table's
 *                             Parquet files (typically `dataset_path/table_name`).
 * @param params Optimization parameters (row groups, zone map partition size, decimal type, and
 *               compression fields) that key the on-disk layout.
 */
[[nodiscard]] std::filesystem::path serialized_table_root(
  std::filesystem::path table_data_directory, optimization_parameters const& params);

/**
 * @brief Snapshot root for @p table_name, derived from the session's `serialized_dataset_location`
 * rather than a caller-supplied directory.
 *
 * Composes `<serialized_dataset_location>/<table_name>` and forwards to @ref
 * serialized_table_root, so the serialize and deserialize triggers agree on where a table's
 * snapshot lives without either of them spelling the convention out.
 *
 * @param[in] params Optimization parameters holding the dataset location and the layout key.
 * @param[in] table_name Name of the table whose snapshot root is wanted.
 *
 * @throw std::invalid_argument if `serialized_dataset_location` is not set, since there is no
 *        default snapshot location to fall back on.
 */
[[nodiscard]] std::string serialized_data_root(optimization_parameters const& params,
                                               std::string const& table_name);

/**
 * @brief Column snapshot directory for one row group under the table serialized-data root.
 *
 * Path: `<table_serialized_data_root>/rg-{row_group_index}/`
 */
[[nodiscard]] std::filesystem::path serialized_row_group_column_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index);

/**
 * @brief Zone-map directory for one row group under the table serialized-data root.
 *
 * Path: `<table_serialized_data_root>/rg-{row_group_index}/zone_maps/`
 */
[[nodiscard]] std::filesystem::path serialized_row_group_zone_maps_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index);

/**
 * @brief Completion marker of the snapshot under the table serialized-data root.
 *
 * Path: `<table_serialized_data_root>/_COMPLETE`
 *
 * Serialization removes the marker before it writes any snapshot file and writes it after the last
 * one. Deserialization refuses a snapshot without it, so a snapshot that a failed or interrupted
 * serialization left behind is never loaded.
 */
[[nodiscard]] std::filesystem::path serialized_snapshot_marker_path(
  std::filesystem::path table_serialized_data_root);

}  // namespace gqe::utility
