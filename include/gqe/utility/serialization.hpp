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

#include <cstddef>
#include <filesystem>
#include <string>

namespace gqe::utility {

// Major version of Serialization implementation
inline constexpr int serialization_version_major = 1;
// Minor version of Serialization implementation
inline constexpr int serialization_version_minor = 0;

/**
 * @brief Root directory for serialized in-memory table snapshots (the `zmps-*` folder).
 *
 * Layout under the table Parquet directory (`dataset_location/table_name/`):
 *
 * @code{.unparsed}
 * <table>/
 * └── serialized_data/
 *     └── max_partitions-{max_num_partitions}/
 *         └── zmps-{zone_map_rows}/    # returned by serialized_table_root()
 *             ├── rg-0/
 *             │   ├── zone_maps/       # row-group zone map
 *             │   └── ANS/             # compression format dir
 *             │       └── 128KB_chunks/
 *             │           ├── col_a.bin    # serialized column data
 *             │           └── col_a.json   # serialized column metadata
 *             ├── rg-1/
 *             │   ├── zone_maps/
 *             │   └── ANS/
 *             │       └── 128KB_chunks/
 *             │           └── ...
 *             └── rg-<n>/
 *                 └── ...
 * @endcode
 *
 * Compression directory names: `ANS`, `LZ4`, `Snappy`, … When `use_cpu_compression` is true,
 * `_CPU` is appended. For `compression_format::lz4` with CPU compression, the level is appended
 * as well (e.g. `LZ4_CPU_10`).
 *
 * Matches the path convention previously built in gqe-python.
 *
 * @param table_data_directory Absolute or relative path to the folder that holds the table's
 *                             Parquet files (typically `dataset_path/table_name`).
 * @param params Optimization parameters (row groups, zone map partition size, compression fields).
 */
[[nodiscard]] std::filesystem::path serialized_table_root(
  std::filesystem::path table_data_directory, optimization_parameters const& params);

/**
 * @brief Column snapshot directory for one row group under the table serialized-data root.
 *
 * Path: `<table_serialized_data_root>/rg-{row_group_index}/{compression}/{chunk}/`
 */
[[nodiscard]] std::filesystem::path serialized_row_group_column_root(
  std::filesystem::path table_serialized_data_root,
  size_t row_group_index,
  optimization_parameters const& params);

/**
 * @brief Zone-map directory for one row group under the table serialized-data root.
 *
 * Path: `<table_serialized_data_root>/rg-{row_group_index}/zone_maps/`
 */
[[nodiscard]] std::filesystem::path serialized_row_group_zone_maps_root(
  std::filesystem::path table_serialized_data_root, size_t row_group_index);

}  // namespace gqe::utility
