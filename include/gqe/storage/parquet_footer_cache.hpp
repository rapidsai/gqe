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

#include <cudf/io/parquet_schema.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace gqe {
namespace storage {

/**
 * @brief Shared cache of parsed Parquet footers.
 *
 * Parsing a footer is the expensive step of constructing a hybrid-scan reader. This cache parses
 * each file's footer once and shares the resulting `FileMetaData`, so a reference-generating
 * iterate task and a resolve task construct readers cheaply from the same parsed state, including
 * for references minted on another rank.
 *
 * # Design
 *
 * The cache is externally owned and injected into tasks as a shared handle. Entries are keyed by
 * file path and hold only parsed file metadata, never task configuration, so one cache instance
 * can serve every task. Entries are shared immutable snapshots that remain valid regardless of
 * the cache's later lifetime.
 *
 * # Staleness
 *
 * Entries are keyed by file path and never invalidated; Parquet files are treated as immutable.
 * Keying by path plus file size and modification time is follow-up work.
 *
 * # Thread Safety
 *
 * All member functions are thread-safe.
 */
class parquet_footer_cache {
 public:
  /**
   * @brief Return the parsed footer for a file, fetching and parsing it on a miss.
   *
   * @param[in] file_path Path of the Parquet file.
   *
   * @throws std::invalid_argument If `file_path` is empty.
   *
   * @return A shared immutable snapshot of the file's parsed footer.
   */
  [[nodiscard]] std::shared_ptr<cudf::io::parquet::FileMetaData const> get_or_load(
    std::string const& file_path);

  /**
   * @brief Return the number of cached footers.
   */
  [[nodiscard]] std::size_t size() const;

 private:
  mutable std::mutex _mutex;
  std::unordered_map<std::string, std::shared_ptr<cudf::io::parquet::FileMetaData const>> _entries;
};

}  // namespace storage
}  // namespace gqe
