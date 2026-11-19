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

#include <gqe/storage/parquet_footer_cache.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

std::shared_ptr<cudf::io::parquet::FileMetaData const> parquet_footer_cache::get_or_load(
  std::string const& file_path)
{
  GQE_EXPECTS(
    !file_path.empty(), "parquet_footer_cache: file_path must not be empty", std::invalid_argument);

  // Serialize the miss path as well as map access. This prevents concurrent lookups of the same
  // file from redundantly fetching and parsing its footer.
  std::lock_guard<std::mutex> const guard{_mutex};
  if (auto const found = _entries.find(file_path); found != _entries.end()) {
    return found->second;
  }

  utility::nvtx_scoped_range range{"parquet_footer_cache::load"};

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(file_path));
  auto footers = cudf::io::read_parquet_footers(
    cudf::host_span<std::unique_ptr<cudf::io::datasource> const>{sources.data(), sources.size()});
  GQE_EXPECTS(footers.size() == 1,
              "parquet_footer_cache: expected exactly one parsed footer",
              std::logic_error);

  auto entry = std::make_shared<cudf::io::parquet::FileMetaData const>(std::move(footers.front()));
  auto const [inserted, was_inserted] = _entries.try_emplace(file_path, std::move(entry));
  GQE_EXPECTS(
    was_inserted, "parquet_footer_cache: failed to cache a newly parsed footer", std::logic_error);
  return inserted->second;
}

std::size_t parquet_footer_cache::size() const
{
  std::lock_guard<std::mutex> const guard{_mutex};
  return _entries.size();
}

}  // namespace storage
}  // namespace gqe
