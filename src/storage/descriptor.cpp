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

#include <gqe/storage/descriptor.hpp>

#include <gqe/utility/helpers.hpp>

#include <boost/container_hash/hash.hpp>

namespace gqe::storage {

namespace {

void hash_data_type(std::size_t& seed, cudf::data_type const& dt)
{
  boost::hash_combine(seed, static_cast<int32_t>(dt.id()));
  boost::hash_combine(seed, dt.scale());
}

}  // namespace

std::size_t boost_shared_memory_descriptor::hash() const
{
  std::size_t seed = 0;
  boost::hash_combine(seed, table_name);
  boost::hash_combine(seed, column_names);
  for (auto const& dt : data_types) {
    hash_data_type(seed, dt);
  }
  return seed;
}

std::size_t shared_numa_pool_memory_descriptor::hash() const
{
  std::size_t seed = 0;
  boost::hash_combine(seed, table_name);
  boost::hash_combine(seed, numa_node_id);
  boost::hash_combine(seed, column_names);
  for (auto const& dt : data_types) {
    hash_data_type(seed, dt);
  }
  return seed;
}

std::size_t parquet_file_descriptor::hash() const
{
  std::size_t seed = 0;
  boost::hash_combine(seed, table_name);
  boost::hash_combine(seed, file_paths);
  return seed;
}

std::size_t local_memory_descriptor::hash() const
{
  std::size_t seed = 0;
  boost::hash_combine(seed, table_name);
  return seed;
}

std::string table_name_of(descriptor const& desc)
{
  return std::visit([](auto const& d) { return d.table_name; }, desc);
}

descriptor make_descriptor(storage_kind::type const& kind,
                           std::string const& table_name,
                           std::vector<std::string> const& column_names,
                           std::vector<cudf::data_type> const& column_types)
{
  return std::visit(
    utility::overloaded{[&](storage_kind::parquet_file const& f) -> descriptor {
                          return parquet_file_descriptor{table_name, f.file_paths};
                        },
                        [&](storage_kind::boost_shared_memory const&) -> descriptor {
                          return boost_shared_memory_descriptor{
                            table_name, column_names, column_types};
                        },
                        [&](storage_kind::shared_numa_pool_memory const& m) -> descriptor {
                          return shared_numa_pool_memory_descriptor{
                            table_name, m.numa_node_id, column_names, column_types};
                        },
                        [&](auto const&) -> descriptor {
                          // Process-local memory kinds (system, device, pinned, managed, etc.)
                          return local_memory_descriptor{table_name};
                        }},
    kind);
}

}  // namespace gqe::storage
