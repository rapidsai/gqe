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

#include <gqe/rpc/serialization/storage.hpp>

#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/utility/helpers.hpp>

#include <stdexcept>

namespace gqe::rpc {

namespace {

void serialize_common_metadata(proto::StorageDescriptor& out,
                               std::string const& table_name,
                               std::vector<std::string> const& column_names,
                               std::vector<cudf::data_type> const& data_types)
{
  out.set_table_name(table_name);
  for (auto const& name : column_names) {
    out.add_column_names(name);
  }
  for (auto const& dt : data_types) {
    *out.add_data_types() = serialize_data_type(dt);
  }
}

}  // namespace

proto::StorageDescriptor serialize_storage_descriptor(storage::descriptor const& desc)
{
  proto::StorageDescriptor out;
  std::visit(utility::overloaded{
               [&](storage::boost_shared_memory_descriptor const& d) {
                 out.mutable_boost_shared_memory();
                 serialize_common_metadata(out, d.table_name, d.column_names, d.data_types);
               },
               [&](storage::shared_numa_pool_memory_descriptor const& d) {
                 auto* msg = out.mutable_shared_numa_pool_memory();
                 msg->set_numa_node_id(d.numa_node_id);
                 serialize_common_metadata(out, d.table_name, d.column_names, d.data_types);
               },
               [&](storage::parquet_file_descriptor const& d) {
                 auto* msg = out.mutable_parquet_file();
                 for (auto const& path : d.file_paths) {
                   msg->add_file_paths(path);
                 }
                 out.set_table_name(d.table_name);
               },
               [&](storage::local_memory_descriptor const&) {
                 throw std::logic_error(
                   "local_memory_descriptor cannot be serialized for multi-process execution");
               }},
             desc);
  return out;
}

storage::descriptor deserialize_storage_descriptor(proto::StorageDescriptor const& proto)
{
  auto table_name = proto.table_name();
  std::vector<std::string> column_names(proto.column_names().begin(), proto.column_names().end());
  std::vector<cudf::data_type> data_types;
  data_types.reserve(proto.data_types_size());
  for (auto const& dt : proto.data_types()) {
    data_types.push_back(deserialize_data_type(dt));
  }

  switch (proto.storage_case()) {
    case proto::StorageDescriptor::kBoostSharedMemory:
      return storage::boost_shared_memory_descriptor{
        std::move(table_name), std::move(column_names), std::move(data_types)};
    case proto::StorageDescriptor::kSharedNumaPoolMemory:
      return storage::shared_numa_pool_memory_descriptor{
        std::move(table_name),
        proto.shared_numa_pool_memory().numa_node_id(),
        std::move(column_names),
        std::move(data_types)};
    case proto::StorageDescriptor::kParquetFile: {
      std::vector<std::string> paths(proto.parquet_file().file_paths().begin(),
                                     proto.parquet_file().file_paths().end());
      return storage::parquet_file_descriptor{std::move(table_name), std::move(paths)};
    }
    case proto::StorageDescriptor::STORAGE_NOT_SET:
      throw std::logic_error("StorageDescriptor has no storage set");
    default: throw std::logic_error("Unknown StorageDescriptor type");
  }
}

}  // namespace gqe::rpc
