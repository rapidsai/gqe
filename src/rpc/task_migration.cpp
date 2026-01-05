/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/rpc/task_migration.hpp>

#include <gqe/executor/task.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/mpi_helpers.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <grpc/grpc.h>
#include <nvshmemx.h>
#include <proto/task.grpc.pb.h>
#include <proto/task.pb.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <shared_mutex>
#include <unordered_map>

namespace gqe {

rpc_server::rpc_server(std::vector<grpc::Service*> services)
{
  grpc::ServerBuilder builder;
  for (auto& service : services) {
    builder.RegisterService(service);
  }
  int port;
  // FIXME: Currently we only support intra node communication. The server address should be a input
  // parameter.
  server_address = "0.0.0.0:0";
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port);
  server = builder.BuildAndStart();
  if (server == nullptr) { throw std::runtime_error("Failed to start RPC server"); }
  GQE_LOG_INFO("RPC server listening on port {}", port);
  server_address = "0.0.0.0:" + std::to_string(port);
}

grpc::Status task_migration_service::FetchTaskMetadata(grpc::ServerContext* context,
                                                       const proto::TaskId* task_id,
                                                       proto::TaskMetadata* task_metadata)
{
  std::shared_lock<std::shared_mutex> lock(_task_map_mutex);
  auto it = _task_map.find(task_id->id());
  if (it == _task_map.end()) { return grpc::Status(grpc::StatusCode::NOT_FOUND, "Task not found"); }

  // Task result must be kept alive till all requests for this task in the task graph are satisfied
  assert(it->second != nullptr);

  *task_metadata = to_proto(*it->second);
  return grpc::Status::OK;
}

void task_migration_service::register_task(task* t)
{
  std::unique_lock<std::shared_mutex> lock(_task_map_mutex);
  _task_map[std::to_string(t->task_id())] = t;
}

/** Convert a cudf::data_type to a proto::DataType
 *
 * @note This function is required because gRPC does not allow importing an enum.
 *
 * @param[in] dtype cudf::data_type to convert
 * @return proto::DataType
 */
proto::DataType task_migration_service::to_proto(cudf::data_type const& dtype)
{
  proto::DataType proto_dtype;
  switch (dtype.id()) {
    case cudf::type_id::EMPTY: proto_dtype = proto::DataType::EMPTY; break;
    case cudf::type_id::INT8: proto_dtype = proto::DataType::INT8; break;
    case cudf::type_id::INT16: proto_dtype = proto::DataType::INT16; break;
    case cudf::type_id::INT32: proto_dtype = proto::DataType::INT32; break;
    case cudf::type_id::INT64: proto_dtype = proto::DataType::INT64; break;
    case cudf::type_id::UINT8: proto_dtype = proto::DataType::UINT8; break;
    case cudf::type_id::UINT16: proto_dtype = proto::DataType::UINT16; break;
    case cudf::type_id::UINT32: proto_dtype = proto::DataType::UINT32; break;
    case cudf::type_id::UINT64: proto_dtype = proto::DataType::UINT64; break;
    case cudf::type_id::FLOAT32: proto_dtype = proto::DataType::FLOAT32; break;
    case cudf::type_id::FLOAT64: proto_dtype = proto::DataType::FLOAT64; break;
    case cudf::type_id::BOOL8: proto_dtype = proto::DataType::BOOL8; break;
    case cudf::type_id::TIMESTAMP_DAYS: proto_dtype = proto::DataType::TIMESTAMP_DAYS; break;
    case cudf::type_id::TIMESTAMP_SECONDS: proto_dtype = proto::DataType::TIMESTAMP_SECONDS; break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      proto_dtype = proto::DataType::TIMESTAMP_MILLISECONDS;
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      proto_dtype = proto::DataType::TIMESTAMP_MICROSECONDS;
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      proto_dtype = proto::DataType::TIMESTAMP_NANOSECONDS;
      break;
    case cudf::type_id::DURATION_DAYS: proto_dtype = proto::DataType::DURATION_DAYS; break;
    case cudf::type_id::DURATION_SECONDS: proto_dtype = proto::DataType::DURATION_SECONDS; break;
    case cudf::type_id::DURATION_MILLISECONDS:
      proto_dtype = proto::DataType::DURATION_MILLISECONDS;
      break;
    case cudf::type_id::DURATION_MICROSECONDS:
      proto_dtype = proto::DataType::DURATION_MICROSECONDS;
      break;
    case cudf::type_id::DURATION_NANOSECONDS:
      proto_dtype = proto::DataType::DURATION_NANOSECONDS;
      break;
    case cudf::type_id::DICTIONARY32: proto_dtype = proto::DataType::DICTIONARY32; break;
    case cudf::type_id::STRING: proto_dtype = proto::DataType::STRING; break;
    case cudf::type_id::LIST: proto_dtype = proto::DataType::LIST; break;
    case cudf::type_id::DECIMAL32: proto_dtype = proto::DataType::DECIMAL32; break;
    case cudf::type_id::DECIMAL64: proto_dtype = proto::DataType::DECIMAL64; break;
    case cudf::type_id::DECIMAL128: proto_dtype = proto::DataType::DECIMAL128; break;
    case cudf::type_id::STRUCT: proto_dtype = proto::DataType::STRUCT; break;
    default: throw std::runtime_error("Unsupported data type");
  }
  return proto_dtype;
}

/**
 * Generate a proto::Column from a cudf::column_view
 */
proto::Column task_migration_service::to_proto(cudf::column_view const& col)
{
  proto::Column proto_col;
  proto_col.set_type(to_proto(col.type()));
  proto_col.set_size(col.size());
  proto_col.set_null_count(col.null_count());

  if (cudf::is_fixed_width(col.type())) {
    proto_col.set_data_bytes(col.size() * cudf::size_of(col.type()));
  } else if (to_proto(col.type()) == proto::DataType::STRING) {
    GQE_CUDA_TRY(cudaSetDevice(_device_id.value()));
    rmm::cuda_stream temp_stream;
    proto_col.set_data_bytes(cudf::strings_column_view(col).chars_size(temp_stream.view()));
  } else if (to_proto(col.type()) == proto::DataType::LIST ||
             to_proto(col.type()) == proto::DataType::STRUCT) {
    proto_col.set_data_bytes(0);
  } else {
    throw std::runtime_error("Unsupported data type");
  }

  auto data      = col.data<std::byte>();
  auto null_mask = col.null_mask();

  proto_col.set_data_location(reinterpret_cast<uint64_t>(data));
  proto_col.set_null_mask_location(reinterpret_cast<uint64_t>(null_mask));
  for (auto child = col.child_begin(); child != col.child_end(); ++child) {
    proto_col.add_children()->CopyFrom(to_proto(*child));
  }
  return proto_col;
}

/**
 * Generate a proto::Table from a cudf::table_view
 */
proto::Table task_migration_service::to_proto(cudf::table_view const& table)
{
  proto::Table proto_table;
  proto_table.set_num_rows(table.num_rows());
  for (auto col = table.begin(); col != table.end(); ++col) {
    proto_table.add_columns()->CopyFrom(to_proto(*col));
  }
  return proto_table;
}

proto::TaskMetadata task_migration_service::to_proto(task const& t)
{
  proto::TaskMetadata proto_task_metadata;
  auto status = t._status.load();
  proto_task_metadata.set_status(status);

  proto::TaskId* proto_task_id = proto_task_metadata.mutable_id();
  proto_task_id->set_id(std::to_string(t._task_id));

  proto::Result* proto_result = proto_task_metadata.mutable_result();

  if (status == task::status_type::finished) {
    std::visit(
      utility::overloaded{[&](const result_kind::owned& result) {
                            proto_result->mutable_table()->CopyFrom(to_proto(result.table->view()));
                          },
                          [&](const result_kind::borrowed& result) {
                            proto_result->mutable_table()->CopyFrom(to_proto(result.view));
                          }},
      t._result);
  }
  return proto_task_metadata;
}

void task_migration_client::set_owned_result(gqe::task* t, std::unique_ptr<cudf::table> result)
{
  t->emit_result(std::move(result));
}

void task_migration_client::set_borrowed_result(gqe::task* t, cudf::table_view result)
{
  t->emit_result(result);
}

nvshmem_task_migration_client::nvshmem_task_migration_client(nvshmem_communicator* comm,
                                                             rpc_server const& local_server,
                                                             void* pgas_base_ptr)
  : _comm(comm)
{
  // All gather PGAS base pointer allocations
  auto base_ptrs = gqe::utility::multi_process::all_gather_ptr(comm->mpi_comm(), pgas_base_ptr);

  // All gather ranks
  // We need to know the ranks since MPI rank might not be the same as the communicator rank
  auto ranks = gqe::utility::multi_process::all_gather_int(comm->mpi_comm(), _comm->rank());

  // All gather RPC server addresses
  auto all_server_addresses = gqe::utility::multi_process::all_gather_string(
    comm->mpi_comm(), local_server.get_server_address());

  for (int i = 0; i < comm->world_size(); ++i) {
    _stubs.insert({ranks[i],
                   proto::TaskMigration::NewStub(grpc::CreateChannel(
                     all_server_addresses[i], grpc::InsecureChannelCredentials()))});
    _pgas_base_ptrs.insert({ranks[i], base_ptrs[i]});
  }
}

void nvshmem_task_migration_client::migrate_result(gqe::task* t,
                                                   std::unordered_set<int32_t> candidate_ranks)
{
  proto::TaskMetadata task_metadata;
  task_metadata.set_status(proto::TaskStatus::not_started);

  for (auto rank : candidate_ranks) {
    GQE_LOG_TRACE(
      "Migrating result for task {}, stage {}, from rank {}", t->task_id(), t->stage_id(), rank);
    auto stub_iter = _stubs.find(rank);
    if (stub_iter == _stubs.end()) {
      GQE_LOG_ERROR("No channel to rank {}, skipping", rank);
      continue;
    }
    auto stub = stub_iter->second.get();
    try {
      task_metadata = FetchTaskMetadata(stub, t);
    } catch (std::runtime_error& e) {
      GQE_LOG_ERROR("RPC to rank {} failed for task {}, error: {}", rank, t->task_id(), e.what());
      continue;
    }
    if (task_metadata.status() == proto::TaskStatus::failed) {
      GQE_LOG_ERROR(
        "Cannot migrate result for task {} from rank {}, task failed", t->task_id(), rank);
      continue;
    } else if (task_metadata.status() == proto::TaskStatus::finished) {
      auto result = from_proto(task_metadata.result().table(), rank);
      set_owned_result(t, std::move(result));
      return;
    } else {
      // We only migrate tasks which belong to stages before the currently executing stage
      throw std::logic_error("Error migrating result for task " + std::to_string(t->task_id()) +
                             " from rank " + std::to_string(rank) +
                             ": Dependent task from previous stage is not executed.");
    }
  }
  throw std::runtime_error("Could not find candidate rank for task " +
                           std::to_string(t->task_id()) + " migration");
}

proto::TaskMetadata nvshmem_task_migration_client::FetchTaskMetadata(
  proto::TaskMigration::Stub* stub, gqe::task const* t)
{
  proto::TaskMetadata task_metadata;
  proto::TaskId task_id;
  task_id.set_id(std::to_string(t->task_id()));

  grpc::ClientContext context;
  grpc::Status status = stub->FetchTaskMetadata(&context, task_id, &task_metadata);
  if (!status.ok()) {
    throw std::runtime_error(
      "Failed to fetch task metadata. RPC error code: " + std::to_string(status.error_code()) +
      ", RPC error message: " + status.error_message());
  }
  return task_metadata;
}

/** Convert a proto::DataType to a cudf::data_type
 *
 * @note This function is required because gRPC does not allow importing an enum.
 *
 * @param[in] proto_dtype proto::DataType to convert
 * @return cudf::data_type
 */
cudf::data_type nvshmem_task_migration_client::from_proto(proto::DataType proto_dtype)
{
  cudf::data_type dtype;
  switch (proto_dtype) {
    case proto::DataType::EMPTY: dtype = cudf::data_type(cudf::type_id::EMPTY); break;
    case proto::DataType::INT8: dtype = cudf::data_type(cudf::type_id::INT8); break;
    case proto::DataType::INT16: dtype = cudf::data_type(cudf::type_id::INT16); break;
    case proto::DataType::INT32: dtype = cudf::data_type(cudf::type_id::INT32); break;
    case proto::DataType::INT64: dtype = cudf::data_type(cudf::type_id::INT64); break;
    case proto::DataType::UINT8: dtype = cudf::data_type(cudf::type_id::UINT8); break;
    case proto::DataType::UINT16: dtype = cudf::data_type(cudf::type_id::UINT16); break;
    case proto::DataType::UINT32: dtype = cudf::data_type(cudf::type_id::UINT32); break;
    case proto::DataType::UINT64: dtype = cudf::data_type(cudf::type_id::UINT64); break;
    case proto::DataType::FLOAT32: dtype = cudf::data_type(cudf::type_id::FLOAT32); break;
    case proto::DataType::FLOAT64: dtype = cudf::data_type(cudf::type_id::FLOAT64); break;
    case proto::DataType::BOOL8: dtype = cudf::data_type(cudf::type_id::BOOL8); break;
    case proto::DataType::TIMESTAMP_DAYS:
      dtype = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
      break;
    case proto::DataType::TIMESTAMP_SECONDS:
      dtype = cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
      break;
    case proto::DataType::TIMESTAMP_MILLISECONDS:
      dtype = cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
      break;
    case proto::DataType::TIMESTAMP_MICROSECONDS:
      dtype = cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
      break;
    case proto::DataType::TIMESTAMP_NANOSECONDS:
      dtype = cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
      break;
    case proto::DataType::DURATION_DAYS:
      dtype = cudf::data_type(cudf::type_id::DURATION_DAYS);
      break;
    case proto::DataType::DURATION_SECONDS:
      dtype = cudf::data_type(cudf::type_id::DURATION_SECONDS);
      break;
    case proto::DataType::DURATION_MILLISECONDS:
      dtype = cudf::data_type(cudf::type_id::DURATION_MILLISECONDS);
      break;
    case proto::DataType::DURATION_MICROSECONDS:
      dtype = cudf::data_type(cudf::type_id::DURATION_MICROSECONDS);
      break;
    case proto::DataType::DURATION_NANOSECONDS:
      dtype = cudf::data_type(cudf::type_id::DURATION_NANOSECONDS);
      break;
    case proto::DataType::DICTIONARY32: dtype = cudf::data_type(cudf::type_id::DICTIONARY32); break;
    case proto::DataType::STRING: dtype = cudf::data_type(cudf::type_id::STRING); break;
    case proto::DataType::LIST: dtype = cudf::data_type(cudf::type_id::LIST); break;
    case proto::DataType::DECIMAL32: dtype = cudf::data_type(cudf::type_id::DECIMAL32); break;
    case proto::DataType::DECIMAL64: dtype = cudf::data_type(cudf::type_id::DECIMAL64); break;
    case proto::DataType::DECIMAL128: dtype = cudf::data_type(cudf::type_id::DECIMAL128); break;
    case proto::DataType::STRUCT: dtype = cudf::data_type(cudf::type_id::STRUCT); break;
    default: throw std::runtime_error("Unsupported data type");
  }
  return dtype;
}

/**
 * Generate a proto::Column from a cudf::column_view
 */
std::unique_ptr<cudf::column> nvshmem_task_migration_client::from_proto(
  proto::Column const& proto_col, int32_t source_rank)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto child_idx = 0; child_idx < proto_col.children_size(); ++child_idx) {
    children.push_back(from_proto(proto_col.children(child_idx), source_rank));
  }
  rmm::device_buffer data_buffer{};
  rmm::device_buffer null_mask_buffer{};
  if (proto_col.data_bytes() > 0) {
    data_buffer = rmm::device_buffer(proto_col.data_bytes(), cudf::get_default_stream());
    migrate_buffer(data_buffer.data(),
                   reinterpret_cast<void*>(proto_col.data_location()),
                   proto_col.data_bytes(),
                   source_rank);
  }
  if (proto_col.null_count() > 0) {
    auto buffer_size = proto_col.size() * sizeof(cudf::bitmask_type);
    null_mask_buffer = rmm::device_buffer(buffer_size, cudf::get_default_stream());
    migrate_buffer(null_mask_buffer.data(),
                   reinterpret_cast<void*>(proto_col.null_mask_location()),
                   buffer_size,
                   source_rank);
  }
  auto col = std::make_unique<cudf::column>(from_proto(proto_col.type()),
                                            proto_col.size(),
                                            std::move(data_buffer),
                                            std::move(null_mask_buffer),
                                            proto_col.null_count(),
                                            std::move(children));
  return col;
}

/**
 * Generate a proto::Table from a cudf::table_view
 */
std::unique_ptr<cudf::table> nvshmem_task_migration_client::from_proto(
  proto::Table const& proto_table, int32_t source_rank)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto col_idx = 0; col_idx < proto_table.columns_size(); ++col_idx) {
    columns.push_back(from_proto(proto_table.columns(col_idx), source_rank));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void* nvshmem_task_migration_client::get_translated_ptr(const void* ptr, const int32_t source_rank)
{
  auto self_ptr_iter   = _pgas_base_ptrs.find(_comm->rank());
  auto source_ptr_iter = _pgas_base_ptrs.find(source_rank);
  if (self_ptr_iter == _pgas_base_ptrs.end()) {
    throw std::logic_error("No pointer to PGAS resource for rank " + std::to_string(_comm->rank()));
  }
  if (source_ptr_iter == _pgas_base_ptrs.end()) {
    throw std::logic_error("No pointer to PGAS resource for rank " + std::to_string(source_rank));
  }
  return static_cast<std::byte*>(self_ptr_iter->second) +
         (reinterpret_cast<std::uintptr_t>(ptr) -
          reinterpret_cast<std::uintptr_t>(source_ptr_iter->second));
}

void nvshmem_task_migration_client::migrate_buffer(void* local_buffer,
                                                   void* remote_ptr,
                                                   size_t size,
                                                   int32_t source_rank)
{
  nvshmemx_get8_nbi_on_stream(static_cast<std::byte*>(local_buffer),
                              get_translated_ptr(remote_ptr, source_rank),
                              size,
                              source_rank,
                              cudf::get_default_stream());
  nvshmemx_quiet_on_stream(cudf::get_default_stream());
}

}  // namespace gqe
