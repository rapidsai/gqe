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
#include <gqe/qep/state.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/thread_entry.hpp>

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

namespace {

/// Unlimited: query results and migrated tables are full serialized tables,
/// which exceed gRPC's 4 MiB receive default.
constexpr int max_receive_message_size = -1;
constexpr int max_send_message_size    = -1;

}  // namespace

void apply_default_server_args(grpc::ServerBuilder& builder)
{
  // Fail-fast on bind when a stale listener still holds the port.
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  // Intentionally applied to every internal server: they only talk to other GQE processes, so
  // gRPC's default ping-abuse protection isn't needed.
  // Accept the keepalive pings that clients send every `grpc_keepalive_time_ms`, including on
  // idle connections. gRPC's server defaults allow one ping per 5 minutes without data and none
  // without a call, so a quiet log stream collects ping strikes and gets a "too_many_pings" GOAWAY.
  // Must stay in sync with the client-side GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS in
  // `grpc_log_sink`.
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
                             grpc_min_recv_ping_interval_ms);
  builder.SetMaxReceiveMessageSize(max_receive_message_size);
  builder.SetMaxSendMessageSize(max_send_message_size);
}

void apply_default_channel_args(grpc::ChannelArguments& args)
{
  args.SetMaxReceiveMessageSize(max_receive_message_size);
  args.SetMaxSendMessageSize(max_send_message_size);
}

rpc_server::rpc_server(std::vector<grpc::Service*> services)
{
  grpc::ServerBuilder builder;
  apply_default_server_args(builder);
  for (auto& service : services) {
    builder.RegisterService(service);
  }
  int port;
  // FIXME: Currently we only support intra node communication. The server address should be a input
  // parameter.
  server_address = std::string(grpc_localhost_addr) + ":0";
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port);
  server = builder.BuildAndStart();
  if (server == nullptr) { throw std::runtime_error("Failed to start RPC server"); }
  GQE_LOG_INFO("RPC server listening on port {}", port);
  server_address = std::string(grpc_localhost_addr) + ":" + std::to_string(port);
}

grpc::Status task_migration_service::FetchTaskMetadata(grpc::ServerContext* context,
                                                       const ::proto::TaskId* task_id,
                                                       ::proto::TaskMetadata* task_metadata)
{
  // `to_proto` does GPU work, and gRPC runs this handler on threads that GQE did not start. A
  // migration service serves one device, so each thread enters it once.
  thread_local bool is_device_entered = false;
  if (!is_device_entered) {
    utility::enter_device_thread(_device_id);
    is_device_entered = true;
  }

  std::shared_lock<std::shared_mutex> lock(_task_map_mutex);
  auto query_it = _task_map.find(rpc::deserialize_uuid(task_id->query_id()));
  if (query_it == _task_map.end()) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Query not found");
  }
  auto task_it = query_it->second.find(task_id->id());
  if (task_it == query_it->second.end()) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Task not found");
  }

  // Task result must be kept alive till all requests for this task in the task graph are satisfied
  assert(task_it->second != nullptr);

  *task_metadata = to_proto(*task_it->second);
  return grpc::Status::OK;
}

void task_migration_service::register_task(task* t)
{
  std::unique_lock<std::shared_mutex> lock(_task_map_mutex);
  _task_map[t->query_id()][std::to_string(t->task_id())] = t;
}

void task_migration_service::unregister_task(task const& t)
{
  std::unique_lock<std::shared_mutex> lock(_task_map_mutex);
  auto query_it = _task_map.find(t.query_id());
  if (query_it != _task_map.end()) { query_it->second.erase(std::to_string(t.task_id())); }
}

void task_migration_service::unregister_query(utility::uuid query_id)
{
  std::unique_lock<std::shared_mutex> lock(_task_map_mutex);
  _task_map.erase(query_id);
}

/**
 * Generate a ::proto::Column from a cudf::column_view
 */
::proto::Column task_migration_service::to_proto(cudf::column_view const& col)
{
  ::proto::Column proto_col;
  *proto_col.mutable_type() = rpc::serialize_data_type(col.type());
  proto_col.set_size(col.size());
  proto_col.set_null_count(col.null_count());

  if (cudf::is_fixed_width(col.type())) {
    proto_col.set_data_bytes(col.size() * cudf::size_of(col.type()));
  } else if (col.type().id() == cudf::type_id::STRING) {
    rmm::cuda_stream temp_stream;
    proto_col.set_data_bytes(cudf::strings_column_view(col).chars_size(temp_stream));
  } else if (col.type().id() == cudf::type_id::LIST || col.type().id() == cudf::type_id::STRUCT) {
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
 * Generate a ::proto::Table from a cudf::table_view
 */
::proto::Table task_migration_service::to_proto(cudf::table_view const& table)
{
  ::proto::Table proto_table;
  proto_table.set_num_rows(table.num_rows());
  for (auto col = table.begin(); col != table.end(); ++col) {
    proto_table.add_columns()->CopyFrom(to_proto(*col));
  }
  return proto_table;
}

::proto::TaskMetadata task_migration_service::to_proto(task const& t)
{
  ::proto::TaskMetadata proto_task_metadata;
  auto status = t._status.load();
  proto_task_metadata.set_status(status);

  ::proto::TaskId* proto_task_id = proto_task_metadata.mutable_id();
  proto_task_id->set_id(std::to_string(t._task_id));
  *proto_task_id->mutable_query_id() = rpc::serialize_uuid(t.query_id());

  ::proto::Result* proto_result = proto_task_metadata.mutable_result();

  if (status == task::status_type::finished) {
    std::visit(utility::overloaded{
                 [&](const result_kind::owned& result) {
                   proto_result->mutable_table()->CopyFrom(to_proto(result.table->view()));
                 },
                 [&](const result_kind::borrowed& result) {
                   proto_result->mutable_table()->CopyFrom(to_proto(result.view));
                 },
                 [&](const result_kind::qep_state& result) {
                   auto view = qep::to_table_view(result.container);
                   GQE_EXPECTS(
                     view.has_value(),
                     "task_migration: qep_state container is not convertible to a table_view for "
                     "serialization");
                   proto_result->mutable_table()->CopyFrom(to_proto(*view));
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
                                                             std::span<peer_info const> peers)
  : _comm(comm)
{
  grpc::ChannelArguments args;
  apply_default_channel_args(args);

  for (auto const& peer : peers) {
    _stubs.insert({peer.rank,
                   ::proto::TaskMigration::NewStub(grpc::CreateCustomChannel(
                     peer.grpc_address, grpc::InsecureChannelCredentials(), args))});
    _pgas_base_ptrs.insert({peer.rank, peer.pgas_base_ptr});
  }
}

void nvshmem_task_migration_client::migrate_result(gqe::task* t,
                                                   std::unordered_set<int32_t> candidate_ranks)
{
  ::proto::TaskMetadata task_metadata;
  task_metadata.set_status(::proto::TaskStatus::not_started);

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
    if (task_metadata.status() == ::proto::TaskStatus::failed) {
      GQE_LOG_ERROR(
        "Cannot migrate result for task {} from rank {}, task failed", t->task_id(), rank);
      continue;
    } else if (task_metadata.status() == ::proto::TaskStatus::finished) {
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

::proto::TaskMetadata nvshmem_task_migration_client::FetchTaskMetadata(
  ::proto::TaskMigration::Stub* stub, gqe::task const* t)
{
  ::proto::TaskMetadata task_metadata;
  ::proto::TaskId task_id;
  task_id.set_id(std::to_string(t->task_id()));
  *task_id.mutable_query_id() = rpc::serialize_uuid(t->query_id());

  grpc::ClientContext context;
  grpc::Status status = stub->FetchTaskMetadata(&context, task_id, &task_metadata);
  if (!status.ok()) {
    throw std::runtime_error(
      "Failed to fetch task metadata. RPC error code: " + std::to_string(status.error_code()) +
      ", RPC error message: " + status.error_message());
  }
  return task_metadata;
}

/**
 * Generate a ::proto::Column from a cudf::column_view
 */
std::unique_ptr<cudf::column> nvshmem_task_migration_client::from_proto(
  ::proto::Column const& proto_col, int32_t source_rank)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto child_idx = 0; child_idx < proto_col.children_size(); ++child_idx) {
    children.push_back(from_proto(proto_col.children(child_idx), source_rank));
  }
  rmm::device_buffer data_buffer{};
  rmm::device_buffer null_mask_buffer{};
  if (proto_col.data_bytes() > 0) {
    data_buffer = rmm::device_buffer(proto_col.data_bytes(), cudf::get_default_stream().get());
    migrate_buffer(data_buffer.data(),
                   reinterpret_cast<void*>(proto_col.data_location()),
                   proto_col.data_bytes(),
                   source_rank);
  }
  if (proto_col.null_count() > 0) {
    auto buffer_size = proto_col.size() * sizeof(cudf::bitmask_type);
    null_mask_buffer = rmm::device_buffer(buffer_size, cudf::get_default_stream().get());
    migrate_buffer(null_mask_buffer.data(),
                   reinterpret_cast<void*>(proto_col.null_mask_location()),
                   buffer_size,
                   source_rank);
  }
  auto col = std::make_unique<cudf::column>(rpc::deserialize_data_type(proto_col.type()),
                                            proto_col.size(),
                                            std::move(data_buffer),
                                            std::move(null_mask_buffer),
                                            proto_col.null_count(),
                                            std::move(children));
  return col;
}

/**
 * Generate a ::proto::Table from a cudf::table_view
 */
std::unique_ptr<cudf::table> nvshmem_task_migration_client::from_proto(
  ::proto::Table const& proto_table, int32_t source_rank)
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
                              cudf::get_default_stream().get());
  nvshmemx_quiet_on_stream(cudf::get_default_stream().get());
}

}  // namespace gqe
