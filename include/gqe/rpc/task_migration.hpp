/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <gqe/communicator.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <grpcpp/grpcpp.h>
#include <proto/task.grpc.pb.h>
#include <proto/task.pb.h>
#include <rmm/cuda_stream.hpp>
#include <shared_mutex>
#include <unordered_map>
namespace gqe {

/**
 * @brief RPC server
 *
 * @warning The services registered with the server must outlive the lifetime of this object.
 */
struct rpc_server {
  /**
   * @brief Construct a new rpc_server object
   *
   * @param[in] services Services to register with the server
   */
  rpc_server(std::vector<grpc::Service*> services);

  /**
   * @brief Get the server address
   *
   * @return std::string Server address as ip:port combination
   */
  std::string get_server_address() const { return server_address; }

 private:
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
};

/**
 * @brief RPC service for task migration
 */
class task_migration_service final : public proto::TaskMigration::Service {
 public:
  /**
   * @brief Construct a new task migration service object
   *
   * @param[in] device_id CUDA device id to use for the service
   */
  task_migration_service(rmm::cuda_device_id device_id) : _device_id(device_id) {}

  /**
   * @brief gRPC method to fetch task metadata
   *
   * @param[in] context gRPC server context
   * @param[in] task_id Id of the task to fetch
   * @param[out] task_metadata Serialized task metadata
   * @return grpc::Status gRPC status
   */
  grpc::Status FetchTaskMetadata(grpc::ServerContext* context,
                                 const proto::TaskId* task_id,
                                 proto::TaskMetadata* task_metadata) override;
  /**
   * @brief Register a task with the service. The task result must outlive all requests to
   * FetchTaskMetadata.
   *
   * @param[in] t Task to register
   */
  void register_task(task* t);

  /**
   * @brief Serialize task metadata to protobuf
   *
   * @param[in] t Task to serialize
   * @return proto::TaskMetadata Serialized task metadata
   */
  proto::TaskMetadata to_proto(task const& t);

 private:
  std::shared_mutex
    _task_map_mutex;  // mutex to allow multiple readers and one writer access to the task map
  std::unordered_map<std::string, task*> _task_map;  // map of task id to task

  /**
   * @brief Convert cudf::data_type to  proto::DataType
   *
   * @param[in] dtype cudf::data_type to convert
   * @return proto::DataType Converted protobuf data type
   */
  proto::DataType to_proto(cudf::data_type const& dtype);

  /**
   * @brief Convert cudf::column_view to proto::Column
   *
   * @param[in] col cudf::column_view to convert
   * @return proto::Column Converted protobuf column
   */
  proto::Column to_proto(cudf::column_view const& col);

  /**
   * @brief Convert cudf::table_view to proto::Table
   *
   * @param[in] table cudf::table_view to convert
   * @return proto::Table Converted protobuf table
   */
  proto::Table to_proto(cudf::table_view const& table);
  rmm::cuda_device_id _device_id;
};

/**
 * @brief Abstract class for task migration client
 */
class task_migration_client {
 public:
  virtual ~task_migration_client() = default;
  /**
   * @brief Migrate the result of a task from any of the given candidate ranks to the current rank
   *
   * @param[in] t Task to migrate
   * @param[in] candidate_ranks Candidate ranks to migrate from
   */
  virtual void migrate_result(gqe::task* t, std::unordered_set<int32_t> candidate_ranks) = 0;

 protected:
  /**
   * @brief Set the owned result of a task
   *
   * @note This function is required to allow inheriting classes to access members of task
   *
   * @param[in] t Task to set the result for
   * @param[in] result Result to set
   */
  void set_owned_result(gqe::task* t, std::unique_ptr<cudf::table> result);
  /**
   * @brief Set the borrowed result of a task
   *
   * @note This function is required to allow inheriting classes to access members of task
   *
   * @param[in] t Task to set the result for
   * @param[in] result Result to set
   */
  void set_borrowed_result(gqe::task* t, cudf::table_view result);
};

/**
 * @brief Task migration client to be used with NVSHMEM
 */
class nvshmem_task_migration_client : public task_migration_client {
 public:
  /**
   * @brief Construct a new nvshmem_task_migration_client object
   *
   * @param[in] comm NVSHMEM communicator
   * @param[in] local_server Local RPC server
   * @param[in] pgas_base_ptr Pointer to the PGAS base pointer on the current rank
   */
  nvshmem_task_migration_client(nvshmem_communicator* comm,
                                rpc_server const& local_server,
                                void* pgas_base_ptr);
  void migrate_result(gqe::task* t, std::unordered_set<int32_t> candidate_ranks) override;

 private:
  std::unordered_map<int32_t, std::unique_ptr<proto::TaskMigration::Stub>>
    _stubs;                                            // map of rank to gRPC stub
  std::unordered_map<int32_t, void*> _pgas_base_ptrs;  // map of rank to PGAS base pointer
  nvshmem_communicator* _comm;

  /**
   * @brief gRPC function to fetch task metadata
   *
   * @param[in] stub gRPC stub to use for the request
   * @param[in] t Task to fetch metadata for
   * @return proto::TaskMetadata Task metadata
   */
  proto::TaskMetadata FetchTaskMetadata(proto::TaskMigration::Stub* stub, gqe::task const* t);
  /**
   * @brief Convert a proto::Table to a cudf::table
   *
   * @param[in] proto_table Protobuf table to convert
   * @param[in] source_rank Source rank for the table
   * @return std::unique_ptr<cudf::table> Converted cudf::table
   */
  std::unique_ptr<cudf::table> from_proto(proto::Table const& proto_table, int32_t source_rank);
  /**
   * @brief Convert a proto::Column to a cudf::column
   *
   * @param[in] proto_col Protobuf column to convert
   * @param[in] source_rank Source rank for the column
   * @return std::unique_ptr<cudf::column> Converted cudf::column
   */
  std::unique_ptr<cudf::column> from_proto(proto::Column const& proto_col, int32_t source_rank);

  /**
   * @brief Convert a proto::DataType to a cudf::data_type
   *
   * @param[in] proto_dtype Protobuf data type to convert
   * @return cudf::data_type Converted cudf::data_type
   */
  cudf::data_type from_proto(proto::DataType proto_dtype);
  /**
   * @brief Translate a pointer from a given rank to that on the current rank
   *
   * @param[in] ptr Pointer on the given rank
   * @param[in] source_rank Source rank for ptr translation
   * @return void* Translated pointer
   */
  void* get_translated_ptr(const void* ptr, const int32_t source_rank);
  /**
   * @brief Migrate a buffer from a given rank to the current rank
   *
   * @param[in] local_buffer Buffer to migrate
   * @param[in] remote_ptr Pointer to the buffer on the remote rank
   * @param[in] size Size of the buffer
   * @param[in] source_rank Source rank of the buffer
   */
  void migrate_buffer(void* local_buffer, void* remote_ptr, size_t size, int32_t source_rank);
};

}  // namespace gqe
