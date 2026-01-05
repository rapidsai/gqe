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

#include <gqe/task_manager_context.hpp>

#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <rmm/aligned.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

namespace gqe {
task_manager_context::task_manager_context(optimization_parameters params,
                                           std::unique_ptr<rmm::mr::device_memory_resource> mr)
  : _optimization_parameters(std::move(params)), _mr(std::move(mr))
{
  _table_memory_resource_latch = std::make_unique<std::mutex>();
  rmm::mr::set_current_device_resource(_mr.get());
}

rmm::device_async_resource_ref task_manager_context::get_table_memory_resource(
  memory_kind::type const& kind)
{
  std::lock_guard<std::mutex> guard(*_table_memory_resource_latch);

  auto it = _table_memory_resources.find(kind);
  if (it != _table_memory_resources.end()) { return *it->second; }

  return *create_table_memory_resource_unsafe(kind);
}

rmm::mr::device_memory_resource* task_manager_context::get_table_memory_resource_ptr(
  memory_kind::type const& kind)
{
  std::lock_guard<std::mutex> guard(*_table_memory_resource_latch);

  auto it = _table_memory_resources.find(kind);
  if (it != _table_memory_resources.end()) { return it->second.get(); }

  return create_table_memory_resource_unsafe(kind);
}

rmm::mr::device_memory_resource* task_manager_context::create_table_memory_resource_unsafe(
  memory_kind::type const& kind)
{
  // 85% is default, because 90% typically fails on Grace with 480 GB. However, 85% is enough for
  // TPC-H SF1k.
  const auto max_pool_percentage = 85;
  const auto initial_pool_bytes  = _optimization_parameters.initial_task_manager_memory;
  const auto max_pool_bytes =
    _optimization_parameters.max_task_manager_memory == std::numeric_limits<std::size_t>::max()
      ? std::nullopt
      : std::make_optional(_optimization_parameters.max_task_manager_memory);

  // Create the memory resource lazily
  auto resource = std::visit(
    utility::overloaded{
      [](const memory_kind::system&) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        GQE_LOG_DEBUG("Creating system memory resource");
        return std::make_unique<memory_resource::system_memory_resource>();
      },
      [initial_pool_bytes, max_pool_bytes](
        const memory_kind::numa& numa) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        std::size_t max_pool_bytes_set = 0;
        if (!max_pool_bytes) {
          auto numa_memory_capacity =
            memory_resource::available_numa_node_memory(numa.numa_node_set).second;
          max_pool_bytes_set =
            memory_resource::percent_of_memory(numa_memory_capacity, max_pool_percentage);
        } else {
          max_pool_bytes_set = max_pool_bytes.value();
        }

        using upstream_mr = memory_resource::numa_memory_resource;
        using pool_mr     = rmm::mr::pool_memory_resource<upstream_mr>;
        using wrapper_mr  = rmm::mr::owning_wrapper<pool_mr, upstream_mr>;

        auto upstream = std::make_unique<upstream_mr>(numa.numa_node_set, numa.page_kind, false);
        GQE_LOG_DEBUG("Creating numa pool with size {} bytes", initial_pool_bytes);
        return std::make_unique<wrapper_mr>(
          std::move(upstream), initial_pool_bytes, std::make_optional(max_pool_bytes_set));
      },
      [](const memory_kind::pinned&) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        GQE_LOG_DEBUG("Creating pinned memory resource");
        return std::make_unique<memory_resource::pinned_memory_resource>();
      },
      [initial_pool_bytes, max_pool_bytes](const memory_kind::numa_pinned& numa_pinned)
        -> std::unique_ptr<rmm::mr::device_memory_resource> {
        std::size_t max_pool_bytes_set = 0;
        if (!max_pool_bytes) {
          auto numa_memory_capacity =
            memory_resource::available_numa_node_memory(numa_pinned.numa_node_set).second;
          max_pool_bytes_set =
            memory_resource::percent_of_memory(numa_memory_capacity, max_pool_percentage);
        } else {
          max_pool_bytes_set = max_pool_bytes.value();
        }

        using upstream_mr = memory_resource::numa_memory_resource;
        using pool_mr     = rmm::mr::pool_memory_resource<upstream_mr>;
        using wrapper_mr  = rmm::mr::owning_wrapper<pool_mr, upstream_mr>;

        auto upstream =
          std::make_unique<upstream_mr>(numa_pinned.numa_node_set, numa_pinned.page_kind, true);
        GQE_LOG_DEBUG("Creating numa_pinned pool with size {} bytes", initial_pool_bytes);
        return std::make_unique<wrapper_mr>(
          std::move(upstream), initial_pool_bytes, std::make_optional(max_pool_bytes_set));
      },
      [](const memory_kind::device&) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        // FIXME: specify device instead of allocating on default CUDA device
        GQE_LOG_DEBUG("Creating device memory resource");
        return std::make_unique<rmm::mr::cuda_memory_resource>();
      },
      [](const memory_kind::managed&) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        GQE_LOG_DEBUG("Creating managed memory resource");
        return std::make_unique<rmm::mr::managed_memory_resource>();
      },
      [](const memory_kind::boost_shared&) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        GQE_LOG_DEBUG("Creating boost_shared memory resource");
        return std::make_unique<memory_resource::boost_shared_memory_resource>();
      }},
    kind);

  auto ptr                      = resource.get();
  _table_memory_resources[kind] = std::move(resource);
  return ptr;
}

task_manager_context::~task_manager_context()
{
  // We need to call both these APIs because of a bug in RMM.
  // https://github.com/rapidsai/rmm/issues/1953
  rmm::mr::set_current_device_resource(nullptr);
  rmm::mr::reset_current_device_resource_ref();
}

void task_manager_context::finalize() { GQE_CUDA_TRY(cudaDeviceSynchronize()); }

multi_process_task_manager_context::multi_process_task_manager_context(
  std::unique_ptr<gqe::communicator> comm,
  std::unique_ptr<gqe::scheduler> scheduler,
  std::unique_ptr<gqe::task_migration_client> migration_client,
  std::unique_ptr<gqe::task_migration_service> migration_service,
  gqe::rpc_server&& server,
  optimization_parameters params,
  std::unique_ptr<gqe::pgas_memory_resource> upstream_mr)
  : task_manager_context(std::move(params)),
    comm(std::move(comm)),
    scheduler(std::move(scheduler)),
    migration_client(std::move(migration_client)),
    migration_service(std::move(migration_service)),
    _rpc_server(std::move(server)),
    _upstream_pgas_mr(upstream_mr.get())
{
  using upstream_mr_type = gqe::pgas_memory_resource;
  using mr_type          = rmm::mr::pool_memory_resource<upstream_mr_type>;
  _mr                    = std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(
    std::move(upstream_mr), upstream_mr->get_bytes(), upstream_mr->get_bytes());
  rmm::mr::set_current_device_resource(_mr.get());
}

std::unique_ptr<multi_process_task_manager_context>
multi_process_task_manager_context::default_init(MPI_Comm mpi_comm,
                                                 gqe::SCHEDULER_TYPE type,
                                                 optimization_parameters params)
{
  auto comm = std::make_unique<nvshmem_communicator>(mpi_comm);
  comm->init();

  auto pool_size = params.initial_task_manager_memory;
  if (comm->num_ranks_per_device() > 1) {
    GQE_LOG_WARN("Node process count {} >= number of GPUs {}. Using MPG mode for NVSHMEM",
                 comm->world_size(),
                 comm->num_ranks_per_device());
    pool_size = rmm::align_down(pool_size / comm->num_ranks_per_device(), 256);
  }

  // PGAS memory resource has to have the same size on all ranks.
  MPI_Allreduce(&pool_size, &pool_size, 1, MPI_LONG_LONG, MPI_MIN, mpi_comm);
  GQE_LOG_INFO("Setting pool size to {}", pool_size);

  auto pgas_mr = std::make_unique<gqe::pgas_memory_resource>(pool_size);

  std::unique_ptr<gqe::scheduler> scheduler;
  if (type == gqe::SCHEDULER_TYPE::ROUND_ROBIN) {
    scheduler = std::make_unique<round_robin_scheduler>(comm->world_size());
  } else if (type == gqe::SCHEDULER_TYPE::ALL_TO_ALL) {
    scheduler = std::make_unique<all_to_all_scheduler>(comm->world_size());
  }

  auto migration_service = std::make_unique<task_migration_service>(comm->device_id());
  auto server            = rpc_server(std::vector<grpc::Service*>{migration_service.get()});

  std::unique_ptr<nvshmem_task_migration_client> migration_client;

  migration_client = std::make_unique<nvshmem_task_migration_client>(
    comm.get(), server, pgas_mr->get_local_base_ptr());
  return std::make_unique<multi_process_task_manager_context>(std::move(comm),
                                                              std::move(scheduler),
                                                              std::move(migration_client),
                                                              std::move(migration_service),
                                                              std::move(server),
                                                              std::move(params),
                                                              std::move(pgas_mr));
}

void multi_process_task_manager_context::update_scheduler(gqe::SCHEDULER_TYPE type)
{
  if (type == gqe::SCHEDULER_TYPE::ROUND_ROBIN) {
    scheduler = std::make_unique<round_robin_scheduler>(comm->world_size());
  } else if (type == gqe::SCHEDULER_TYPE::ALL_TO_ALL) {
    scheduler = std::make_unique<all_to_all_scheduler>(comm->world_size());
  }
}

void multi_process_task_manager_context::finalize()
{
  GQE_CUDA_TRY(cudaDeviceSynchronize());
  _upstream_pgas_mr->finalize();
  comm->finalize();
}

}  // namespace gqe
