/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
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
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

namespace gqe {
task_manager_context::task_manager_context(std::unique_ptr<rmm::mr::device_memory_resource> mr)
{
  _mr = std::move(mr);

  rmm::mr::set_current_device_resource(_mr.get());
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
  std::unique_ptr<gqe::pgas_memory_resource> upstream_mr)
  : task_manager_context(),
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
multi_process_task_manager_context::default_init(MPI_Comm mpi_comm, gqe::SCHEDULER_TYPE type)
{
  auto comm = std::make_unique<nvshmem_communicator>(mpi_comm);
  comm->init();

  auto pool_size = gqe::utility::default_device_memory_pool_size();

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
