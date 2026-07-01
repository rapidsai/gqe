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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/numa_pool_memory_resource.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/shared_numa_pool_memory_resource.hpp>
#include <gqe/memory_resource/shared_resource_adaptor.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda/memory_resource>
#include <cudf/types.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/traits.hpp>
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <rmm/aligned.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <memory>
#include <optional>

namespace gqe {

namespace {
using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

}  // namespace

task_manager_context::task_manager_context(optimization_parameters params,
                                           cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _optimization_parameters(std::move(params)), _mr(std::move(mr))
{
  _table_memory_resource_latch = std::make_unique<std::mutex>();
  rmm::mr::set_current_device_resource(_mr);
}

rmm::device_async_resource_ref task_manager_context::get_table_memory_resource(
  memory_kind::type const& kind)
{
  std::lock_guard<std::mutex> guard(*_table_memory_resource_latch);

  auto it = _table_memory_resources.find(kind);
  if (it != _table_memory_resources.end()) { return it->second; }

  return create_table_memory_resource_unsafe(kind);
}

[[nodiscard]] memory_resource::boost_shared_memory_resource*
task_manager_context::get_boost_shared_resource()
{
  std::lock_guard<std::mutex> guard(*_table_memory_resource_latch);
  return _boost_shared_resource.get();
}

rmm::device_async_resource_ref task_manager_context::create_table_memory_resource_unsafe(
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
  const auto resolve_max_pool_bytes =
    [max_pool_bytes, max_pool_percentage](const cpu_set& numa_node_set) -> std::size_t {
    if (max_pool_bytes) { return max_pool_bytes.value(); }
    auto numa_memory_capacity = memory_resource::available_numa_node_memory(numa_node_set).second;
    return memory_resource::percent_of_memory(numa_memory_capacity, max_pool_percentage);
  };

  // Local handle to the side-channel `_boost_shared_resource` so the visitor lambda can write to
  // it without capturing `this`.
  auto& boost_shared_slot = _boost_shared_resource;

  // Create the memory resource lazily
  auto resource = std::visit(
    utility::overloaded{
      [](const memory_kind::system&) -> any_device_resource {
        GQE_LOG_DEBUG("Creating system memory resource");
        return any_device_resource{memory_resource::system_memory_resource{}};
      },
      [initial_pool_bytes,
       resolve_max_pool_bytes](const memory_kind::numa& numa) -> any_device_resource {
        std::size_t max_pool_bytes_set = resolve_max_pool_bytes(numa.numa_node_set);
        GQE_LOG_DEBUG("Task manager creating `numa` memory resource on NUMA nodes: {}",
                      numa.numa_node_set.pretty_print());

        // `numa_memory_resource` is stateless (configuration only; mmap state lives in
        // the per-allocation buffer), so we hand it to the pool by value.
        // Task-manager resources are type-erased as `device_accessible`; therefore use the
        // device-accessible NUMA specialization here.
        memory_resource::numa_device_accessible_resource numa_mr{numa.numa_node_set,
                                                                 numa.page_kind};
        GQE_LOG_DEBUG("Creating numa pool with size {} bytes", initial_pool_bytes);
        return any_device_resource{
          rmm::mr::pool_memory_resource{any_device_resource{std::move(numa_mr)},
                                        initial_pool_bytes,
                                        std::make_optional(max_pool_bytes_set)}};
      },
      [](const memory_kind::pinned&) -> any_device_resource {
        GQE_LOG_DEBUG("Creating pinned memory resource");
        return any_device_resource{memory_resource::pinned_memory_resource{}};
      },
      [initial_pool_bytes,
       resolve_max_pool_bytes](const memory_kind::numa_pinned& numa_pinned) -> any_device_resource {
        std::size_t max_pool_bytes_set = resolve_max_pool_bytes(numa_pinned.numa_node_set);
        GQE_LOG_DEBUG("Task manager creating `numa_pinned` memory resource on NUMA nodes: {}",
                      numa_pinned.numa_node_set.pretty_print());

        memory_resource::numa_device_accessible_resource numa_mr{numa_pinned.numa_node_set,
                                                                 numa_pinned.page_kind};
        GQE_LOG_DEBUG("Creating numa_pinned pool with size {} bytes", initial_pool_bytes);
        return any_device_resource{
          rmm::mr::pool_memory_resource{any_device_resource{std::move(numa_mr)},
                                        initial_pool_bytes,
                                        std::make_optional(max_pool_bytes_set)}};
      },
      [](const memory_kind::device&) -> any_device_resource {
        // FIXME: specify device instead of allocating on default CUDA device
        GQE_LOG_DEBUG("Creating device memory resource");
        return any_device_resource{rmm::mr::cuda_memory_resource{}};
      },
      [](const memory_kind::managed&) -> any_device_resource {
        GQE_LOG_DEBUG("Creating managed memory resource");
        return any_device_resource{rmm::mr::managed_memory_resource{}};
      },
      [&boost_shared_slot](const memory_kind::boost_shared&) -> any_device_resource {
        GQE_LOG_DEBUG("Creating boost_shared memory resource");
        // Stash a typed `shared_ptr` so `get_boost_shared_resource()` can return it; the same
        // resource is also handed to the type-erased map below via `shared_resource_adaptor`.
        boost_shared_slot = std::make_shared<memory_resource::boost_shared_memory_resource>();
        return any_device_resource{memory_resource::shared_resource_adaptor{boost_shared_slot}};
      },
      [initial_pool_bytes,
       resolve_max_pool_bytes](const memory_kind::numa_pool& numa_pool) -> any_device_resource {
        std::size_t max_pool_bytes_set = resolve_max_pool_bytes(cpu_set{numa_pool.numa_node_id});
        GQE_LOG_DEBUG("Creating numa_pool memory resource");
        // Use the same default-cap policy as other NUMA-backed task manager resources.
        auto numa_pool_mr = std::make_shared<memory_resource::numa_pool_memory_resource>(
          numa_pool.numa_node_id, initial_pool_bytes, std::make_optional(max_pool_bytes_set));
        return any_device_resource{
          memory_resource::shared_resource_adaptor{std::move(numa_pool_mr)}};
      },
      [initial_pool_bytes, resolve_max_pool_bytes](
        const memory_kind::shared_numa_pool& shared_numa_pool) -> any_device_resource {
        std::size_t max_pool_bytes_set =
          resolve_max_pool_bytes(cpu_set{shared_numa_pool.numa_node_id});
        GQE_LOG_DEBUG("Creating shared_numa_pool memory resource");
        auto shared_numa_pool_mr = std::make_shared<memory_resource::shared_numa_pool_resource>(
          shared_numa_pool.numa_node_id,
          initial_pool_bytes,
          std::make_optional(max_pool_bytes_set));
        return any_device_resource{
          memory_resource::shared_resource_adaptor{std::move(shared_numa_pool_mr)}};
      }},
    kind);

  auto [it, _] = _table_memory_resources.emplace(kind, std::move(resource));
  return it->second;
}

task_manager_context::~task_manager_context() { rmm::mr::reset_current_device_resource(); }

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
  // `pgas_memory_resource` owns NVSHMEM symmetric state and is non-copyable, so it has to be
  // wrapped in `shared_resource_adaptor` to be type-erased into a `cuda::mr::any_resource`.
  // The pool then takes the resulting upstream by value and keeps it alive for as long as
  // the pool lives.
  auto const pool_bytes = upstream_mr->get_bytes();
  std::shared_ptr<gqe::pgas_memory_resource> shared_upstream{std::move(upstream_mr)};
  auto pgas_upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
    memory_resource::shared_resource_adaptor{std::move(shared_upstream)}};
  _mr = cuda::mr::any_resource<cuda::mr::device_accessible>{
    rmm::mr::pool_memory_resource{std::move(pgas_upstream), pool_bytes, pool_bytes}};
  rmm::mr::set_current_device_resource(_mr);
}

std::unique_ptr<multi_process_task_manager_context>
multi_process_task_manager_context::default_init(int rank,
                                                 int nranks,
                                                 nvshmemx_uniqueid_t const& uid,
                                                 std::span<peer_info const> peers,
                                                 gqe::SCHEDULER_TYPE type,
                                                 optimization_parameters params)
{
  auto comm = std::make_unique<nvshmem_communicator>(rank, nranks, uid);
  comm->init();

  // When running in multi-GPU mode we cannot grow the pool since symmetric memory allocation has to
  // be upfront. Therefore we ignore the GQE_INITIAL_QUERY_MEMORY parameter.
  bool const pool_size_auto_derived = !params.max_query_memory.has_value();
  auto pool_size = params.max_query_memory.value_or(detail::default_device_memory_pool_size());
  if (comm->num_ranks_per_device() > 1) {
    GQE_LOG_WARN("Node process count {} >= number of GPUs {}. Using MPG mode for NVSHMEM",
                 comm->world_size(),
                 comm->num_ranks_per_device());
    pool_size = rmm::align_down(pool_size / comm->num_ranks_per_device(), 256);
  }

  // Pool size must be the same on all ranks. The node manager is responsible for
  // passing a consistent value or the caller can pre-negotiate.  We use the local
  // value directly since MPI is no longer available for collective negotiation.
  GQE_LOG_INFO("Setting pool size to {} bytes ({:.2f} MiB)",
               pool_size,
               static_cast<double>(pool_size) / (1024.0 * 1024.0));
  if (pool_size_auto_derived) {
    GQE_LOG_INFO(
      "Pool size was auto-derived from 90% of free device memory and is non-deterministic "
      "across runs. If you hit RMM pool exhaustion during data loading or query execution, "
      "set GQE_MAX_QUERY_MEMORY=<bytes> in the shell that launches gqe_node_manager to pin "
      "the pool to a known-good size.");
  }

  auto pgas_mr = std::make_unique<gqe::pgas_memory_resource>(pool_size);

  // Force initialization of the pinned memory resource to avoid the first allocation happening
  // during execution.
  std::ignore = cudf::get_pinned_memory_resource();

  std::unique_ptr<gqe::scheduler> scheduler;
  if (type == gqe::SCHEDULER_TYPE::ROUND_ROBIN) {
    scheduler = std::make_unique<round_robin_scheduler>(comm->world_size());
  } else if (type == gqe::SCHEDULER_TYPE::ALL_TO_ALL) {
    scheduler = std::make_unique<all_to_all_scheduler>(comm->world_size());
  }

  auto migration_service = std::make_unique<task_migration_service>(comm->device_id());
  auto server            = rpc_server(std::vector<grpc::Service*>{migration_service.get()});

  auto migration_client = std::make_unique<nvshmem_task_migration_client>(comm.get(), peers);
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

void* multi_process_task_manager_context::pgas_base_ptr() const
{
  return _upstream_pgas_mr->get_local_base_ptr();
}

std::string multi_process_task_manager_context::migration_grpc_address() const
{
  return _rpc_server.get_server_address();
}

void multi_process_task_manager_context::finalize()
{
  GQE_CUDA_TRY(cudaDeviceSynchronize());
  _upstream_pgas_mr->finalize();
  comm->finalize();
}

}  // namespace gqe
