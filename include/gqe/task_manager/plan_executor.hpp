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

#pragma once

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/physical/plan.hpp>
#include <gqe/storage/descriptor.hpp>
#include <gqe/task_manager/remote_table_cache.hpp>
#include <gqe/task_manager/service.hpp>

#include <memory>
#include <vector>

namespace gqe {
class task_manager_context;
}

namespace gqe::task_manager {

/**
 * @brief Functor that builds and executes a task graph from a deserialized physical plan.
 *
 * Each call to operator() receives a plan and a set of storage descriptors,
 * updates the remote table cache, builds a task graph, executes it, and
 * returns the result as Arrow IPC bytes.
 *
 * Every execution phase is wrapped in an NVTX range so that per-stage timing can
 * be recorded externally. Each phase sits in its own block so that its range ends
 * at the phase boundary; values used after a phase are declared outside that block.
 * Wrap any phase added here in its own NVTX range.
 *
 * ## Thread Safety
 *
 * Calls are not synchronized and must not overlap. The calling thread must have run
 * `utility::enter_device_thread` for the context's device. The plan executor does not select the
 * device itself.
 */
class plan_executor {
 public:
  /**
   * @brief Construct a new plan executor.
   *
   * @param[in] ctx           The task manager context (single or multi-process).
   * @param[in] rank          This task manager's rank.
   * @param[in] multi_process Whether to use multi-process execution.
   */
  plan_executor(task_manager_context* ctx, int rank, bool multi_process);

  /**
   * @brief Execute a physical plan.
   *
   * @param[in] plan        The physical relation graph and its query-scoped node IDs.
   * @param[in] query_id    Unique identifier for this query.
   * @param[in] descriptors Storage descriptors for the tables referenced by the plan.
   * @param[in] opt_params  Optimization parameters forwarded from the node manager for this query.
   * @return The execution result, containing Arrow IPC bytes for SELECT queries.
   * @throws std::invalid_argument If `qep_task_graph` is selected for multi-process execution or a
   *         write plan, because QEP state migration and write-statistics propagation are not
   *         supported.
   */
  service::execution_result operator()(physical::identified_plan plan,
                                       utility::uuid query_id,
                                       std::vector<storage::descriptor> descriptors,
                                       gqe::optimization_parameters opt_params);

  /**
   * @brief Overwrite the contents of shared_segment_memory statistics tables from projected Arrow
   * data.
   *
   * Updates the remote table cache with the stats tables' descriptors, then for each write
   * deserializes its Arrow IPC payload into a device table and replaces the corresponding table's
   * row group (rank 0 writes shared memory; all ranks update local references). Uses the persistent
   * cache so the row groups written here are visible to a subsequent read of the same table.
   *
   * @param[in] writes      Per-table projected contents (Arrow IPC; empty payload clears the
   * table).
   * @param[in] query_id    Unique identifier for this refresh.
   * @param[in] descriptors Storage descriptors for the statistics tables.
   * @param[in] epoch       Monotonic value mixed into shared-object names.
   */
  void refresh_stats(std::vector<service::stats_table_write> writes,
                     utility::uuid query_id,
                     std::vector<storage::descriptor> descriptors,
                     std::uint64_t epoch);

 private:
  task_manager_context* _ctx;
  remote_table_cache _cache;
  int _rank;
  bool _multi_process;
};

}  // namespace gqe::task_manager
