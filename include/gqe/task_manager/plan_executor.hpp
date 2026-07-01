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
#include <gqe/physical/relation.hpp>
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
   * @param[in] plan        The deserialized physical plan to execute.
   * @param[in] query_id    Unique identifier for this query.
   * @param[in] descriptors Storage descriptors for the tables referenced by the plan.
   * @param[in] opt_params  Optimization parameters forwarded from the node manager for this query.
   * @return The execution result, containing Arrow IPC bytes for SELECT queries.
   */
  service::execution_result operator()(std::shared_ptr<physical::relation> plan,
                                       utility::uuid query_id,
                                       std::vector<storage::descriptor> descriptors,
                                       gqe::optimization_parameters opt_params);

 private:
  task_manager_context* _ctx;
  remote_table_cache _cache;
  int _rank;
  bool _multi_process;
};

}  // namespace gqe::task_manager
