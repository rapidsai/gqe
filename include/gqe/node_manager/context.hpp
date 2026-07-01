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

#include <gqe/task_manager_context.hpp>

#include <boost/interprocess/shared_memory_object.hpp>

#include <optional>

namespace gqe::node_manager {

/// Name of the boost interprocess shared memory segment used for table storage.
/// Used by the node manager (create), task managers (open), and cleanup handlers.
inline constexpr char shared_memory_name[] = "gqe_shared_memory";

/**
 * @brief Lightweight context for the node manager process.
 *
 * The node manager owns the catalog but does not execute queries on GPU —
 * execution is delegated to task manager subprocesses. This context provides
 * only what the catalog needs (table memory resources for shared memory
 * segments) without allocating a GPU query memory pool, leaving GPU memory
 * available for task managers.
 *
 * TODO: Remove inheritance from task_manager_context. This context only needs
 * table memory resources, but inherits GPU execution state (streams, semaphores)
 * that it doesn't use. The node manager context should own its table memory
 * resources directly, and catalog/in_memory_table should be updated to not
 * require a task_manager_context pointer.
 */
struct context : public gqe::task_manager_context {
  /**
   * @brief Constructs a node manager context with a minimal memory resource.
   *
   * Does not allocate a GPU pool or set the current device resource.
   */
  explicit context(optimization_parameters params = make_optimization_parameters());

 private:
  /**
   * @brief Removes the shared memory segment when this context is destroyed.
   *
   * Stored as optional so it can be constructed after the segment is created.
   */
  std::optional<boost::interprocess::remove_shared_memory_on_destroy> _shm_remover;
};

}  // namespace gqe::node_manager
