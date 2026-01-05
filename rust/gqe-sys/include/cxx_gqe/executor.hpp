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

#pragma once

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>

#include <cxx_gqe/api.hpp>
#include <cxx_gqe/physical.hpp>
#include <cxx_gqe/query_context.hpp>
#include <cxx_gqe/task_manager_context.hpp>
#include <cxx_gqe/types.hpp>

#include <memory>
#include <utility>

namespace cxx_gqe {
/*
 * @brief Directly exposes task graph as an opaque type to Rust.
 */
using task_graph = gqe::task_graph;

/*
 * @brief Returns a new optimization parameters instance.
 */
cxx_gqe::optimization_parameters new_optimization_parameters();

/*
 * @brief Converts Rust OptimizationParameters to C++ `gqe::optimization_parameters`.
 */
gqe::optimization_parameters to_gqe(cxx_gqe::optimization_parameters const&);

/*
 * @brief Task graph builder wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * task graph builder.
 */
class task_graph_builder {
 public:
  explicit task_graph_builder(gqe::task_graph_builder&& builder) : _builder(std::move(builder)) {}

  task_graph_builder()                                     = delete;
  task_graph_builder(task_graph_builder const&)            = delete;
  task_graph_builder& operator=(task_graph_builder const&) = delete;

  std::unique_ptr<task_graph> build(std::shared_ptr<physical_relation> root_relation);

 private:
  gqe::task_graph_builder _builder;
};

/*
 * @brief Returns a new task graph builder wrapper.
 */
std::unique_ptr<task_graph_builder> new_task_graph_builder(
  task_manager_context& task_manager_context, query_context& query_context, catalog& catalog);

/*
 * @brief Execute task graph wrapper function.
 */
void execute_task_graph_single_gpu(task_manager_context& task_manager_context,
                                   query_context& query_context,
                                   task_graph const& task_graph);

}  // namespace cxx_gqe
