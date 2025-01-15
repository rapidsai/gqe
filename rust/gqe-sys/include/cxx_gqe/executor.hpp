/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cxx_gqe/api.hpp>
#include <cxx_gqe/physical.hpp>
#include <cxx_gqe/query_context.hpp>
#include <cxx_gqe/task_manager_context.hpp>
#include <cxx_gqe/types.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>

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
