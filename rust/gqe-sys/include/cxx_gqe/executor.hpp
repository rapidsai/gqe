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
#include <cxx_gqe/types.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task_graph.hpp>

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
 * @brief Query context wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * query context.
 */
class query_context {
 public:
  explicit query_context(gqe::query_context&& context) : _context(std::move(context)) {}

  query_context()                     = delete;
  query_context(query_context const&) = delete;
  query_context& operator=(query_context const&) = delete;

  /*
   * @brief Returns the C++ query context.
   *
   * This is a helper method used in the C++ bindings to convert from the wrapper to the actual
   * object.
   */
  inline gqe::query_context& get() { return _context; }

 private:
  gqe::query_context _context;
};

/*
 * @brief Returns a new query context wrapper.
 */
std::unique_ptr<query_context> new_query_context(
  cxx_gqe::optimization_parameters const& parameters);

/*
 * @brief Task graph builder wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * task graph builder.
 */
class task_graph_builder {
 public:
  explicit task_graph_builder(gqe::task_graph_builder&& builder) : _builder(std::move(builder)) {}

  task_graph_builder()                          = delete;
  task_graph_builder(task_graph_builder const&) = delete;
  task_graph_builder& operator=(task_graph_builder const&) = delete;

  std::unique_ptr<task_graph> build(std::shared_ptr<physical_relation> root_relation);

 private:
  gqe::task_graph_builder _builder;
};

/*
 * @brief Returns a new task graph builder wrapper.
 */
std::unique_ptr<task_graph_builder> new_task_graph_builder(query_context& query_context,
                                                           catalog& catalog);

/*
 * @brief Execute task graph wrapper function.
 */
void execute_task_graph_single_gpu(query_context& query_context, task_graph const& task_graph);

}  // namespace cxx_gqe
