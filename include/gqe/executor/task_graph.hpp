/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/query_context.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace gqe {

struct task_graph {
  /**
   * @brief Root tasks of the task graph.
   *
   * @note The task graph needs to share ownership of the root tasks so that the tasks in the task
   * graph are not deallocated.
   */
  std::vector<std::shared_ptr<task>> root_tasks;

  /**
   * @brief Root tasks of each stage.
   *
   * Each root task represents the start point of a pipeline. The length of the outer vector is
   * equal to the number of stages.
   */
  std::vector<std::vector<task*>> stage_root_tasks;
};

/**
 * @brief Execute the task graph on a single GPU.
 *
 * After this function call, the result tables of the root tasks in `task_graph_to_execute` are
 * available to the local GPU.
 *
 * @param[in] query_context Context object with resources and optimization
 * parameters for the query. This should usually be the same object as passed to
 * the task graph builder.
 * @param[in] task_graph The task graph to execute.
 */
void execute_task_graph_single_gpu(query_context* query_context, task_graph const* task_graph);

/**
 * @brief A builder for generating a task graph from a physical plan.
 */
class task_graph_builder {
 public:
  /**
   * @brief Construct a task graph builder object.
   *
   * @param[in] query_context Context object containing optimization parameters
   * and used for establishing resources to be used by the query executor.
   * @param[in] catalog Catalog containing file locations and data types of the input tables.
   */
  task_graph_builder(query_context* query_context, catalog const* catalog);

  /**
   * @brief Generate a new task graph.
   *
   * @param[in] root_relation Root relation of the physical plan.
   *
   * @return The generated task graph.
   */
  std::unique_ptr<task_graph> build(physical::relation* root_relation);

 private:
  // Helper function for generate tasks of a physical relation.
  std::vector<std::shared_ptr<task>> generate_tasks(physical::relation* relation);

  // Helper function for concatenate the input tasks
  std::shared_ptr<task> concatenate(std::vector<std::shared_ptr<task>> input_tasks,
                                    bool is_pipeline_breaker = true);

  // A physical relation visitor used for generating a task graph for the physical relation and its
  // descendants.
  //
  // Since the visitor stores the generated tasks in the member variable `_generated_tasks`, the
  // visitor should not be reused.
  struct generate_task_graph_visitor : public physical::relation_visitor {
    generate_task_graph_visitor(task_graph_builder* builder) : _builder(builder) {}

    void visit(physical::read_relation* relation) override;
    void visit(physical::write_relation* relation) override;
    void visit(physical::broadcast_join_relation* relation) override;
    void visit(physical::project_relation* relation) override;
    void visit(physical::concatenate_sort_relation* relation) override;
    void visit(physical::filter_relation* relation) override;
    void visit(physical::concatenate_aggregate_relation* relation) override;
    void visit(physical::fetch_relation* relation) override;
    void visit(physical::window_relation* relation) override;
    void visit(physical::union_all_relation* relation) override;
    void visit(physical::user_defined_relation* relation) override;
    void visit(physical::gen_ident_col_relation* relation) override;

    // Check the task cache in `_builder`. If the relation is found in the cache, the retrieved
    // tasks are copied to `_generated_tasks`, and the function returns true. Otherwise, the
    // function returns false.
    bool is_cached(physical::relation* relation);
    // Update the task cache in `_builder` with _generated_tasks.
    void update_cache(physical::relation* relation);

    task_graph_builder* _builder;
    std::vector<std::shared_ptr<task>> _generated_tasks;
  };

  // `root_tasks` contains the tasks passed to the parent relation
  void insert_pipeline_breaker(std::vector<task*> root_tasks)
  {
    std::vector<task*> current_stage_tasks;
    for (auto const& task : root_tasks) {
      // Checking the stage is necessary because a task from `root_tasks` can belong to a previous
      // stage.
      if (task->stage_id() == _current_stage_id) current_stage_tasks.push_back(task);
    }

    _stage_root_tasks.push_back(std::move(current_stage_tasks));
    _current_stage_id++;
  }

  query_context* _query_context;
  catalog const* _catalog;
  std::vector<std::vector<task*>> _stage_root_tasks = {};
  int32_t _current_stage_id                         = 0;
  int32_t _current_task_id                          = 0;
  std::unordered_map<physical::relation*, std::vector<std::weak_ptr<task>>> _tasks_cache;
};

}  // namespace gqe
