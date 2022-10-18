/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/catalog.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>

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
 */
void execute_task_graph_single_gpu(task_graph const* task_graph_to_execute);

/**
 * @brief A builder for generating a task graph from a physical plan.
 */
class task_graph_builder {
 public:
  /**
   * @brief Construct a task graph builder object.
   *
   * @param[in] cat Catalog containing file locations and data types of the input tables.
   */
  task_graph_builder(catalog const* cat) : _catalog(cat) {}

  /**
   * @brief Generate a new task graph.
   *
   * @param[in] root_relation Root relation of the physical plan.
   *
   * @return The generated task graph.
   */
  std::unique_ptr<task_graph> build(physical::relation* root_relation);

 private:
  // A physical relation visitor used for generating a task graph for the physical relation and its
  // descendants.
  //
  // Since the visitor stores the generated tasks in the member variable `_generated_tasks`, the
  // visitor should not be reused.
  struct generate_task_graph_visitor : public physical::relation_visitor {
    generate_task_graph_visitor(task_graph_builder* builder) : _builder(builder) {}

    void visit(physical::read_relation* relation) override;
    void visit(physical::broadcast_join_relation* relation) override;
    void visit(physical::project_relation* relation) override;

    // Check the task cache in `_builder`. If the relation is found in the cache, the retrieved
    // tasks are copied to `_generated_tasks`, and the function returns true. Otherwise, the
    // function returns false.
    bool is_cached(physical::relation* relation);
    // Update the task cache in `_builder` with _generated_tasks.
    void update_cache(physical::relation* relation);

    task_graph_builder* _builder;
    std::vector<std::shared_ptr<task>> _generated_tasks;
  };

  // `stage_root_tasks` contains the root tasks of the current stage
  void insert_pipeline_breaker(std::vector<task*> stage_root_tasks)
  {
    _stage_root_tasks.push_back(std::move(stage_root_tasks));
    _current_stage_id++;
  }

  catalog const* _catalog;
  std::vector<std::vector<task*>> _stage_root_tasks = {};
  int32_t _current_stage_id                         = 0;
  int32_t _current_task_id                          = 0;
  std::unordered_map<physical::relation*, std::vector<std::weak_ptr<task>>> _tasks_cache;
};

}  // namespace gqe
