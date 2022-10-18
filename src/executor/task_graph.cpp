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

#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>

#include <stdexcept>

namespace gqe {

std::unique_ptr<task_graph> task_graph_builder::build(physical::relation* root_relation)
{
  generate_task_graph_visitor visitor(this);
  root_relation->accept(visitor);

  // Update _stage_root_tasks with tasks in the last stage
  std::vector<task*> generated_tasks_ptrs;
  generated_tasks_ptrs.reserve(visitor._generated_tasks.size());
  for (auto const& generated_task : visitor._generated_tasks)
    generated_tasks_ptrs.push_back(generated_task.get());
  _stage_root_tasks.push_back(std::move(generated_tasks_ptrs));

  return std::make_unique<task_graph>(
    task_graph({std::move(visitor._generated_tasks), std::move(_stage_root_tasks)}));
}

void execute_task_graph_single_gpu(task_graph const* task_graph_to_execute)
{
  for (auto const& tasks_current_stage : task_graph_to_execute->stage_root_tasks) {
    for (auto& task : tasks_current_stage)
      task->execute();
  }
}

// FIXME: Current implementation of task_graph_builder::generate_task_graph_visitor::visit cannot
// assign the parent task and the child task to the same stage if the child task has already been
// generated with a different stage and is retrieved from the task cache
// (`task_graph_builder::generate_task_graph_visitor::is_cached`), even if the edge is not a
// pipeline breaker.

void task_graph_builder::generate_task_graph_visitor::visit(physical::read_relation* relation)
{
  if (is_cached(relation)) return;

  auto const table_name   = relation->table_name();
  auto const column_names = relation->column_names();

  std::vector<cudf::data_type> data_types;
  data_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    data_types.push_back(_builder->_catalog->column_type(table_name, column_name));

  auto const file_paths = _builder->_catalog->file_paths(table_name);
  for (auto const& file_path : file_paths) {
    _generated_tasks.push_back(
      std::make_shared<read_task>(_builder->_current_task_id,
                                  _builder->_current_stage_id,
                                  file_path,
                                  _builder->_catalog->file_format(table_name),
                                  column_names,
                                  data_types));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::broadcast_join_relation* relation)
{
  if (is_cached(relation)) return;

  auto const relation_join_type = relation->join_type();
  if (relation_join_type == join_type_type::full || relation_join_type == join_type_type::single)
    throw std::logic_error("Broadcast join does not support full join or single join");

  // Generate the right children tasks
  auto const children = relation->children_unsafe();

  generate_task_graph_visitor right_visitor(_builder);
  children[1]->accept(right_visitor);
  auto right_tasks = std::move(right_visitor._generated_tasks);

  // Insert a pipeline breaker since the right tasks need to be broadcasted
  std::vector<task*> right_tasks_ptr;
  right_tasks_ptr.reserve(right_tasks.size());
  for (auto const& right_task : right_tasks)
    right_tasks_ptr.push_back(right_task.get());

  _builder->insert_pipeline_breaker(std::move(right_tasks_ptr));

  // Generate the left children tasks
  generate_task_graph_visitor left_visitor(_builder);
  children[0]->accept(left_visitor);
  auto left_tasks = std::move(left_visitor._generated_tasks);

  // Concatenate the right child tasks if there are more than 1
  std::shared_ptr<task> concatenated_right_task;
  if (right_tasks.size() == 1) {
    concatenated_right_task = std::move(right_tasks[0]);
  } else {
    concatenated_right_task = std::make_shared<concatenate_task>(
      _builder->_current_task_id, _builder->_current_stage_id, std::move(right_tasks));
    _builder->_current_task_id++;
  }

  // Generate the join tasks
  for (auto& left_task : left_tasks) {
    _generated_tasks.push_back(std::make_shared<join_task>(_builder->_current_task_id,
                                                           _builder->_current_stage_id,
                                                           std::move(left_task),
                                                           concatenated_right_task,
                                                           relation_join_type,
                                                           relation->condition()->clone(),
                                                           relation->projection_indices()));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::project_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);

  generate_task_graph_visitor visitor(_builder);
  children[0]->accept(visitor);
  auto input_tasks = std::move(visitor._generated_tasks);

  // Generate the project tasks
  for (auto& task : input_tasks) {
    std::vector<std::unique_ptr<expression>> exprs;
    for (auto const& expr : relation->output_expressions_unsafe())
      exprs.push_back(expr->clone());

    _generated_tasks.push_back(std::make_shared<project_task>(
      _builder->_current_task_id, _builder->_current_stage_id, std::move(task), std::move(exprs)));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

bool task_graph_builder::generate_task_graph_visitor::is_cached(physical::relation* relation)
{
  if (_generated_tasks.size() != 0)
    throw std::logic_error("generate_task_graph_visitor already contains generated tasks");

  // Check the cache to see whether tasks for `relation` have already been generated
  auto tasks_cache_iter = _builder->_tasks_cache.find(relation);
  if (tasks_cache_iter != _builder->_tasks_cache.end()) {
    for (auto const& task_ptr : tasks_cache_iter->second)
      _generated_tasks.emplace_back(task_ptr);
    return true;
  }

  return false;
}

void task_graph_builder::generate_task_graph_visitor::update_cache(physical::relation* relation)
{
  std::vector<std::weak_ptr<task>> tasks;
  for (auto const& task_ptr : _generated_tasks)
    tasks.emplace_back(task_ptr);

  _builder->_tasks_cache[relation] = std::move(tasks);
}

}  // namespace gqe
