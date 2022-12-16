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

#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/fetch.hpp>
#include <gqe/executor/filter.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/sort.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/utility.hpp>

#include <limits>
#include <stdexcept>

namespace gqe {

std::unique_ptr<task_graph> task_graph_builder::build(physical::relation* root_relation)
{
  auto generated_tasks = generate_tasks(root_relation);

  // Update _stage_root_tasks with tasks in the last stage
  _stage_root_tasks.push_back(utility::to_raw_ptrs(generated_tasks));

  return std::make_unique<task_graph>(
    task_graph({std::move(generated_tasks), std::move(_stage_root_tasks)}));
}

std::vector<std::shared_ptr<task>> task_graph_builder::generate_tasks(physical::relation* relation)
{
  generate_task_graph_visitor visitor(this);
  relation->accept(visitor);
  return std::move(visitor._generated_tasks);
}

std::shared_ptr<task> task_graph_builder::concatenate(
  std::vector<std::shared_ptr<task>> input_tasks, bool is_pipeline_breaker)
{
  if (input_tasks.size() == 1) {
    return std::move(input_tasks[0]);
  } else {
    if (is_pipeline_breaker) insert_pipeline_breaker(utility::to_raw_ptrs(input_tasks));

    auto concatenated_input = std::make_shared<concatenate_task>(
      _current_task_id, _current_stage_id, std::move(input_tasks));
    _current_task_id++;
    return concatenated_input;
  }
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

  auto const children = relation->children_unsafe();

  if (relation->policy() == physical::broadcast_policy::right) {
    // Generate the right children tasks
    auto right_tasks = _builder->generate_tasks(children[1]);

    // Insert a pipeline breaker since the right tasks need to be broadcasted
    _builder->insert_pipeline_breaker(utility::to_raw_ptrs(right_tasks));

    // Generate the left children tasks
    auto left_tasks = _builder->generate_tasks(children[0]);

    // Concatenate the right child tasks
    // Note that we don't insert a pipeline breaker here so that the join tasks have the same
    // stage as the concatenate task. This way, the concatenate task implements the broadcasting
    // instead of being executed on a single GPU.
    auto concatenated_right_task = _builder->concatenate(std::move(right_tasks), false);

    // Generate the join tasks
    for (auto& left_task : left_tasks) {
      _generated_tasks.push_back(std::make_shared<join_task>(_builder->_current_task_id,
                                                             _builder->_current_stage_id,
                                                             std::move(left_task),
                                                             concatenated_right_task,
                                                             relation_join_type,
                                                             relation->condition()->clone(),
                                                             relation->projection_indices(),
                                                             relation->compare_nulls()));
      _builder->_current_task_id++;
    }
  } else {
    if (relation_join_type != join_type_type::inner)
      throw std::logic_error("Broadcast join can broadcast the left table only for inner join");

    // Generate the left children tasks
    auto left_tasks = _builder->generate_tasks(children[0]);

    // Insert a pipeline breaker since the left tasks need to be broadcasted
    _builder->insert_pipeline_breaker(utility::to_raw_ptrs(left_tasks));

    // Generate the right children tasks
    auto right_tasks = _builder->generate_tasks(children[1]);

    // Concatenate the left child tasks
    auto concatenated_left_task = _builder->concatenate(std::move(left_tasks), false);

    // Generate the join tasks
    for (auto& right_task : right_tasks) {
      _generated_tasks.push_back(std::make_shared<join_task>(_builder->_current_task_id,
                                                             _builder->_current_stage_id,
                                                             concatenated_left_task,
                                                             std::move(right_task),
                                                             relation_join_type,
                                                             relation->condition()->clone(),
                                                             relation->projection_indices(),
                                                             relation->compare_nulls()));
      _builder->_current_task_id++;
    }
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::project_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

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

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::concatenate_sort_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  // Generate the sort task
  std::vector<std::unique_ptr<expression>> keys;
  keys.reserve(relation->keys_unsafe().size());
  for (auto const& key : relation->keys_unsafe())
    keys.push_back(key->clone());

  _generated_tasks.push_back(std::make_shared<sort_task>(_builder->_current_task_id,
                                                         _builder->_current_stage_id,
                                                         std::move(concatenated_input),
                                                         std::move(keys),
                                                         relation->column_orders(),
                                                         relation->null_precedences()));
  _builder->_current_task_id++;

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::filter_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  // Generate the filter tasks
  for (auto& input_task : input_tasks) {
    _generated_tasks.push_back(
      std::make_shared<filter_task>(_builder->_current_task_id,
                                    _builder->_current_stage_id,
                                    std::move(input_task),
                                    relation->condition_unsafe()->clone()));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

using aggregation_keys_type = std::vector<std::unique_ptr<expression>>;
using aggregation_values_type =
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>>;

namespace {

aggregation_keys_type clone_aggregation_keys(aggregation_keys_type const& other_keys)
{
  aggregation_keys_type aggregation_keys;
  aggregation_keys.reserve(other_keys.size());
  for (auto const& key : other_keys)
    aggregation_keys.push_back(key->clone());
  return aggregation_keys;
}

aggregation_values_type clone_aggregation_values(aggregation_values_type const& other_values)
{
  aggregation_values_type aggregation_values;
  aggregation_values.reserve(other_values.size());
  for (auto const& [kind, expr] : other_values)
    aggregation_values.emplace_back(kind, expr->clone());
  return aggregation_values;
}

// Mappings for apply-concat-apply from the kind of the first aggregation to the kind of the
// second aggregation
cudf::aggregation::Kind get_second_aggregation_kind(cudf::aggregation::Kind first_aggregation_kind)
{
  switch (first_aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_VALID: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_ALL: return cudf::aggregation::SUM;
    case cudf::aggregation::MEAN: return cudf::aggregation::MEAN;
    default:
      throw std::logic_error("Unknown aggregation type in get_second_aggregation_kind: " +
                             std::to_string(first_aggregation_kind));
  }
}

}  // namespace

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::concatenate_aggregate_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  // FIXME: Apply-concat-apply is not necessary when input_tasks.size() == 1

  // Implement apply-concat-apply
  // There are mainly 4 steps:
  // Step 1: Apply aggregations on each partition
  // Step 2: Concatenate all partitions into a single partition
  // Step 3: Apply aggregations on the concatenated partition
  // Step 4: Optional post processing
  // For example, for "avg" aggregation, step 1 will perform "sum" and "count", step 3 will
  // perform "sum", and postprocessing will divide the sum by count.
  // Another example, for "count" aggregation, step 1 will perform "count", step 3 will perform
  // "sum", and there is no postprocessing.

  auto const keys   = relation->keys_unsafe();
  auto const values = relation->values_unsafe();

  // Step 1: Apply aggregations on each partition
  // Keys for the first aggregation are the same as those in the aggregation relation
  aggregation_keys_type first_aggregation_keys;
  first_aggregation_keys.reserve(keys.size());
  for (auto const& key : keys)
    first_aggregation_keys.push_back(key->clone());

  // Values for the first aggregation depend on the kind of the aggregation relation
  aggregation_values_type first_aggregation_values;
  first_aggregation_values.reserve(values.size());

  for (auto const& [kind, expr] : values) {
    switch (kind) {
      case cudf::aggregation::SUM:
        first_aggregation_values.emplace_back(cudf::aggregation::SUM, expr->clone());
        break;
      case cudf::aggregation::MEAN:
        first_aggregation_values.emplace_back(cudf::aggregation::MEAN, expr->clone());
        break;
      default:
        throw std::logic_error("Unknown aggregation type in task_graph_builder: " +
                               std::to_string(kind));
    }
  }

  std::vector<std::shared_ptr<task>> first_aggregation_tasks;
  first_aggregation_tasks.reserve(input_tasks.size());

  for (auto& input_task : input_tasks) {
    first_aggregation_tasks.push_back(
      std::make_shared<aggregate_task>(_builder->_current_task_id,
                                       _builder->_current_stage_id,
                                       std::move(input_task),
                                       clone_aggregation_keys(first_aggregation_keys),
                                       clone_aggregation_values(first_aggregation_values)));
    _builder->_current_task_id++;
  }

  // Step 2: Concatenate all partitions into a single partition
  auto concatenated_task = _builder->concatenate(std::move(first_aggregation_tasks));

  // Step 3: Apply aggregations on the concatenated partition
  // Keys for the second aggregation are the first few columns of the concatenated table
  aggregation_keys_type second_aggregation_keys;
  second_aggregation_keys.reserve(keys.size());
  for (std::size_t column_idx = 0; column_idx < keys.size(); column_idx++)
    second_aggregation_keys.push_back(std::make_unique<column_reference_expression>(column_idx));

  // Values for the second aggregation depend on the kind of the first aggregation
  aggregation_values_type second_aggregation_values;
  second_aggregation_values.reserve(first_aggregation_values.size());
  for (std::size_t column_idx = 0; column_idx < first_aggregation_values.size(); column_idx++) {
    second_aggregation_values.emplace_back(
      get_second_aggregation_kind(first_aggregation_values[column_idx].first),
      std::make_unique<column_reference_expression>(keys.size() + column_idx));
  }

  auto second_aggregation_task =
    std::make_shared<aggregate_task>(_builder->_current_task_id,
                                     _builder->_current_stage_id,
                                     std::move(concatenated_task),
                                     std::move(second_aggregation_keys),
                                     std::move(second_aggregation_values));
  _builder->_current_task_id++;

  // Step 4: Post processing
  std::vector<std::unique_ptr<expression>> output_expressions;

  // Keys don't need post-processing
  for (std::size_t column_idx = 0; column_idx < keys.size(); column_idx++)
    output_expressions.push_back(std::make_unique<column_reference_expression>(column_idx));

  std::size_t in_idx = keys.size();
  for (auto const& value : values) {
    switch (value.first) {
      case cudf::aggregation::SUM:
        // SUM does not need post-processing
        output_expressions.push_back(std::make_unique<column_reference_expression>(in_idx));
        in_idx++;
        break;
      case cudf::aggregation::MEAN:
        // MEAN does not need post-processing
        output_expressions.push_back(std::make_unique<column_reference_expression>(in_idx));
        in_idx++;
        break;
      default:
        throw std::logic_error("Unknown aggregation type in task_graph_builder: " +
                               std::to_string(value.first));
    }
  }
  assert(in_idx == keys.size() + first_aggregation_values.size());

  _generated_tasks.push_back(std::make_shared<project_task>(_builder->_current_task_id,
                                                            _builder->_current_stage_id,
                                                            std::move(second_aggregation_task),
                                                            std::move(output_expressions)));
  _builder->_current_task_id++;

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::fetch_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  if (relation->offset() > std::numeric_limits<cudf::size_type>::max() ||
      relation->count() > std::numeric_limits<cudf::size_type>::max())
    throw std::overflow_error(
      "Offset or limit overflows when building a task graph for a fetch relation");

  _generated_tasks.push_back(std::make_shared<fetch_task>(_builder->_current_task_id,
                                                          _builder->_current_stage_id,
                                                          std::move(concatenated_input),
                                                          relation->offset(),
                                                          relation->count()));
  _builder->_current_task_id++;

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
