/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/executor/gen_ident_col.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/sort.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/executor/window.hpp>
#include <gqe/executor/write.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/utility/logger.hpp>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>

namespace gqe {

std::unique_ptr<task_graph> task_graph_builder::build(physical::relation* root_relation)
{
  auto generated_tasks = generate_tasks(root_relation);

  // Update _stage_root_tasks with tasks in the last stage
  insert_pipeline_breaker(utility::to_raw_ptrs(generated_tasks));

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
    auto const num_tasks_current_stage = tasks_current_stage.size();

    std::size_t num_workers    = 1;
    auto const num_workers_str = std::getenv("MAX_NUM_WORKERS");
    if (num_workers_str != nullptr) num_workers = std::strtoul(num_workers_str, nullptr, 10);
    if (num_workers > num_tasks_current_stage) num_workers = num_tasks_current_stage;
    GQE_LOG_TRACE("Execute the stage with {} workers", num_workers);

    if (num_workers == 1) {
      // If the number of worker threads is 1, we could avoid the thread spawning cost by using the
      // main thread.
      for (auto& task : tasks_current_stage)
        task->execute();
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers);

      for (std::size_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
        workers.emplace_back([=, &tasks_current_stage]() {
          for (std::size_t task_idx = worker_idx; task_idx < num_tasks_current_stage;
               task_idx += num_workers) {
            tasks_current_stage[task_idx]->execute();
          }
        });
      }

      for (auto& worker : workers)
        worker.join();
    }
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

  auto const table_name     = relation->table_name();
  auto const column_names   = relation->column_names();
  auto const partial_filter = relation->partial_filter_unsafe();
  std::shared_ptr<task> concatenated_subquery_task;

  if (partial_filter && (partial_filter->type() == expression::expression_type::subquery)) {
    // If it's a subquery, we know it's an in-predicate
    auto const subquery = dynamic_cast<in_predicate_expression* const>(partial_filter);
    // Make sure dynamic_cast was successful
    assert(subquery != nullptr);
    auto const subquery_relations = relation->subqueries_unsafe();

    auto const subquery_tasks =
      _builder->generate_tasks(subquery_relations[subquery->relation_index()]);
    concatenated_subquery_task = _builder->concatenate(subquery_tasks);
  }

  std::vector<cudf::data_type> data_types;
  data_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    data_types.push_back(_builder->_catalog->column_type(table_name, column_name));

  std::unique_ptr<storage::readable_view> readable_view =
    _builder->_catalog->readable_view(table_name);
  if (!readable_view) { throw std::logic_error("table \"" + table_name + "\" is not readable"); }

  auto num_partitions = _builder->_catalog->num_partitions(table_name);
  for (decltype(num_partitions) partition_idx = 0; partition_idx < num_partitions;
       ++partition_idx) {
    _generated_tasks.push_back(
      readable_view->get_read_task(_builder->_current_task_id,
                                   _builder->_current_stage_id,
                                   num_partitions,
                                   partition_idx,
                                   column_names,
                                   data_types,
                                   partial_filter ? partial_filter->clone() : nullptr,
                                   std::vector<std::shared_ptr<task>>{concatenated_subquery_task}));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::write_relation* relation)
{
  if (is_cached(relation)) return;

  auto const table_name   = relation->table_name();
  auto const column_names = relation->column_names();

  std::vector<cudf::data_type> data_types;
  data_types.reserve(column_names.size());
  for (auto const& column_name : column_names) {
    data_types.push_back(_builder->_catalog->column_type(table_name, column_name));
  }

  auto const children = relation->children_unsafe();
  assert(children.size() == 1);

  // FIXME: This implementation requires that the number of concurrent writes
  // matches the number of child tasks. In future, we should handle mismatches
  // better. For example, we could concatenate child tasks to the exact number
  // of writers, or dynamically rescale the writer's concurrency (e.g., Parquet
  // pages).
  auto input_tasks = _builder->_catalog->max_concurrent_writers(table_name) == 1
                       ? std::vector{_builder->concatenate(_builder->generate_tasks(children[0]))}
                       : _builder->generate_tasks(children[0]);

  std::unique_ptr<storage::writeable_view> writeable_view =
    _builder->_catalog->writeable_view(table_name);
  if (!writeable_view) { throw std::logic_error("Table \"" + table_name + "\" is not writeable"); }

  auto parallelism = input_tasks.size();
  for (decltype(parallelism) instance_idx = 0; instance_idx < parallelism; ++instance_idx) {
    _generated_tasks.push_back(
      writeable_view->get_write_task(_builder->_current_task_id,
                                     _builder->_current_stage_id,
                                     parallelism,
                                     instance_idx,
                                     column_names,
                                     data_types,
                                     std::move(input_tasks[instance_idx])));
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
                                                             relation->projection_indices()));
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
                                                             relation->projection_indices()));
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

  auto const condition = relation->condition_unsafe();

  std::shared_ptr<task> concatenated_subquery_task = nullptr;
  if (condition->type() == expression::expression_type::subquery) {
    // If it's a subquery, we know it's an in-predicate
    auto const subquery           = dynamic_cast<subquery_expression* const>(condition);
    auto const subquery_relations = relation->subqueries_unsafe();

    // FIXME: Make sure the filter tasks and their children are within the same stage.
    auto const subquery_tasks =
      _builder->generate_tasks(subquery_relations[subquery->relation_index()]);
    concatenated_subquery_task = _builder->concatenate(subquery_tasks);
  }

  // Generate the filter tasks
  for (auto& input_task : input_tasks) {
    _generated_tasks.push_back(std::make_shared<filter_task>(
      _builder->_current_task_id,
      _builder->_current_stage_id,
      std::move(input_task),
      relation->condition_unsafe()->clone(),
      std::vector<std::shared_ptr<task>>{concatenated_subquery_task}));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::window_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  cudf::aggregation::Kind aggr_func = relation->aggr_func();
  std::vector<std::unique_ptr<expression>> ident_cols;
  std::vector<std::unique_ptr<expression>> arguments;
  std::vector<std::unique_ptr<expression>> partition_by;
  std::vector<std::unique_ptr<expression>> order_by;

  for (auto const& ident_col : relation->ident_cols_unsafe()) {
    ident_cols.push_back(ident_col->clone());
  }
  for (auto const& argument : relation->arguments_unsafe()) {
    arguments.push_back(argument->clone());
  }
  for (auto const& partition_by_exp : relation->partition_by_unsafe()) {
    partition_by.push_back(partition_by_exp->clone());
  }
  for (auto const& order_by_exp : relation->order_by_unsafe()) {
    order_by.push_back(order_by_exp->clone());
  }
  auto order_dirs         = relation->order_dirs();
  auto window_lower_bound = relation->window_lower_bound();
  auto window_upper_bound = relation->window_upper_bound();

  _generated_tasks.push_back(std::make_shared<window_task>(_builder->_current_task_id,
                                                           _builder->_current_stage_id,
                                                           std::move(concatenated_input),
                                                           aggr_func,
                                                           std::move(ident_cols),
                                                           std::move(arguments),
                                                           std::move(partition_by),
                                                           std::move(order_by),
                                                           std::move(order_dirs),
                                                           window_lower_bound,
                                                           window_upper_bound));
  _builder->_current_task_id++;

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
  // For example, for "mean" aggregation, step 1 will perform "sum" and "count", step 3 will
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
        first_aggregation_values.emplace_back(cudf::aggregation::SUM, expr->clone());
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_VALID, expr->clone());
        break;
      case cudf::aggregation::COUNT_VALID:
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_VALID, expr->clone());
        break;
      case cudf::aggregation::COUNT_ALL:
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_ALL, expr->clone());
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
      case cudf::aggregation::COUNT_VALID:
      case cudf::aggregation::COUNT_ALL:
        // SUM, COUNT_VALID, COUNT_ALL do not need post-processing
        output_expressions.push_back(std::make_unique<column_reference_expression>(in_idx));
        in_idx++;
        break;
      case cudf::aggregation::MEAN:
        // MEAN needs to divide the SUM by the COUNT_VALID
        output_expressions.push_back(std::make_unique<divide_expression>(
          std::make_shared<column_reference_expression>(in_idx),
          std::make_shared<column_reference_expression>(in_idx + 1)));
        in_idx += 2;
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

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::gen_ident_col_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  for (auto& task : input_tasks) {
    _generated_tasks.push_back(std::make_shared<gen_ident_col_task>(
      _builder->_current_task_id, _builder->_current_stage_id, std::move(task)));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::union_all_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 2);
  auto left_tasks = _builder->generate_tasks(children[0]);
  _builder->insert_pipeline_breaker(utility::to_raw_ptrs(left_tasks));
  auto right_tasks = _builder->generate_tasks(children[1]);

  // Concatenate the left tasks and right tasks
  for (auto& task : left_tasks)
    _generated_tasks.push_back(std::move(task));

  for (auto& task : right_tasks)
    _generated_tasks.push_back(std::move(task));

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::user_defined_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the tasks for child relations
  auto const children                  = relation->children_unsafe();
  auto const last_child_break_pipeline = relation->last_child_break_pipeline();

  std::vector<std::vector<std::shared_ptr<task>>> children_tasks;
  for (std::size_t child_idx = 0; child_idx < children.size(); child_idx++) {
    auto child_task = _builder->generate_tasks(children[child_idx]);
    if (child_idx != children.size() - 1 || last_child_break_pipeline)
      _builder->insert_pipeline_breaker(utility::to_raw_ptrs(child_task));
    children_tasks.push_back(std::move(child_task));
  }

  // Generate tasks for the user-defined relation through the user specified functor
  int32_t task_id = _builder->_current_task_id;
  _generated_tasks =
    relation->task_functor()(std::move(children_tasks), task_id, _builder->_current_stage_id);
  _builder->_current_task_id = task_id;

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
