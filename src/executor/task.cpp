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

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/table/table_view.hpp>

#include <stack>
#include <stdexcept>
#include <variant>

namespace gqe {

task::task(context_reference ctx_ref,
           int32_t task_id,
           int32_t stage_id,
           std::vector<std::shared_ptr<task>> dependencies,
           std::vector<std::shared_ptr<task>> subqueries)
  : _ctx_ref(ctx_ref),
    _task_id(task_id),
    _stage_id(stage_id),
    _dependencies(std::move(dependencies)),
    _subqueries(std::move(subqueries)),
    _status(status_type::not_started)
{
  assert(ctx_ref._task_manager_context != nullptr);
  assert(ctx_ref._query_context != nullptr);
}

void task::migrate() { throw std::logic_error("task::migrate() has not been implemented"); }

std::optional<cudf::table_view> task::result() const noexcept
{
  cudf::table_view view = std::visit(
    utility::overloaded{[](const result_kind::owned& result) { return result.table->view(); },
                        [](const result_kind::borrowed& result) { return result.view; }},
    _result);

  return {view};
}

void task::emit_result(std::unique_ptr<cudf::table> new_result)
{
  _result = result_kind::owned{std::move(new_result)};
  _status = status_type::finished;
}

void task::emit_result(cudf::table_view new_result)
{
  _result = result_kind::borrowed{new_result};
  _status = status_type::finished;
}

void task::fail() { _status = status_type::failed; }

std::vector<task*> task::dependencies() const noexcept
{
  return utility::to_raw_ptrs(_dependencies);
}

std::vector<task*> task::subqueries() const noexcept { return utility::to_raw_ptrs(_subqueries); }

void task::prepare_dependent_tasks(std::vector<std::shared_ptr<task>>& dependent_tasks)
{
  for (auto const& dependent_task : dependent_tasks) {
    auto expected = status_type::not_started;
    if (dependent_task->_status.compare_exchange_strong(expected, status_type::in_progress)) {
      if (dependent_task->stage_id() == this->stage_id()) {
        // If the dependent task belongs to the same stage, it has not been executed by any other
        // GPUs, so the current GPU executes the task.
        try {
          dependent_task->execute();
        } catch (const std::exception&) {
          dependent_task->fail();
          throw;
        }
      } else if (dependent_task->stage_id() < this->stage_id()) {
        // If the dependent task belongs to a previous stage, it has already been executed by
        // another GPU, so we migrate the result to the current GPU.
        dependent_task->migrate();
      } else {
        throw std::logic_error("Dependent task belongs to a later stage than the current task");
      }
    } else {
      while (dependent_task->_status != status_type::finished &&
             dependent_task->_status != status_type::failed) {}
      if (dependent_task->_status == status_type::failed) {
        throw std::runtime_error("Dependent task failed execution");
      }
    }
  }
}

void task::prepare_dependencies()
{
  prepare_dependent_tasks(_dependencies);
  prepare_dependent_tasks(_subqueries);
}

void task::assign_pipeline(int32_t pipeline_id) noexcept
{
  std::stack<task*> tasks;
  tasks.push(this);
  while (!tasks.empty()) {
    auto current_task = tasks.top();
    tasks.pop();
    current_task->_pipeline_ids.insert(pipeline_id);
    for (auto dependency : current_task->dependencies()) {
      if (dependency->stage_id() == this->stage_id()) { tasks.push(dependency); }
    }
    for (auto subquery : current_task->subqueries()) {
      if (subquery->stage_id() == this->stage_id()) { tasks.push(subquery); }
    }
  }
}

}  // namespace gqe
