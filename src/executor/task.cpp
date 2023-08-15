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

#include <gqe/executor/task.hpp>
#include <gqe/utility/helpers.hpp>

#include <stdexcept>

namespace gqe {

task::task(query_context* query_context,
           int32_t task_id,
           int32_t stage_id,
           std::vector<std::shared_ptr<task>> dependencies,
           std::vector<std::shared_ptr<task>> subqueries)
  : _query_context(query_context),
    _task_id(task_id),
    _stage_id(stage_id),
    _dependencies(std::move(dependencies)),
    _subqueries(std::move(subqueries)),
    _status(status_type::not_started)
{
  assert(query_context != nullptr);
}

void task::migrate() { throw std::logic_error("task::migrate() has not been implemented"); }

void task::update_result_cache(std::unique_ptr<cudf::table> new_result)
{
  _result_cache = std::move(new_result);
  _result       = _result_cache->view();
  _status       = status_type::finished;
}

std::vector<task*> task::dependencies() const noexcept
{
  return utility::to_raw_ptrs(_dependencies);
}

std::vector<task*> task::subqueries() const noexcept { return utility::to_raw_ptrs(_subqueries); }

query_context& task::get_query_context() const noexcept { return *_query_context; }

const optimization_parameters& task::get_optimization_parameters() const noexcept
{
  return *_query_context->parameters;
}

void task::prepare_dependent_tasks(std::vector<std::shared_ptr<task>>& dependent_tasks)
{
  for (auto const& dependent_task : dependent_tasks) {
    auto expected = status_type::not_started;
    if (dependent_task->_status.compare_exchange_strong(expected, status_type::in_progress)) {
      if (dependent_task->stage_id() == this->stage_id()) {
        // If the dependent task belongs to the same stage, it has not been executed by any other
        // GPUs, so the current GPU executes the task.
        dependent_task->execute();
      } else if (dependent_task->stage_id() < this->stage_id()) {
        // If the dependent task belongs to a previous stage, it has already been executed by
        // another GPU, so we migrate the result to the current GPU.
        dependent_task->migrate();
      } else {
        throw std::logic_error("Dependent task belongs to a later stage than the current task");
      }
    } else {
      while (dependent_task->_status != status_type::finished) {}
    }
  }
}

void task::prepare_dependencies() { prepare_dependent_tasks(_dependencies); }

void task::prepare_subqueries() { prepare_dependent_tasks(_subqueries); }

}  // namespace gqe
