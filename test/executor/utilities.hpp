/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe {

namespace test {

/**
 * @brief A task that has a valid result during construction time. Used for testing.
 */
class executed_task : public task {
 public:
  executed_task(gqe::context_reference ctx_ref,
                int32_t task_id,
                int32_t stage_id,
                std::unique_ptr<cudf::table> result,
                std::vector<std::shared_ptr<task>> dependencies   = {},
                std::vector<std::shared_ptr<task>> subquery_tasks = {})
    : task(ctx_ref, task_id, stage_id, dependencies, subquery_tasks)
  {
    emit_result(std::move(result));
  }

  void execute() override {}
};

class no_op_task : public task {
 public:
  no_op_task(gqe::context_reference ctx_ref,
             int32_t task_id,
             int32_t stage_id,
             std::vector<std::shared_ptr<task>> dependencies   = {},
             std::vector<std::shared_ptr<task>> subquery_tasks = {})
    : task(ctx_ref, task_id, stage_id, dependencies, subquery_tasks)
  {
  }

  void execute() override {}
};

/**
 * @brief Table provider exposing one named table to physical QEP lowering.
 */
class single_table_provider final : public storage::table_provider {
 public:
  single_table_provider(std::string name, std::shared_ptr<storage::table> table)
    : _name(std::move(name)), _table(std::move(table))
  {
  }

  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override
  {
    if (table_name != _name) {
      throw std::logic_error("single_table_provider: unknown table " + std::string{table_name});
    }
    return _table;
  }

 private:
  std::string _name;
  std::shared_ptr<storage::table> _table;
};

/** @brief Return every task of `plan` whose dynamic type is `Task`, in plan order. */
template <typename Task>
[[nodiscard]] std::vector<qep::task const*> tasks_of_type(qep::query_execution_plan const& plan)
{
  std::vector<qep::task const*> matches;
  for (auto const* plan_task : plan.tasks()) {
    if (dynamic_cast<Task const*>(plan_task) != nullptr) { matches.push_back(plan_task); }
  }
  return matches;
}

/** @brief Execute every stage root in the order required by the task graph. */
inline void execute_stage_roots(gqe::task_graph const& graph)
{
  for (auto const& stage : graph.stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      GQE_EXPECTS(root != nullptr, "execute_stage_roots: stage root expired");
      root->execute();
    }
  }
}

}  // namespace test

}  // namespace gqe
