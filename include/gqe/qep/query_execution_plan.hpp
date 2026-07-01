/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/qep/task.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Identifier of a pipeline within a `pipeline_partition`.
 *
 * Pipelines are densely numbered `0 .. num_pipelines - 1` for direct indexing into
 * `pipeline_partition`'s vectors. The strong-typed wrapper around `std::size_t` prevents
 * accidental mixing with other index types (task ids, stage ids).
 */
class pipeline_id {
 public:
  constexpr pipeline_id() = default;
  constexpr explicit pipeline_id(std::size_t v) noexcept : _value(v) {}

  [[nodiscard]] constexpr std::size_t value() const noexcept { return _value; }

  constexpr bool operator==(pipeline_id const&) const  = default;
  constexpr auto operator<=>(pipeline_id const&) const = default;

 private:
  std::size_t _value{};
};

}  // namespace qep
}  // namespace gqe

namespace std {

/**
 * @brief `std::hash` specialisation so `std::unordered_map<pipeline_id, V>` works with the
 *        default template arguments.
 */
template <>
struct hash<gqe::qep::pipeline_id> {
  size_t operator()(gqe::qep::pipeline_id p) const noexcept { return hash<size_t>{}(p.value()); }
};

}  // namespace std

namespace gqe {
namespace qep {

/**
 * @brief Visitor for QEP tasks.
 *
 * Traverse or inspect tasks in a QEP. The traversal order is undefined.
 *
 * The QEP dispatches to the appropriate overload based on the concrete task type.
 */
class qep_visitor {
 public:
  virtual ~qep_visitor() = default;

  virtual void visit(optional_transform_task const& t) = 0;
  virtual void visit(fold_task const& t)               = 0;
  virtual void visit(iterate_task const& t)            = 0;
  virtual void visit(stateful_transform_task const& t) = 0;
};

/**
 * @brief Query Execution Plan (QEP).
 *
 * A QEP is a directed acyclic graph of tasks. The QEP specifies the dataflow between tasks. The
 * tasks are connected by their input and output states.
 *
 * The QEP does *not* track execution status. Status tracking is left to the executor in order to
 * separate concerns and enable multiple executor implementations.
 *
 * # Terminology
 *
 *  - `A -> B`: A sends data to B. A is a *predecessor* of B; B is a *successor* of A. The DAG
 * specifies the dataflow.
 *
 * # Design
 *
 * The QEP is immutable after construction. This ensures pointer stability.
 *
 * The graph edges are stored in both directions (predecessors and successors) to facilitate QEP
 * execution.
 */
class query_execution_plan {
  friend class query_execution_plan_builder;

 public:
  ~query_execution_plan()                                                = default;
  query_execution_plan(const query_execution_plan& other)                = delete;
  query_execution_plan& operator=(const query_execution_plan& other)     = delete;
  query_execution_plan(query_execution_plan&& other) noexcept            = default;
  query_execution_plan& operator=(query_execution_plan&& other) noexcept = default;

  /**
   * @brief Dispatch a visitor to each task in the QEP.
   *
   * @param[in,out] visitor The visitor to dispatch to each task.
   */
  void accept(qep_visitor& visitor) const;

  /**
   * @brief Return non-owning pointers to every task in the QEP.
   *
   * Order is unspecified — callers that need topological order should run
   * `qep::sort_topologically` with this as the node list.
   *
   * @return Non-owning pointers to every task in the QEP.
   */
  [[nodiscard]] std::vector<task const*> tasks() const;

  /**
   * @brief Lookup the predecessors of a task.
   *
   * @param[in] t The task.
   *
   * @return The predecessors of the task (tasks that send data to `t`). Empty if the task is a
   *         root.
   */
  [[nodiscard]] std::vector<task const*> predecessors(task const* t) const;

  /**
   * @brief Lookup the successors of a task.
   *
   * @param[in] t The task.
   *
   * @return The successors of the task (tasks that `t` sends data to). Empty if the task is a
   *         sink.
   */
  [[nodiscard]] std::vector<task const*> successors(task const* t) const;

 private:
  query_execution_plan(std::vector<std::unique_ptr<task>>&& tasks,
                       std::unordered_map<task const*, std::vector<task const*>>&& predecessors,
                       std::unordered_map<task const*, std::vector<task const*>>&& successors);

  std::vector<std::unique_ptr<task>> _tasks;  ///< Owned tasks.
  /// Predecessors of each task.
  std::unordered_map<task const*, std::vector<task const*>> _predecessors;
  /// Successors of each task.
  std::unordered_map<task const*, std::vector<task const*>> _successors;
};

/**
 * @brief A QEP builder.
 */
class query_execution_plan_builder {
 public:
  query_execution_plan_builder();
  ~query_execution_plan_builder()                                                        = default;
  query_execution_plan_builder(const query_execution_plan_builder& other)                = delete;
  query_execution_plan_builder& operator=(const query_execution_plan_builder& other)     = delete;
  query_execution_plan_builder(query_execution_plan_builder&& other) noexcept            = default;
  query_execution_plan_builder& operator=(query_execution_plan_builder&& other) noexcept = default;

  /**
   * @brief Register a task in the QEP.
   *
   * Successor relationships between registered tasks are added afterward via `add_successor`.
   * Tasks with no predecessors become roots.
   *
   * @param[in] new_task The task to add.
   */
  query_execution_plan_builder& add_task(std::unique_ptr<task> new_task);

  /**
   * @brief Register a successor relationship between two tasks.
   *
   * Declares that `predecessor` sends data to `successor` (dataflow:
   * `predecessor -> successor`). Both tasks must already have been added via `add_task`.
   *
   * # Task Argument Order
   *
   * The order in which successors are declared is maintained. This order becomes the positional
   * argument order to tasks.
   *
   * @param[in] predecessor The task that produces data.
   * @param[in] successor The task that consumes data from `predecessor`.
   */
  query_execution_plan_builder& add_successor(task const* predecessor, task const* successor);

  /**
   * @brief Build a QEP.
   *
   * @return The built QEP.
   */
  [[nodiscard]] query_execution_plan build();

 private:
  std::unordered_set<std::unique_ptr<task>, task_ptr_hash, task_ptr_equal> _tasks;
  std::unordered_map<task const*, std::vector<task const*>> _predecessors;
  std::unordered_map<task const*, std::vector<task const*>> _successors;
};

}  // namespace qep
}  // namespace gqe
