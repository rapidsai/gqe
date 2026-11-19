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

#include <gqe/physical/plan.hpp>

#include <cudf/types.hpp>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace gqe {

namespace physical {
class relation;
}

namespace qep {

class task;

/**
 * @brief Semantic role of a QEP task produced while lowering one physical relation.
 *
 * A physical relation may produce multiple QEP tasks. The role distinguishes those tasks without
 * depending on concrete task types or their construction order. Input materialization prepares an
 * encoded predecessor for the relation, while output materialization decodes a result produced by
 * the relation.
 */
enum class lowering_role : std::uint8_t {
  invalid = 0,  ///< Sentinel that is not a valid task role.

  read,              ///< Read the physical relation's input table.
  aggregate_reduce,  ///< Reduce a scalar aggregate.

  aggregate_partial,  ///< Produce partial grouped-aggregate state.
  aggregate_final,    ///< Combine partial grouped-aggregate state.

  aggregate_build,  ///< Build bounded grouped-aggregate state.
  aggregate_scan,   ///< Scan bounded grouped-aggregate state.

  top_k,              ///< Select the relation's ordered row range.
  filter,             ///< Apply the relation's predicate.
  project,            ///< Evaluate or select the relation's output expressions.
  materialize_input,  ///< Decode an input required by the relation.

  write,               ///< Write the relation's output table.
  join_build,          ///< Build the relation's join state.
  join_probe,          ///< Probe the relation's join state.
  materialize_output,  ///< Decode an encoded result produced by the relation.

  sort,   ///< Sort the relation's input.
  fetch,  ///< Select the relation's positional row range.

  join_complete,  ///< Finish marking after all join probes complete.
  join_scan,      ///< Emit matched or unmatched join build rows.

  read_resolve,      ///< Resolve source references into deferred columns.
  join_materialize,  ///< Gather selective sources of the completed join output.
};

/**
 * @brief Query-scoped identity of one QEP task produced from a physical plan node.
 *
 * Keys are unique within one physical QEP transformation result. Tasks produced from the same
 * physical relation must have distinct lowering roles.
 */
struct stable_task_key {
  /** @brief Node ID of the physical relation responsible for the task. */
  physical::plan_node_id relation_id;

  /** @brief Semantic role that distinguishes tasks produced by the same relation. */
  lowering_role role;

  bool operator==(stable_task_key const&) const  = default;
  auto operator<=>(stable_task_key const&) const = default;
};

/**
 * @brief Hash a `stable_task_key` for use in unordered containers.
 *
 * The hash is a process-local container implementation detail and is not a distributed identity.
 */
struct stable_task_key_hash {
  std::size_t operator()(stable_task_key const& key) const noexcept
  {
    auto const relation_hash = std::hash<physical::plan_node_id>{}(key.relation_id);
    auto const role_hash     = std::hash<std::underlying_type_t<lowering_role>>{}(
      static_cast<std::underlying_type_t<lowering_role>>(key.role));

    return relation_hash ^ (role_hash + 0x9e3779b9U + (relation_hash << 6) + (relation_hash >> 2));
  }
};

/** @brief Output column types for tasks in a QEP, keyed by task identity. */
using task_output_data_types = std::unordered_map<task const*, std::vector<cudf::data_type>>;

/** @brief Stable task keys for tasks in QEP, keyed by task identity. */
using task_stable_keys = std::unordered_map<task const*, stable_task_key>;

/**
 * @brief Physical-relation provenance for QEP tasks, keyed by task identity.
 *
 * The mapping associates each QEP task with a physical cohort:
 *
 * - A key identifies a QEP task.
 * - A value identifies the physical relation used as the task's stable cohort identity.
 * - Tasks mapped to the same relation pointer belong to the same cohort. Relation contents are not
 *   inspected.
 *
 * For a manually constructed QEP, associate tasks in the same intended cohort with the same stable
 * relation instance and tasks in different cohorts with different instances.
 *
 * Task and relation pointers are non-owning. The QEP that owns the tasks and the physical relation
 * graph must remain alive while the mapping is used. Relation pointers must be non-null when used
 * to classify a pipeline boundary.
 *
 * @see pipeline_normalization
 */
using task_physical_relations = std::unordered_map<task const*, physical::relation const*>;

}  // namespace qep
}  // namespace gqe
