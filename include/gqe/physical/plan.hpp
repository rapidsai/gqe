/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/physical/relation.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace gqe::physical {

/** @brief Stable identifier assigned to a relation within one physical plan. */
using plan_node_id = std::int32_t;

/** @brief Plan node IDs keyed by the identified physical relation. */
using plan_node_ids = std::unordered_map<relation const*, plan_node_id>;

/**
 * @brief Own a physical relation graph and its query-scoped node identities.
 *
 * Node IDs identify relations only within this plan. The relation pointers used as map keys remain
 * valid while this object or a copy of its root owner remains alive.
 *
 * Construction rejects null roots, associations for relations outside the owned graph,
 * non-positive IDs, and IDs assigned to more than one relation. Consumers that require every
 * relation to be identified should use `node_id`, which rejects a relation missing from the map.
 */
class identified_plan {
 public:
  /**
   * @brief Construct an identified physical plan.
   *
   * @param[in] root Owning root of the physical relation graph.
   * @param[in] node_ids Query-scoped node IDs for relations in `root`.
   */
  identified_plan(std::shared_ptr<relation> root, plan_node_ids node_ids);

  /** @brief Return the owning root of the physical relation graph. */
  [[nodiscard]] std::shared_ptr<relation> const& root() const noexcept { return _root; }

  /**
   * @brief Return the node ID assigned to `relation`.
   *
   * @throws std::logic_error If `relation` does not have a node ID in this plan.
   */
  [[nodiscard]] plan_node_id node_id(relation const* relation) const;

  /** @brief Return all relation-to-node-ID associations in this plan. */
  [[nodiscard]] plan_node_ids const& node_ids() const noexcept { return _node_ids; }

 private:
  std::shared_ptr<relation> _root;
  plan_node_ids _node_ids;
};

/**
 * @brief Assign deterministic node IDs to an existing physical relation graph.
 *
 * Relations are identified in root-first child and subquery order. A relation reached through
 * multiple paths receives one ID.
 *
 * @param[in] root Owning root of the physical relation graph.
 *
 * @return The physical plan and its assigned node IDs.
 */
[[nodiscard]] identified_plan make_identified_plan(std::shared_ptr<relation> root);

}  // namespace gqe::physical
