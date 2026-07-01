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

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/optimizer/estimator.hpp>
#include <gqe/physical/relation.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace gqe::logical {
class join_relation;
class window_relation;
}  // namespace gqe::logical

namespace gqe {

/**
 * @brief A builder for generating the physical plan from the logical plan.
 */
class physical_plan_builder {
 public:
  /**
   * @brief Construct a physical plan builder.
   *
   * @param[in] cat Catalog containing table metadata.
   * @param[in] params Optimization parameters used to gate per-operator algorithm choices.
   */
  physical_plan_builder(catalog const* cat, optimization_parameters const* params = nullptr)
    : _catalog(cat), _params(params), _estimator(cat)
  {
  }

  /**
   * @brief Generate a physical plan from a logical plan.
   *
   * @param[in] logical_plan Root relation of the logical plan.
   *
   * @return Generated physical plan.
   */
  std::shared_ptr<physical::relation> build(logical::relation const* logical_plan);

 private:
  std::shared_ptr<physical::relation> plan_join(
    logical::join_relation const& node,
    std::vector<logical::relation*> const& children_logical,
    std::vector<std::shared_ptr<physical::relation>> children_physical,
    std::vector<std::shared_ptr<physical::relation>> subqueries_physical);

  std::shared_ptr<physical::relation> plan_window(
    logical::window_relation const& node,
    std::vector<logical::relation*> const& children_logical,
    std::vector<std::shared_ptr<physical::relation>> children_physical,
    std::vector<std::shared_ptr<physical::relation>> subqueries_physical);

  catalog const* _catalog;
  optimization_parameters const* _params;
  // A cache for keeping track of the previously-generated physical relations. Without the cache,
  // the builder can only generate a tree instead of a DAG.
  std::unordered_map<logical::relation const*, std::weak_ptr<physical::relation>> _cache;
  estimator _estimator;
};

}  // namespace gqe
