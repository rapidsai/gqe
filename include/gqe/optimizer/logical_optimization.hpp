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

#pragma once

#include <cstddef>
#include <gqe/catalog.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/estimator.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/utility/helpers.hpp>

#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gqe {
namespace optimizer {

/**
 * @brief The optimization rule will operate at the logical plan level. The optimization steps
 * should be overloaded by the rule developer in `try_optimize()`. There are helper functions to
 * enable modifications of logical relations and their expressions:
 *  - `replace_child_at(logical::relation*, ...)` can be used to replace the origimal relation child
 * of the current relation with a new optimized relation child
 *  - `rewrite_relation_expressions(logical::relation*, ...)` can be used to apply a rewrite rule to
 * all of the expressions in the input relation
 */
class optimization_rule {
 public:
  enum class transform_direction {
    NONE,  // transform current node
    UP,    // post-order traversal
    DOWN   // pre-order traversal
  };

  /**
   * @brief This functor is used to defined the modification of the input expressions.
   *
   * @note If the functor does not modify the input expression, it should return a `nullptr`.
   */
  using expression_modifier_functor = std::function<std::unique_ptr<expression>(expression* expr)>;

  /**
   * @brief Construct a new optimization rule object
   *
   * @param cat Catalog to be used by the estimator
   * @param direction Determine how the query optimizer should apply this rule to the logical plan
   */
  optimization_rule(catalog const* cat, transform_direction direction = transform_direction::NONE)
    : _direction(direction), _estimator(cat)
  {
  }

  virtual ~optimization_rule() = default;

  /**
   * @brief Definition of the rule for optimizing logical relations
   *
   * @param logical_relation Relation to be optimized/rewritten
   * @param rule_applied Store the value of whether the rule has been applied
   * @return The optimized relation or the input relation if the rule cannot be applied
   */
  virtual std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const = 0;

  [[nodiscard]] virtual logical_optimization_rule_type type() const noexcept = 0;

  [[nodiscard]] transform_direction direction() const noexcept { return _direction; }

 protected:
  /**
   * @brief Replace the input relation's child at index `child_idx`
   *
   * @param relation Relation to replace child for
   * @param child_idx Index of child to replace
   * @param child The new relation child to be used for replacement
   */
  static void replace_child_at(logical::relation* relation,
                               std::size_t child_idx,
                               std::shared_ptr<logical::relation> child);

  /**
   * @brief Rewrite all expression members of relation using procedure defined in`f`
   *
   * @param relation Relation to modify the expression(s) for
   * @param f Functor defining how to rewrite each expression
   * @param direction In case of a nested expressions, how to traverse the expression tree
   */
  static void rewrite_relation_expressions(logical::relation* relation,
                                           expression_modifier_functor f,
                                           transform_direction direction);

  estimator get_estimator() const noexcept { return _estimator; }

 private:
  /**
   * @brief Optimize/rewrite the input expression non-recursively
   *
   * @note This function gets called when the `transform_direction` in
   * `rewrite_relation_expressions` is NONE
   *
   * @param expr Expression to be rewritten
   * @param f How to rewrite
   * @return The rewritten expression or null pointer if the rewrite has not been applied
   */
  static std::unique_ptr<expression> _expression_rewrite(expression* expr,
                                                         expression_modifier_functor f);

  /**
   * @brief Optimize/rewrite the input expression recursively
   *
   * @note This function gets called when the `transform_direction` in
   * `rewrite_relation_expressions` is DOWN
   *
   * @param expr Expression to be rewritten
   * @param f How to rewrite
   * @return The rewritten expression or null pointer if the rewrite has not been applied
   */
  static std::unique_ptr<expression> _expression_rewrite_down(expression* expr,
                                                              expression_modifier_functor f);

  /**
   * @brief Optimize/rewrite the input expression recursively
   *
   * @note This function gets called when the `transform_direction` in
   * `rewrite_relation_expressions` is UP
   *
   * @param expr Expression to be rewritten
   * @param f How to rewrite
   * @return The rewritten expression or null pointer if the rewrite has not been applied
   */
  static std::unique_ptr<expression> _expression_rewrite_up(expression* expr,
                                                            expression_modifier_functor f);
  transform_direction _direction;
  estimator _estimator;
};

class logical_optimizer {
 public:
  logical_optimizer() {}

  logical_optimizer(optimization_configuration* config, catalog const* cat)
    : _config(std::move(config))
  {
    // Instantiate rules enabled in config
    for (auto on_rule : _config->on_rules()) {
      _rules.push_back(_make_rule(on_rule, cat));
    }
  }

  virtual ~logical_optimizer() = default;

  /**
   * @brief Call optimization traversal and attempt to optimize based on each rule definition and
   * direction
   *
   * @param logical_relation The root of the logical query plan to optimize
   * @return Optimized logical plan. If no enabled rules are applicable, then the return plan will
   * be the same as the input plan.
   */
  std::shared_ptr<logical::relation> optimize(std::shared_ptr<logical::relation> logical_relation);

  /**
   * @brief Return the raw pointers of on-rule objects
   */
  std::vector<optimization_rule*> rules_unsafe() { return gqe::utility::to_raw_ptrs(_rules); }

  /**
   * @brief Return how many times the specified rule was actually applied
   *
   * @param rule_to_check The rule to get the count on
   * @return Number of rule application
   */
  size_t get_rule_count(gqe::optimizer::logical_optimization_rule_type rule_to_check)
  {
    return _applied_rule_counts[rule_to_check];
  }

 private:
  /**
   * @brief Attempt to optimize the logical relation with the specified rule
   *
   * @param logical_relation Logical relation to optimize
   * @param rule Rule to be applied to the input relation
   * @param rule_applied Whether the rule has been applied to the input logical relation
   * @return Optimized relation or input relation if the rule is not applicable
   */
  std::shared_ptr<logical::relation> _optimize(std::shared_ptr<logical::relation> logical_relation,
                                               const optimization_rule& rule,
                                               bool& rule_applied);

  /**
   * @brief Attempt to optimize the logical DAG starting at the input relation using the specified
   * rule in a BFS manner
   *
   * @param logical_relation The start logical relation to optimize
   * @param rule Rule to be applied to the input relation
   * @param rule_applied Whether the rule has been applied to the input logical relation
   * @return Optimized plan or subplan if the rule is applicable
   */
  std::shared_ptr<logical::relation> _optimize_down(
    std::shared_ptr<logical::relation> logical_relation,
    const optimization_rule& rule,
    bool& rule_applied);

  /**
   * @brief Attempt to optimize the logical DAG starting at the input relation using the specified
   * rule in a DFS manner
   *
   * @param logical_relation The start logical relation to optimize
   * @param rule Rule to be applied to the input relation
   * @param rule_applied Whether the rule has been applied to the input logical relation
   * @return Optimized plan or subplan if the rule is applicable
   */
  std::shared_ptr<logical::relation> _optimize_up(
    std::shared_ptr<logical::relation> logical_relation,
    const optimization_rule& rule,
    bool& rule_applied);

  /**
   * @brief Instantiate optimization rule
   *
   * @param rule_type The type of rule to instantiate
   * @param cat The catalog to be used by the estimator if applicable
   * @return The instantiated rule
   */
  std::unique_ptr<optimization_rule> _make_rule(logical_optimization_rule_type rule_type,
                                                catalog const* cat);

  optimization_configuration* _config;
  std::vector<std::unique_ptr<optimization_rule>> _rules;
  std::unordered_map<gqe::optimizer::logical_optimization_rule_type, size_t> _applied_rule_counts;
};

}  // namespace optimizer
}  // namespace gqe
