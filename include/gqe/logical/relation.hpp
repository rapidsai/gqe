/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/optimizer/relation_traits.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace optimizer {
class logical_optimizer;
class optimization_rule;
}  // namespace optimizer
namespace logical {

class relation;

class aggregate_relation;
class fetch_relation;
class filter_relation;
class join_relation;
class project_relation;
class read_relation;
class set_relation;
class sort_relation;
class user_defined_relation;
class window_relation;
class write_relation;

/**
 * @brief Base interface for a logical relation visitor.
 *
 * The default traversal is **post-order** (children before parent): the base
 * `visit_relation` catch-all calls `visit_children`, so every un-overridden
 * node type is automatically walked bottom-up. The optimizer driver does no
 * traversal; each rule's visitor is self-contained.
 *
 * Subclasses control traversal by overriding specific `visit(SpecificType*)`
 * methods and placing `visit_children(rel)` at the desired point:
 *  - `visit_children(rel)` first → post-order (bottom-up, default)
 *  - `visit_children(rel)` last → pre-order (top-down)
 *  - Omit `visit_children(rel)` → process only this node (no subtree walk)
 *
 * Un-overridden types fall through to `visit_relation`, which recurses
 * transparently. Override `visit_relation` with a no-op to suppress automatic
 * recursion for types not explicitly handled (e.g. in per-node-only visitors
 * like `rewrite_expressions_visitor`).
 *
 */
struct relation_visitor {
  virtual ~relation_visitor() = default;

  /// Generic catch-all. Default: post-order recurse via `visit_children`.
  /// Override to handle every node type uniformly, or to suppress recursion.
  virtual void visit_relation(relation* rel);

  virtual void visit(aggregate_relation*);
  virtual void visit(fetch_relation*);
  virtual void visit(filter_relation*);
  virtual void visit(join_relation*);
  virtual void visit(project_relation*);
  virtual void visit(read_relation*);
  virtual void visit(set_relation*);
  virtual void visit(sort_relation*);
  virtual void visit(user_defined_relation*);
  virtual void visit(window_relation*);
  virtual void visit(write_relation*);

  /// Walk all children and subqueries of `rel`, calling `accept(*this)` on each.
  /// Call from within a `visit(SpecificType*)` override to opt into recursion.
  void visit_children(relation* rel);
};

class relation {
  friend class gqe::optimizer::logical_optimizer;
  friend class gqe::optimizer::optimization_rule;

 public:
  enum class relation_type {
    fetch,
    sort,
    project,
    aggregate,
    join,
    window,
    read,
    write,
    filter,
    set,
    user_defined
  };

  /**
   * @brief Construct a relation node.
   *
   * @param[in] children Child nodes of the new relation node.
   * @param[in] subqueries Subquery relations references by expressions owned by this relation
   */
  relation(std::vector<std::shared_ptr<relation>> children,
           std::vector<std::shared_ptr<relation>> subqueries);

  virtual ~relation()                  = default;
  relation(const relation&)            = delete;
  relation& operator=(const relation&) = delete;

  /**
   * @brief Return the operator type of the relation.
   */
  [[nodiscard]] virtual relation_type type() const noexcept = 0;

  /**
   * @brief Accept a visitor.
   *
   * Implement the visitor pattern (https://en.wikipedia.org/wiki/Visitor_pattern) through double
   * dispatch.
   */
  virtual void accept(relation_visitor& visitor) = 0;

  /**
   * @brief Return the output data types of this relation.
   *
   * @return A vector whose size is equal to the number of columns in the
   * output relation. Element `i` of the vector records the type of column
   * `i`.
   */
  [[nodiscard]] virtual std::vector<cudf::data_type> data_types() const = 0;

  /**
   * @brief Return a string representation (in json format) of this relation.
   *
   * @note The returned json string is not prettified. This is meant to be used in
   * conjunction with tools like [PlantUML](www.plantuml.com).
   */
  [[nodiscard]] virtual std::string to_string() const = 0;

  /**
   * @brief Return the number of columns in the output relation.
   *
   * @return Number of columns.
   */
  [[nodiscard]] cudf::size_type num_columns() const;

  /**
   * @brief Return the children nodes as a list of `shared_ptr`.
   *
   * @note The returned relations share ownership with the caller. This is less
   * performant than the `children_unsafe()` function. This function should
   * only be used in place of its `_unsafe` counterpart if sharing of ownership
   * is absolutely necessary.
   */
  [[nodiscard]] std::vector<std::shared_ptr<relation>> children_safe() const noexcept;

  /**
   * @brief Return the children nodes as a list of raw pointers.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> children_unsafe() const noexcept;

  /**
   * @brief Return the number of children
   */
  [[nodiscard]] std::size_t children_size() const noexcept;

  /**
   * @brief Return the subquery relation nodes as a list of `shared_ptr`.
   *
   * @note The returned relations share ownership with the caller. This is the shared-ownership
   * counterpart of `subqueries_unsafe()`, mirroring `children_safe()`. Use it when subqueries
   * must outlive this relation, e.g. when an optimizer rule moves them onto a new node.
   */
  [[nodiscard]] std::vector<std::shared_ptr<relation>> subqueries_safe() const noexcept;

  /**
   * @brief Return the subquery relation nodes as a list of raw pointers.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> subqueries_unsafe() const noexcept;

  /**
   * @brief Return the number of subqueries that reference this relation
   */
  [[nodiscard]] std::size_t subqueries_size() const noexcept;

  /**
   * @brief Overloading == operator to compare this relation with another.
   *
   * @note This function compares the logical logical plans rooted at `this` and `other`
   * recursively and literally. It does not compare whether the two plans are semantically
   * equivalent.
   *
   * @param other The other relation to compare to
   * @return true if this and other relation have the same structure and members
   * @return false otherwise
   */
  virtual bool operator==(const relation& other) const = 0;

  /**
   * @brief Return relation traits associated with this relation
   */
  [[nodiscard]] optimizer::relation_traits const& relation_traits() const noexcept;

  /**
   * @brief Set relation traits associated with this relation
   *
   * @param traits The relation traits to be set to
   */
  void set_relation_traits(std::unique_ptr<optimizer::relation_traits> traits);

 protected:
  /**
   * @brief Function to compare members defined in the logical relation base class
   *
   * @note We do not want to define this funtion in operator==() of this class to keep it pure
   * virtual
   *
   * @param other The other relation to compare with
   * @return true If all base class members are equal
   * @return false Otherwise
   */
  [[nodiscard]] bool compare_relation_members(const relation& other) const;

 private:
  // Child nodes of the current relation
  std::vector<std::shared_ptr<relation>> _children;
  // Input relations to child subquery expressions
  std::vector<std::shared_ptr<relation>> _subqueries;
  // Property traits accessible by the optimizers
  std::unique_ptr<optimizer::relation_traits> _relation_traits;
};

}  // namespace logical
}  // namespace gqe
