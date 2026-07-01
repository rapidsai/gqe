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

#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace physical {

class relation;
class read_relation;
class write_relation;
class broadcast_join_relation;
class shuffle_join_relation;
class project_relation;
class concatenate_sort_relation;
class filter_relation;
class concatenate_aggregate_relation;
class fetch_relation;
class union_all_relation;
class user_defined_relation;
class window_relation;
class gen_ident_col_relation;
class shuffle_relation;

/**
 * @brief Base interface for a physical relation visitor.
 *
 * By default, each visit method recurses into children and subqueries.
 * Override individual methods to customize behavior for specific relation types.
 */
struct relation_visitor {
  virtual ~relation_visitor() = default;

  virtual void visit(read_relation*);
  virtual void visit(write_relation*);
  virtual void visit(broadcast_join_relation*);
  virtual void visit(shuffle_join_relation*);
  virtual void visit(project_relation*);
  virtual void visit(concatenate_sort_relation*);
  virtual void visit(filter_relation*);
  virtual void visit(concatenate_aggregate_relation*);
  virtual void visit(fetch_relation*);
  virtual void visit(union_all_relation*);
  virtual void visit(user_defined_relation*);
  virtual void visit(window_relation*);
  virtual void visit(gen_ident_col_relation*);
  virtual void visit(shuffle_relation*);

 protected:
  /// Visit all children and subqueries of a relation.
  void visit_children(relation* rel);
};

/**
 * @brief Abstract base class for all physical relations.
 *
 * Compared to a logical relation, a physical relation encodes information on how to execute the
 * operation. For example, a logical join relation could correspond to either BroadcastJoin or
 * RepartitionedJoin physical relation.
 */
class relation {
 public:
  enum class relation_type {
    broadcast_join,
    concatenate_aggregate,
    concatenate_sort,
    fetch,
    filter,
    gen_ident_col,
    project,
    read,
    shuffle,
    shuffle_join,
    union_all,
    user_defined,
    window,
    write
  };

  /**
   * @brief Construct a new physical relation.
   *
   * @param[in] children Child nodes of the new relation.
   * @param[in] subqueries Subquery relations that are referenced within the expressions
   * associated with this relation.
   */
  relation(std::vector<std::shared_ptr<relation>> children,
           std::vector<std::shared_ptr<relation>> subqueries)
    : _children(std::move(children)), _subqueries(std::move(subqueries))
  {
  }

  virtual ~relation()                  = default;
  relation(const relation&)            = delete;
  relation& operator=(const relation&) = delete;

  /**
   * @brief Return the child nodes of the current relation.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> children_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_children);
  }

  [[nodiscard]] std::vector<relation*> subqueries_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_subqueries);
  }

  /**
   * @brief Return the number of children
   */
  [[nodiscard]] std::size_t children_size() const noexcept { return _children.size(); }

  /**
   * @brief Return the type of this physical relation.
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
  [[nodiscard]] virtual std::vector<cudf::data_type> output_data_types() const = 0;

  /**
   * @brief Return a string representation (in json format) of this relation.
   *
   * @note The returned json string is not prettified. This is meant to be used in
   * conjunction with tools like [PlantUML](www.plantuml.com).
   */
  [[nodiscard]] virtual std::string to_string() const = 0;

 private:
  std::vector<std::shared_ptr<relation>> _children;
  std::vector<std::shared_ptr<relation>> _subqueries;
};

}  // namespace physical
}  // namespace gqe
