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

#pragma once

#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace physical {

class read_relation;
class write_relation;
class broadcast_join_relation;
class project_relation;
class concatenate_sort_relation;
class filter_relation;
class concatenate_aggregate_relation;
class fetch_relation;
class union_all_relation;
class user_defined_relation;
class window_relation;
class gen_ident_col_relation;

/**
 * @brief Base interface for a physical relation visitor.
 *
 * A concrete visitor needs to override these methods to customize the behavior.
 */
struct relation_visitor {
  virtual void visit(read_relation* relation)
  {
    throw std::logic_error("Visiting read_relation is not implemented");
  }
  virtual void visit(write_relation* relation)
  {
    throw std::logic_error("Visiting write_relation is not implemented");
  }
  virtual void visit(broadcast_join_relation* relation)
  {
    throw std::logic_error("Visiting broadcast_join_relation is not implemented");
  }
  virtual void visit(project_relation* relation)
  {
    throw std::logic_error("Visiting project_relation is not implemented");
  }
  virtual void visit(concatenate_sort_relation* relation)
  {
    throw std::logic_error("Visiting concatenate_sort_relation is not implemented");
  }
  virtual void visit(filter_relation* relation)
  {
    throw std::logic_error("Visiting filter_relation is not implemented");
  }
  virtual void visit(concatenate_aggregate_relation* relation)
  {
    throw std::logic_error("Visiting concatenate_aggregate_relation is not implemented");
  }
  virtual void visit(fetch_relation* relation)
  {
    throw std::logic_error("Visiting fetch_relation is not implemented");
  }
  virtual void visit(union_all_relation* relation)
  {
    throw std::logic_error("Visiting union_all_relation is not implemented");
  }
  virtual void visit(user_defined_relation* relation)
  {
    throw std::logic_error("Visiting user_defined_relation is not implemented");
  }
  virtual void visit(window_relation* relation)
  {
    throw std::logic_error("Visiting window_relation is not implemented");
  }
  virtual void visit(gen_ident_col_relation* relation)
  {
    throw std::logic_error("Visiting gen_ident_col_relation is not implemented");
  }
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
