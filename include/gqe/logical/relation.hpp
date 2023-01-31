/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/types.hpp>
#include <gqe/utility.hpp>

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
namespace logical {

class relation {
 public:
  enum class relation_type { fetch, sort, project, aggregate, join, read, filter, set };

  /**
   * @brief Construct a relation node.
   *
   * @param[in] children Child nodes of the new relation node.
   * @param[in] subqueries Subquery relations references by expressions own by this relation
   */
  relation(std::vector<std::shared_ptr<relation>> children,
           std::vector<std::shared_ptr<relation>> subqueries)
    : _children(std::move(children)), _subqueries(std::move(subqueries))
  {
  }

  virtual ~relation()       = default;
  relation(const relation&) = delete;
  relation& operator=(const relation&) = delete;

  /**
   * @brief Return the operator type of the relation.
   */
  [[nodiscard]] virtual relation_type type() const noexcept = 0;

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
  [[nodiscard]] cudf::size_type num_columns() const { return data_types().size(); }

  /**
   * @brief Return the children nodes as a list of `shared_ptr`.
   *
   * @note The returned relations share ownership with the caller. This is less
   * performant than the `children_unsafe()` function. This function should
   * only be used in place of its `_unsafe` counterpart if sharing of ownership
   * is absolutely necessary.
   */
  [[nodiscard]] std::vector<std::shared_ptr<relation const>> children_safe() const noexcept
  {
    std::vector<std::shared_ptr<const relation>> children_to_return;
    children_to_return.reserve(_children.size());

    for (auto const& child : _children) {
      children_to_return.push_back(child);
    }

    return children_to_return;
  }

  /**
   * @brief Return the children nodes as a list of raw pointers.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> children_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_children);
  }

  /**
   * @brief Return the number of children
   */
  [[nodiscard]] std::size_t children_size() const noexcept { return _children.size(); }

  /**
   * @brief Return the subquery relation nodes as a list of raw pointers.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> subqueries_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_subqueries);
  }

  /**
   * @brief Return the number of subqueries that reference this relation
   */
  [[nodiscard]] std::size_t subqueries_size() const noexcept { return _subqueries.size(); }

 private:
  // Child nodes of the current relation
  std::vector<std::shared_ptr<relation>> _children;
  // Input relations to child subquery expressions
  std::vector<std::shared_ptr<relation>> _subqueries;
};

}  // namespace logical
}  // namespace gqe
