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

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>
#include <optional>
#include <ostream>
#include <regex>
#include <string>
#include <vector>

namespace gqe {
namespace logical {

class relation {
 public:
  enum class relation_type { fetch, sort, project, aggregation, join, read, filter };

  /**
   * @brief Construct a relation node.
   *
   * @param[in] children Child nodes of the new relation node.
   */
  relation(std::vector<std::shared_ptr<relation>> children) : _children(std::move(children)) {}

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
   * @brief Return a string representation of this relation.
   */
  [[nodiscard]] virtual std::string to_string() const = 0;

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
    std::vector<relation*> children_to_return;
    children_to_return.reserve(_children.size());

    for (auto const& child : _children) {
      children_to_return.push_back(child.get());
    }

    return children_to_return;
  }

  /**
   * @brief Return the number of children
   */
  [[nodiscard]] std::size_t children_size() const noexcept { return _children.size(); }

 private:
  // Child nodes of the current relation
  std::vector<std::shared_ptr<relation>> _children;
};

class read_relation : public relation {
 public:
  /**
   * @brief Construct a read relation.
   */
  read_relation(std::vector<std::string> column_names,
                std::vector<cudf::data_type> column_types,
                std::string table_name);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::read; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    return this->_data_types;  // initialized in constructor
  }
  /**
   * @copydoc relation::to_string()
   */
  std::string to_string() const override;

  /**
   * @brief Getter for name of the table to read from
   *
   * @return Name of table to read from
   */
  std::string table_name() const { return _table_name; }

 private:
  // List of columns to read
  std::vector<std::string> _column_names;
  // Name of the table to read data from
  std::string _table_name;
  // Data types of columns in the output relation
  mutable std::vector<cudf::data_type> _data_types;
};

class project_relation : public relation {
 public:
  /**
   * @brief Constructs a projection relation.
   */
  project_relation(std::shared_ptr<relation> children,
                   std::vector<std::shared_ptr<expression>> output_expressions);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::project; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  std::string to_string() const override;

  // List of one or more expressions to add to the input
  // This is usually used in SELECT and its order of selection
  std::vector<std::shared_ptr<expression>> output_expressions;
  // Data types of columns in the output relation
  mutable std::optional<std::vector<cudf::data_type>> _data_types;
};
}  // namespace logical
}  // namespace gqe