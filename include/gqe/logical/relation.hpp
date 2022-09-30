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
#include <gqe/types.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <stdexcept>
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
   * @brief Return the number of columns in the output relation.
   *
   * @return Number of columns.
   */
  [[nodiscard]] virtual cudf::size_type num_columns() const = 0;

  /**
   * @brief Return a string representation (in json format) of this relation.
   *
   * @note The returned json string is not prettified. This is meant to be used in
   * conjunction with tools like [PlantUML](www.plantuml.com).
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

class join_relation : public relation {
 public:
  /**
   * @brief Construct a new join relation object
   *
   * @param left The left input relation
   * @param right The right input relation
   * @param condition The expression to apply to input keys
   * @param join_type Type of join
   */
  join_relation(std::shared_ptr<relation> left,
                std::shared_ptr<relation> right,
                std::unique_ptr<expression> condition,
                join_type_type join_type);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::join; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::num_columns()
   */
  [[nodiscard]] cudf::size_type num_columns() const override;

  /**
   * @copydoc relation::to_string()
   */
  std::string to_string() const override;

  /**
   * @brief Return join type for this relation
   *
   * @return Type of join to perform
   */
  [[nodiscard]] join_type_type join_type() const noexcept { return _join_type; }

  /**
   * @brief Return the join condition for this relation
   *
   * @return Join condition
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   */
  [[nodiscard]] expression* condition() const noexcept { return _condition.get(); }

  /**
   * @brief Return the list of projection indices that indicate columns to return
   *
   * @return List of projection indices
   */
  [[nodiscard]] std::vector<cudf::size_type> projection_indices() const noexcept
  {
    return _projection_indices;
  }

 private:
  void _init_data_types() const;
  std::unique_ptr<expression>
    _condition;  //!< Join condition to define when a left tuple matches with a right tuple
  std::vector<cudf::size_type> _projection_indices;  //!< Columns to retain after joined
  join_type_type _join_type;
  mutable std::vector<cudf::data_type>
    _data_types;  //!< Data types of columns in the output relation
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
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Getter for name of the table to read from
   *
   * @return Name of table to read from
   */
  [[nodiscard]] std::string table_name() const { return _table_name; }

  /**
   * @copydoc relation::num_columns()
   */
  [[nodiscard]] cudf::size_type num_columns() const override { return this->_data_types.size(); }

 private:
  std::vector<std::string> _column_names;    //!< List of columns to read
  std::string _table_name;                   //!< Name of the table to read data from
  std::vector<cudf::data_type> _data_types;  //!< Data types of columns in the output relation
};

class project_relation : public relation {
 public:
  /**
   * @brief Constructs a projection relation.
   */
  project_relation(std::shared_ptr<relation> children,
                   std::vector<std::unique_ptr<expression>> output_expressions);

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

  /**
   * @copydoc relation::num_columns()
   */
  [[nodiscard]] cudf::size_type num_columns() const override
  {
    if (!(this->_data_types)) { this->_init_data_types(); }
    return this->_data_types.value().size();
  }

  /**
   * @brief Return a list of raw pointers to the output expressions.
   *
   * @return Vector of output expression raw pointers
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  std::vector<expression*> output_expressions_unsafe() const
  {
    std::vector<expression*> expressions_to_return;
    expressions_to_return.reserve(_output_expressions.size());

    for (auto const& expr : _output_expressions) {
      expressions_to_return.push_back(expr.get());
    }
    return expressions_to_return;
  }

 private:
  void _init_data_types() const;
  //! List of one or more expressions to add to the input
  /*!
    This is usually used in SELECT and its order of selection.
  */
  std::vector<std::unique_ptr<expression>> _output_expressions;
  mutable std::optional<std::vector<cudf::data_type>>
    _data_types;  //!< Data types of columns in the output relation
  // TODO: Pass projection information into JOIN. For now, we're going to return all columns.
  //       Projection will be handled in its own relation.
};
}  // namespace logical
}  // namespace gqe
