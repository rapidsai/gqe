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
#include <gqe/utility.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace logical {

class relation {
 public:
  enum class relation_type { fetch, sort, project, aggregate, join, read, filter };

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

 private:
  // Child nodes of the current relation
  std::vector<std::shared_ptr<relation>> _children;
};

/**
 * @brief The fetch relation is used for limiting the number of rows returned.
 *
 * If the offset is specified, it will also return the row starting from the spcified
 * offset.
 */
class fetch_relation : public relation {
 public:
  /**
   * @brief Construct a new fetch relation object
   *
   * @param input_relation Input to the fetch relation
   * @param offset The offset experssed in number of records
   * @param count The number of records to return
   */
  fetch_relation(std::shared_ptr<relation> input_relation, int64_t offset, int64_t count);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::fetch; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the offset for the retrieval records
   *
   * @return The index of the starting row to be returned
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; };

  /**
   * @brief Return the count for the retrieval records
   *
   * @return The number of rows to be returned
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; };

 private:
  int64_t _offset;
  int64_t _count;
  std::vector<cudf::data_type> _data_types;
};

class aggregate_relation : public relation {
 public:
  aggregate_relation(
    std::shared_ptr<relation> input_relation,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::aggregate; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the list of keys to group by
   *
   * @note The returned keys do not share ownership. This object must be kept alive for the
   * returned keys to be valid.
   *
   * @return List of group by keys
   */
  [[nodiscard]] std::vector<expression*> keys_unsafe() const noexcept;

  /**
   * @brief Return the list of measures
   *
   * Each measure is a pair of cudf aggregation operation kind and expression.
   * This indicates the type of aggregate operation to perform on each value.
   * For example, the query:
   *
   * `select c0, sum(c1) from table_name grouby c0;`
   *
   * will result in a plan with
   * `keys = {col_reference(0)}`
   * `measures = {cudf::aggregation::SUM : col_reference(1)}`
   *
   * @return List of aggregate measures
   */
  [[nodiscard]] std::vector<std::pair<cudf::aggregation::Kind, expression*>> measures_unsafe()
    const noexcept;

 private:
  void _init_data_types() const;
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _measures;
  mutable std::optional<std::vector<cudf::data_type>> _data_types;
};

class sort_relation : public relation {
 public:
  sort_relation(std::shared_ptr<relation> input_relation,
                std::vector<cudf::order> column_orders,
                std::vector<cudf::null_order> null_precedences,
                std::vector<std::unique_ptr<expression>> expressions);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::sort; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the list of expressions.
   *
   * @note The returned expressions do not share ownership. This object must be kept alive for the
   * returned expressions to be valid.
   *
   * @return List of output expressions
   */
  [[nodiscard]] std::vector<expression*> expressions_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_expressions);
  }

  /**
   * @brief Accessor for column orders. Indicates the direction of the sort for each column
   */
  [[nodiscard]] std::vector<cudf::order> column_orders() const noexcept { return _column_orders; }

  /**
   * @brief Accessor for null orders. Indicates whether to return NULLs first or last for each
   * column
   */
  [[nodiscard]] std::vector<cudf::null_order> null_orders() const noexcept { return _null_orders; }

 private:
  std::vector<std::unique_ptr<expression>> _expressions;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_orders;
  std::vector<cudf::data_type> _data_types;
};

class filter_relation : public relation {
 public:
  /**
   * @brief Construct a new filter relation object
   *
   * @param input_relation Input relation to apply filter on
   * @param condition Filter expression to apply to the input
   */
  filter_relation(std::shared_ptr<relation> input_relation, std::unique_ptr<expression> condition);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::filter; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the filter condition for this relation
   *
   * The condition defines which rows to return
   *
   * @return Filter condition
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   */
  [[nodiscard]] expression* condition() const noexcept { return _condition.get(); }

 private:
  std::vector<cudf::data_type> _data_types;
  std::unique_ptr<expression> _condition;
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
   * @param projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   */
  join_relation(std::shared_ptr<relation> left,
                std::shared_ptr<relation> right,
                std::unique_ptr<expression> condition,
                join_type_type join_type,
                std::vector<cudf::size_type> projection_indices);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::join; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return join type for this relation
   *
   * @return Type of join to perform
   */
  [[nodiscard]] join_type_type join_type() const noexcept { return _join_type; }

  /**
   * @brief Return the join condition for this relation
   *
   * The condition defines when a left key matches a right key
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
  std::unique_ptr<expression> _condition;
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  mutable std::vector<cudf::data_type> _data_types;
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
   * @brief Return the name of the table to read from
   *
   * @return Name of table to read from
   */
  [[nodiscard]] std::string table_name() const { return _table_name; }

  /**
   * @brief Return the names of the columns to read
   *
   * @return List of columns to read
   */
  [[nodiscard]] std::vector<std::string> column_names() const { return _column_names; }

 private:
  std::vector<std::string> _column_names;
  std::string _table_name;
  std::vector<cudf::data_type> _data_types;
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
   * @brief Return a list of raw pointers to the output expressions.
   *
   * @return Vector of output expression raw pointers
   *
   * @note The returned expressions do not share ownership. This object must be kept alive for the
   * returned expressions to be valid.
   */
  std::vector<expression*> output_expressions_unsafe() const
  {
    return utility::to_raw_ptrs(_output_expressions);
  }

 private:
  void _init_data_types() const;
  //! List of one or more expressions to add to the input
  /*!
    This is usually used in SELECT and its order of selection.
  */
  std::vector<std::unique_ptr<expression>> _output_expressions;
  mutable std::optional<std::vector<cudf::data_type>> _data_types;
  // TODO: Pass projection information into JOIN. For now, we're going to return all columns.
  //       Projection will be handled in its own relation.
};
}  // namespace logical
}  // namespace gqe
