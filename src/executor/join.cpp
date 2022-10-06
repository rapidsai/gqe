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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <stdexcept>
#include <tuple>

namespace gqe {

join_task::join_task(int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> left,
                     std::shared_ptr<task> right,
                     join_type_type join_type,
                     std::unique_ptr<expression> condition,
                     std::vector<cudf::size_type> projection_indices)
  : task(task_id, stage_id, {std::move(left), std::move(right)}),
    _join_type(join_type),
    _condition(std::move(condition)),
    _projection_indices(std::move(projection_indices))
{
}

namespace {

/**
 * @brief A helper class for holding join keys.
 */
class join_keys_container {
 public:
  /**
   * @brief Construct a container for holding join keys.
   *
   * @param[in] left Left table to be joined.
   * @param[in] right Right table to be joined.
   */
  join_keys_container(cudf::table_view left, cudf::table_view right)
    : _left(std::move(left)), _right(std::move(right))
  {
  }

  /**
   * @brief Helper function for recursively parsing a join condition and store the key expressions.
   *
   * @param[in] condition Join condition to be parsed.
   * @param[out] left_keys_expr Expressions of the left key columns. The parsed expressions will be
   * appended to the vector.
   * @param[out] right_keys_expr Expressions of the right key columns. The parsed expressions will
   * be appended to the vector.
   */
  void parse_join_condition(expression const* condition,
                            std::vector<expression const*>& left_keys_expr,
                            std::vector<expression const*>& right_keys_expr)
  {
    // The top-level expression must be either equal or logical_and. Both are binary_op expressions.
    if (condition->type() != expression::expression_type::binary_op)
      throw std::logic_error("Unsupported join condition expression: " + condition->to_string());
    auto binary_op_condition = dynamic_cast<binary_op_expression const*>(condition);
    auto child_exprs         = condition->children();
    assert(child_exprs.size() == 2);

    // FIXME: Support inequality joins.
    switch (binary_op_condition->binary_operator()) {
      case cudf::binary_operator::EQUAL:
        left_keys_expr.push_back(child_exprs[0]);
        right_keys_expr.push_back(child_exprs[1]);
        break;
      case cudf::binary_operator::LOGICAL_AND:
        // If the top-level expression is AND, we recursively parse the two children expressions
        parse_join_condition(child_exprs[0], left_keys_expr, right_keys_expr);
        parse_join_condition(child_exprs[1], left_keys_expr, right_keys_expr);
        break;
      default: throw std::logic_error("Cannot parse join condition: " + condition->to_string());
    }
  }

  /**
   * @brief Add a join condition.
   *
   * @note Instead of returning the parsed expression, this function appends the parsed result
   * internally. `left_keys()` and `right_keys()` can be used to retrieve the parsing result
   * afterwards.
   *
   * @param[in] condition Join condition to be parsed.
   */
  void add_join_condition(expression const* condition)
  {
    std::vector<expression const*> left_keys_expr;
    std::vector<expression const*> right_keys_expr;

    parse_join_condition(condition, left_keys_expr, right_keys_expr);

    // Evaluate the expressions to get the left key columns
    auto [left_keys, left_cached_columns] = evaluate_expressions(_left, left_keys_expr);
    _left_keys.reserve(_left_keys.size() + left_keys.size());
    for (auto& left_key : left_keys)
      _left_keys.push_back(std::move(left_key));

    // Evaluate the expressions to get the right key columns
    // Note that the column index in `condition` references the combination of the left and the
    // right table. To get the column index within the right table, we need to subtract the number
    // of columns in the left table.
    auto [right_keys, right_cached_columns] =
      evaluate_expressions(_right, right_keys_expr, _left.num_columns());
    _right_keys.reserve(_right_keys.size() + right_keys.size());
    for (auto& right_key : right_keys)
      _right_keys.push_back(std::move(right_key));

    // Store the cached columns in the current object
    _column_cache.reserve(_column_cache.size() + left_cached_columns.size() +
                          right_cached_columns.size());
    for (auto& cached_column : left_cached_columns)
      _column_cache.push_back(std::move(cached_column));
    for (auto& cached_column : right_cached_columns)
      _column_cache.push_back(std::move(cached_column));
  }

  /**
   * @brief Return the parsed left key columns.
   *
   * @note This object must be kept alive for the return columns to be valid.
   */
  std::vector<cudf::column_view> const& left_keys() { return _left_keys; }

  /**
   * @brief Return the parsed right key columns.
   *
   * @note This object must be kept alive for the return columns to be valid.
   */
  std::vector<cudf::column_view> const& right_keys() { return _right_keys; }

 private:
  cudf::table_view _left;
  cudf::table_view _right;
  std::vector<cudf::column_view> _left_keys;
  std::vector<cudf::column_view> _right_keys;
  std::vector<std::unique_ptr<cudf::column>> _column_cache;
};

/**
 * @brief Materialize the join result.
 *
 * @param[in] left Left table to be joined.
 * @param[in] Right Right table to be joined.
 * @param[in] left_indices Row indices of the left table in the join result.
 * @param[in] left_policy Out-of-bound policy for gathering from the left table. For example, we
 * need to specify out_of_bounds_policy::NULLIFY for the full join because it could contain
 * out-of-bound indices.
 * @param[in] right_indices Row indices of the right table in the join result. If the join type does
 * not need to materialize the right table, this argument can be an empty column.
 * @param[in] right_policy Out-of-bound policy for gathering from the right table. For example, we
 * need to specify out_of_bounds_policy::NULLIFY for the left/full join because it could contain
 * out-of-bound indices.
 * @param[in] projection_indices Column indices of the combined table to materialize.
 */
std::unique_ptr<cudf::table> materialize(cudf::table_view const& left,
                                         cudf::table_view const& right,
                                         cudf::column_view const& left_indices,
                                         cudf::out_of_bounds_policy left_policy,
                                         cudf::column_view const& right_indices,
                                         cudf::out_of_bounds_policy right_policy,
                                         std::vector<cudf::size_type> const& projection_indices)
{
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  auto const left_ncolumns = left.num_columns();

  for (auto const& column_idx : projection_indices) {
    if (column_idx < left_ncolumns) {
      // This column belongs to the left table
      // Note that cudf::gather operates on a table_view instead of a column_view, so we use a
      // table_view with one column.
      auto const current_column = left.select({column_idx});
      auto gathered_column = cudf::gather(current_column, left_indices, left_policy)->release();
      assert(gathered_column.size() == 1);
      result_columns.push_back(std::move(gathered_column[0]));
    } else {
      // This column belongs to the right table
      assert(!right_indices.is_empty());
      auto const current_column = right.select({column_idx - left_ncolumns});
      auto gathered_column = cudf::gather(current_column, right_indices, right_policy)->release();
      assert(gathered_column.size() == 1);
      result_columns.push_back(std::move(gathered_column[0]));
    }
  }

  return std::make_unique<cudf::table>(std::move(result_columns));
}

}  // namespace

void join_task::execute()
{
  prepare_dependencies();

  auto dependent_tasks = dependencies();
  auto left_view       = *dependent_tasks[0]->result();
  auto right_view      = *dependent_tasks[1]->result();

  // Parse the join condition to get the keys
  join_keys_container join_keys(left_view, right_view);
  join_keys.add_join_condition(_condition.get());
  cudf::table_view left_keys(join_keys.left_keys());
  cudf::table_view right_keys(join_keys.right_keys());

  // Execute the join and get the result indicies
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices;

  switch (_join_type) {
    case join_type_type::inner:
      std::tie(left_indices, right_indices) =
        cudf::inner_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
      break;
    case join_type_type::left:
      std::tie(left_indices, right_indices) =
        cudf::left_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
      break;
    case join_type_type::left_semi:
      left_indices = cudf::left_semi_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
      break;
    case join_type_type::left_anti:
      left_indices = cudf::left_anti_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
      break;
    case join_type_type::full:
      std::tie(left_indices, right_indices) =
        cudf::full_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
      break;
    default: throw std::logic_error("Unknown join type");
  }

  // Convert the result indices into libcudf columns
  // FIXME: Use the new column constructor from device_uvector&& once we migrate to libcudf 22.10
  // Ref: https://github.com/rapidsai/cudf/pull/11356.
  auto const result_nrows  = left_indices->size();
  auto left_indices_column = std::make_unique<cudf::column>(
    cudf::data_type(cudf::type_to_id<cudf::size_type>()), result_nrows, left_indices->release());

  std::unique_ptr<cudf::column> right_indices_column;
  if (right_indices) {
    assert(right_indices->size() == result_nrows);
    right_indices_column = std::make_unique<cudf::column>(
      cudf::data_type(cudf::type_to_id<cudf::size_type>()), result_nrows, right_indices->release());
  } else {
    // If the program reaches here, the join does not need to materialize the right table
    // (e.g., left semi join).
    right_indices_column = cudf::make_empty_column(cudf::type_to_id<cudf::size_type>());
  }

  // Materialize the join result
  cudf::out_of_bounds_policy left_policy  = cudf::out_of_bounds_policy::NULLIFY;
  cudf::out_of_bounds_policy right_policy = cudf::out_of_bounds_policy::NULLIFY;

  switch (_join_type) {
    case join_type_type::inner:
      left_policy  = cudf::out_of_bounds_policy::DONT_CHECK;
      right_policy = cudf::out_of_bounds_policy::DONT_CHECK;
      break;
    case join_type_type::left:
      left_policy  = cudf::out_of_bounds_policy::DONT_CHECK;
      right_policy = cudf::out_of_bounds_policy::NULLIFY;
      break;
    case join_type_type::left_semi: left_policy = cudf::out_of_bounds_policy::DONT_CHECK; break;
    case join_type_type::left_anti: left_policy = cudf::out_of_bounds_policy::DONT_CHECK; break;
    case join_type_type::full:
      left_policy  = cudf::out_of_bounds_policy::NULLIFY;
      right_policy = cudf::out_of_bounds_policy::NULLIFY;
      break;
    default: throw std::logic_error("Unknown join type for materialization");
  }

  auto join_result = materialize(left_view,
                                 right_view,
                                 left_indices_column->view(),
                                 left_policy,
                                 right_indices_column->view(),
                                 right_policy,
                                 _projection_indices);

  update_result_cache(std::move(join_result));
  remove_dependencies();
}

}  // namespace gqe
