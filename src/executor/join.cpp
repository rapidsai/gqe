/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <gqe/executor/mark_join.hpp>
#include <gqe/executor/unique_key_inner_join.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "../libperfect/masked_join.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>

namespace gqe {

hash_join_interface::hash_join_interface(cudf::table_view const& build,
                                         cudf::null_equality compare_nulls,
                                         rmm::cuda_stream_view stream)
  : _hash_join_interface{std::make_unique<cudf::hash_join>(build, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
hash_join_interface::probe(cudf::table_view const& probe, join_type_type join_type) const
{
  switch (join_type) {
    case join_type_type::inner: return _hash_join_interface->inner_join(probe);
    case join_type_type::left: return _hash_join_interface->left_join(probe);
    case join_type_type::full: return _hash_join_interface->full_join(probe);
    default: throw std::logic_error("Unsupported join type");
  }
}

unique_key_join_interface::unique_key_join_interface(cudf::table_view const& build,
                                                     cudf::null_equality compare_nulls,
                                                     rmm::cuda_stream_view stream)
  : _unique_key_join_interface{std::make_unique<gqe::unique_key_join>(build, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_join_interface::probe(cudf::table_view const& probe, join_type_type join_type) const
{
  switch (join_type) {
    case join_type_type::inner: {
      return _unique_key_join_interface->inner_join(probe);
    }
    default: throw std::logic_error("Unsupported join type");
  }
}

mark_join_interface::mark_join_interface(cudf::table_view const& build,
                                         bool is_cached,
                                         cudf::null_equality compare_nulls,
                                         rmm::cuda_stream_view stream)
  : _mark_join_interface{
      std::make_unique<gqe::mark_join>(build, is_cached, compare_nulls, 0.5, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join_interface::probe(cudf::table_view const& probe, join_type_type join_type) const
{
  cudf::table_view empty_conds{};
  constexpr cudf::ast::expression const* ast = nullptr;
  return this->probe(probe, empty_conds, empty_conds, ast, join_type);
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join_interface::probe(cudf::table_view const& probe,
                           cudf::table_view const& left_conditional,
                           cudf::table_view const& right_conditional,
                           cudf::ast::expression const* binary_predicate,
                           join_type_type join_type) const
{
  bool is_anti_join;

  switch (join_type) {
    case join_type_type::left_semi: {
      is_anti_join = false;
      break;
    }
    case join_type_type::left_anti: {
      is_anti_join = true;
      break;
    }
    default: {
      throw std::logic_error("Unsupported join type in mark_join_interface join_type");
    }
  }
  auto positions = _mark_join_interface->perform_mark_join(
    probe, is_anti_join, left_conditional, right_conditional, binary_predicate);
  return positions;
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>>
mark_join_interface::compute_positions_list_from_cached_map(join_type_type join_type) const
{
  bool is_anti_join;
  switch (join_type) {
    case join_type_type::left_semi: {
      is_anti_join = false;
      break;
    }
    case join_type_type::left_anti: {
      is_anti_join = true;
      break;
    }
    default: throw std::logic_error("Unsupported join type in mark_join_interface");
  }

  return _mark_join_interface->compute_positions_list_from_cached_map(is_anti_join);
}

join_interface const* join_hash_map_cache::hash_map(cudf::table_view const& build_keys,
                                                    join_algorithm join_algorithm,
                                                    cudf::null_equality compare_nulls) const
{
  std::unique_lock latch_guard(_hash_map_latch);
  constexpr bool is_cached = true;
  if (!_hash_map) {
    switch (join_algorithm) {
      case join_algorithm::HASH_JOIN:
        _hash_map = std::make_unique<hash_join_interface>(build_keys, compare_nulls);
        break;
      case join_algorithm::UNIQUE_KEY_JOIN:
        _hash_map = std::make_unique<unique_key_join_interface>(build_keys, compare_nulls);
        break;
      case join_algorithm::MARK_JOIN:
        _hash_map = std::make_unique<mark_join_interface>(build_keys, is_cached, compare_nulls);
        break;
    }
  }

  return _hash_map.get();
}

join_interface const* join_hash_map_cache::hash_map() const
{
  std::unique_lock latch_guard(_hash_map_latch);
  if (_hash_map) {
    return _hash_map.get();
  } else {
    return nullptr;
  }
}

join_task::join_task(context_reference ctx_ref,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> left,
                     std::shared_ptr<task> right,
                     join_type_type join_type,
                     std::unique_ptr<expression> condition,
                     std::vector<cudf::size_type> projection_indices,
                     std::shared_ptr<join_hash_map_cache> hash_map_cache,
                     bool materialize_output,
                     gqe::unique_keys_policy unique_keys_pol,
                     bool perfect_hashing,
                     bool mark_join)
  : task(ctx_ref, task_id, stage_id, {std::move(left), std::move(right)}, {}),
    _join_type(join_type),
    _condition(std::move(condition)),
    _projection_indices(std::move(projection_indices)),
    _hash_map_cache(std::move(hash_map_cache)),
    _materialize_output(materialize_output),
    _unique_keys_policy(unique_keys_pol),
    _perfect_hashing(perfect_hashing),
    _mark_join(mark_join)
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

  void update_null_equality(cudf::null_equality policy)
  {
    if (_compare_nulls.has_value() && _compare_nulls.value() != policy)
      throw std::runtime_error(
        "Mixed null equalities in a single join expression is not supported");
    else
      _compare_nulls = policy;
  }

  /**
   * @brief Helper function for recursively parsing a join condition and store the key expressions.
   *
   * @param[in] condition Join condition to be parsed.
   * @param[out] left_key_exprs Expressions of the left key columns. The parsed expressions will be
   * appended to the vector.
   * @param[out] right_key_exprs Expressions of the right key columns. The parsed expressions will
   * be appended to the vector.
   * @param[out] non_equality_exprs Non-equality join condition expressions. The parsed expressions
   * will be appended to the vector.
   */
  void parse_join_condition(expression const* condition,
                            std::vector<expression const*>& left_key_exprs,
                            std::vector<expression const*>& right_key_exprs,
                            std::vector<expression const*>& non_equality_exprs)
  {
    if (condition->type() == expression::expression_type::binary_op) {
      auto binary_op_condition = dynamic_cast<binary_op_expression const*>(condition);
      auto child_exprs         = condition->children();
      assert(child_exprs.size() == 2);

      switch (binary_op_condition->binary_operator()) {
        case cudf::binary_operator::EQUAL:
          left_key_exprs.push_back(child_exprs[0]);
          right_key_exprs.push_back(child_exprs[1]);
          update_null_equality(cudf::null_equality::UNEQUAL);
          break;
        case cudf::binary_operator::NULL_EQUALS:
          left_key_exprs.push_back(child_exprs[0]);
          right_key_exprs.push_back(child_exprs[1]);
          update_null_equality(cudf::null_equality::EQUAL);
          break;
        case cudf::binary_operator::LOGICAL_AND:
        case cudf::binary_operator::NULL_LOGICAL_AND:
          // If the top-level expression is AND, we recursively parse the two children expressions
          parse_join_condition(child_exprs[0], left_key_exprs, right_key_exprs, non_equality_exprs);
          parse_join_condition(child_exprs[1], left_key_exprs, right_key_exprs, non_equality_exprs);
          break;
        default: non_equality_exprs.push_back(condition);
      }
    } else {
      non_equality_exprs.push_back(condition);
    }
  }

  /**
   * @brief Add a join condition.
   *
   * @note Instead of returning the parsed expression, this function appends the parsed result
   * internally. `left_keys()` and `right_keys()` can be used to retrieve the parsed equality key
   * columns afterwards. If a non-equal join condition exists, it can be retrieved via
   * `non_equality_conditions()`.
   *
   * @param[in] condition Join condition to be parsed.
   * @param[in] use_like_shift_and If `true`, use shift_and kernel for computing like filter.
   */
  void add_join_condition(expression const* condition, bool use_like_shift_and = true)
  {
    std::vector<expression const*> left_key_exprs;
    std::vector<expression const*> right_key_exprs;

    parse_join_condition(condition, left_key_exprs, right_key_exprs, _non_equality_conditions);

    // Evaluate the expressions to get the left key columns
    auto [left_keys, left_cached_columns] = evaluate_expressions(
      _left, left_key_exprs, /*column_reference_offset=*/0, use_like_shift_and);
    _left_keys.reserve(_left_keys.size() + left_keys.size());
    for (auto& left_key : left_keys)
      _left_keys.push_back(std::move(left_key));

    // Evaluate the expressions to get the right key columns
    // Note that the column position in `condition` references the combination of the left and the
    // right table. To get the column position within the right table, we need to subtract the
    // number of columns in the left table.
    auto [right_keys, right_cached_columns] =
      evaluate_expressions(_right, right_key_exprs, _left.num_columns(), use_like_shift_and);
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
   * If there is no equality join condition, this function returns an empty vector.
   *
   * @note This object must be kept alive for the return columns to be valid.
   */
  std::vector<cudf::column_view> const& left_keys() { return _left_keys; }

  /**
   * @brief Return the parsed right key columns.
   *
   * If there is no equality join condition, this function returns an empty vector.
   *
   * @note This object must be kept alive for the return columns to be valid.
   */
  std::vector<cudf::column_view> const& right_keys() { return _right_keys; }

  /**
   * @brief Return the parsed null equality policy.
   */
  cudf::null_equality compare_nulls()
  {
    if (_compare_nulls.has_value())
      return _compare_nulls.value();
    else
      throw std::runtime_error("Invalid access of uninitialized null equality policy");
  }

  /**
   * @brief Return all non-equality join conditions.
   */
  std::vector<expression const*> non_equality_conditions() { return _non_equality_conditions; }

 private:
  cudf::table_view _left;
  cudf::table_view _right;
  std::vector<cudf::column_view> _left_keys;
  std::vector<cudf::column_view> _right_keys;
  std::vector<std::unique_ptr<cudf::column>> _column_cache;
  std::optional<cudf::null_equality> _compare_nulls;
  std::vector<expression const*> _non_equality_conditions;
};

/**
 * @brief Convert a non-equality join predicate from a GQE expression to a cuDF AST expression.
 *
 * The output will be generated in the member variable `out_expr`.
 */
struct predicate_rewriter : public expression_visitor {
  using cache_type = std::vector<
    std::variant<std::unique_ptr<cudf::ast::expression>, std::unique_ptr<cudf::scalar>>>;

  /**
   * @brief Construct a predicate rewriter object.
   *
   * @param[in] left_num_columns Number of columns in the left table. Used for applying offsets to
   * column reference expressions.
   * @param[in] cache cuDF AST expressions do not own their child expressions and scalars. We use
   * this cache to keep the child expressions and scalars alive. Therefore, the output `out_expr` is
   * valid only if the cache is valid.
   */
  predicate_rewriter(cudf::size_type left_num_columns, cache_type& cache)
    : left_num_columns(left_num_columns), cache(cache)
  {
  }

  void visit(column_reference_expression const* expression) override
  {
    std::unique_ptr<cudf::ast::expression> cudf_expr;

    auto const column_idx = expression->column_idx();
    if (column_idx < left_num_columns) {
      // The expression references a column from the left table
      cudf_expr =
        std::make_unique<cudf::ast::column_reference>(column_idx, cudf::ast::table_reference::LEFT);
    } else {
      // The expression references a column from the right table
      cudf_expr = std::make_unique<cudf::ast::column_reference>(column_idx - left_num_columns,
                                                                cudf::ast::table_reference::RIGHT);
    }

    out_expr = cudf_expr.get();
    cache.push_back(std::move(cudf_expr));
  }

  void visit(binary_op_expression const* expression) override
  {
    auto child_in_exprs = expression->children();
    assert(child_in_exprs.size() == 2);

    predicate_rewriter lhs_rewriter(left_num_columns, cache);
    child_in_exprs[0]->accept(lhs_rewriter);

    predicate_rewriter rhs_rewriter(left_num_columns, cache);
    child_in_exprs[1]->accept(rhs_rewriter);

    auto cudf_expr =
      std::make_unique<cudf::ast::operation>(cudf_to_ast_operator(expression->binary_operator()),
                                             *lhs_rewriter.out_expr,
                                             *rhs_rewriter.out_expr);

    out_expr = cudf_expr.get();
    cache.push_back(std::move(cudf_expr));
  }

  void visit(unary_op_expression const* expression) override
  {
    auto child_in_exprs = expression->children();
    assert(child_in_exprs.size() == 1);

    predicate_rewriter child_rewriter(left_num_columns, cache);
    child_in_exprs[0]->accept(child_rewriter);

    auto cudf_expr = std::make_unique<cudf::ast::operation>(
      cudf_to_ast_operator(expression->unary_operator()), *child_rewriter.out_expr);

    out_expr = cudf_expr.get();
    cache.push_back(std::move(cudf_expr));
  }

  template <typename T>
  void create_literal_ast(literal_expression<T> const* expression)
  {
    using cudf_scalar_type =
      std::conditional_t<std::is_same_v<T, std::string>,
                         cudf::string_scalar,
                         std::conditional_t<std::is_same_v<T, cudf::timestamp_D>,
                                            cudf::timestamp_scalar<cudf::timestamp_D>,
                                            cudf::numeric_scalar<T>>>;
    auto scalar = std::make_unique<cudf_scalar_type>(expression->value(), !expression->is_null());

    auto cudf_expr = std::make_unique<cudf::ast::literal>(*scalar);
    out_expr       = cudf_expr.get();

    cache.push_back(std::move(scalar));
    cache.push_back(std::move(cudf_expr));
  }

  void visit(literal_expression<int32_t> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(literal_expression<int64_t> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(literal_expression<float> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(literal_expression<double> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(literal_expression<std::string> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(literal_expression<cudf::timestamp_D> const* expression) override
  {
    create_literal_ast(expression);
  }

  void visit(cast_expression const* expression) override
  {
    auto child_in_exprs = expression->children();
    assert(child_in_exprs.size() == 1);

    predicate_rewriter child_rewriter(left_num_columns, cache);
    child_in_exprs[0]->accept(child_rewriter);

    std::unique_ptr<cudf::ast::expression> cudf_expr;
    auto const type = expression->out_type();
    if (cudf::is_integral(type)) {
      cudf_expr = std::make_unique<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_INT64,
                                                         *child_rewriter.out_expr);
    } else if (cudf::is_floating_point(type) || cudf::is_fixed_point(type)) {
      cudf_expr = std::make_unique<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_FLOAT64,
                                                         *child_rewriter.out_expr);
    } else {
      throw std::logic_error("Unsupported cast expression in the join condition");
    }

    out_expr = cudf_expr.get();
    cache.push_back(std::move(cudf_expr));
  }

  cudf::ast::expression* out_expr;
  cudf::size_type left_num_columns;
  cache_type& cache;
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
      if (column_idx - left_ncolumns >= right.num_columns()) {
        throw std::logic_error("Projection indices are out of bounds for join materialization");
      }

      auto const current_column = right.select({column_idx - left_ncolumns});
      auto gathered_column = cudf::gather(current_column, right_indices, right_policy)->release();
      assert(gathered_column.size() == 1);
      result_columns.push_back(std::move(gathered_column[0]));
    }
  }

  return std::make_unique<cudf::table>(std::move(result_columns));
}

/**
 * @brief Returns whether gqe supports building the hash map separately.
 */
bool is_separate_hash_map_supported(join_type_type join_type, bool mark_join)
{
  return join_type == join_type_type::inner || join_type == join_type_type::left ||
         join_type == join_type_type::full ||
         (mark_join &&
          (join_type == join_type_type::left_semi || join_type == join_type_type::left_anti));
}

/**
 * @brief Returns whether gqe supports building the hash map separately when using
 *  mixed join conditions.
 */
bool is_separate_mixed_hash_map_supported(join_type_type join_type, bool mark_join)
{
  return mark_join &&
         (join_type == join_type_type::left_semi || join_type == join_type_type::left_anti);
}

/**
 * @brief Utility function for parsing predicates.
 * @param cache The cache used to retain the parse - the lifetime of the return is the same as the
 * cache.
 * @predicate_expr The expression to be parsed.
 * @num_cols The number of columns present in the expression.
 * @return A pointer to an expression object that can be evaluated. Lifetime depends on cache.
 */
cudf::ast::expression* parse_predicates(predicate_rewriter::cache_type& cache,
                                        std::vector<expression const*> const& predicates,
                                        std::unique_ptr<expression> predicate_expr,
                                        cudf::size_type num_cols)
{
  // Have at least one join condition that is not equality condition
  // Concatenate all non-equality join conditions
  for (std::size_t predicate_idx = 1; predicate_idx < predicates.size(); predicate_idx++) {
    predicate_expr = std::make_unique<logical_and_expression>(std::move(predicate_expr),
                                                              predicates[predicate_idx]->clone());
  }

  // Convert the non-equality join condition into a cuDF AST expression
  predicate_rewriter rewriter(num_cols, cache);
  predicate_expr->accept(rewriter);
  return rewriter.out_expr;
}

}  // namespace

void join_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range join_task_range("join_task");

  auto dependent_tasks = dependencies();
  auto left_view       = *dependent_tasks[0]->result();
  auto right_view      = *dependent_tasks[1]->result();

  bool use_like_shift_and = get_query_context()->parameters.filter_use_like_shift_and;
  // Parse the join condition to get the keys
  join_keys_container join_keys(left_view, right_view);
  join_keys.add_join_condition(_condition.get(), use_like_shift_and);
  cudf::table_view left_keys(join_keys.left_keys());
  cudf::table_view right_keys(join_keys.right_keys());
  auto const predicates = join_keys.non_equality_conditions();

  // Execute the join and get the result indicies
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices;

  if (predicates.size() == 0 && _hash_map_cache &&
      is_separate_hash_map_supported(_join_type, _mark_join)) {
    // Fast path: use the cached hash map for equality-only join
    cudf::table_view build_keys;
    cudf::table_view probe_keys;

    switch (_hash_map_cache->build_side()) {
      case join_hash_map_cache::build_location::left:
        build_keys = left_keys;
        probe_keys = right_keys;
        break;
      case join_hash_map_cache::build_location::right:
        build_keys = right_keys;
        probe_keys = left_keys;
        break;
    }

    join_algorithm join_algo;

    // TODO: if gqe::unique_key_join_supported is false, we can use distinct_hash_join instead,
    // but we need to upgrade to cudf 25.06
    if (((_unique_keys_policy == gqe::unique_keys_policy::left &&
          _hash_map_cache->build_side() == join_hash_map_cache::build_location::left) ||
         (_unique_keys_policy == gqe::unique_keys_policy::right &&
          _hash_map_cache->build_side() == join_hash_map_cache::build_location::right)) &&
        gqe::unique_key_join_supported(build_keys) && gqe::unique_key_join_supported(probe_keys) &&
        _join_type == join_type_type::inner) {
      join_algo = join_algorithm::UNIQUE_KEY_JOIN;
      GQE_LOG_TRACE("Join implementation: unique_key_join.");
    } else if (_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti) {
      join_algo = join_algorithm::MARK_JOIN;
      assert(_hash_map_cache->build_side() == join_hash_map_cache::build_location::left);
      GQE_LOG_TRACE("Join implementation: cached equality gqe::mark_join.");
    } else {
      join_algo = join_algorithm::HASH_JOIN;
      GQE_LOG_TRACE("Join implementation: hash join.");
    }

    auto const hash_map =
      _hash_map_cache->hash_map(build_keys, join_algo, join_keys.compare_nulls());

    std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_indices;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_indices;

    std::tie(probe_indices, build_indices) = hash_map->probe(probe_keys, _join_type);

    switch (_hash_map_cache->build_side()) {
      case join_hash_map_cache::build_location::left:
        left_indices  = std::move(build_indices);
        right_indices = std::move(probe_indices);
        break;
      case join_hash_map_cache::build_location::right:
        left_indices  = std::move(probe_indices);
        right_indices = std::move(build_indices);
        break;
    }
  } else if (_hash_map_cache && is_separate_mixed_hash_map_supported(_join_type, _mark_join)) {
    // only mark_join is supported in this path for now
    assert(_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti);
    assert(left_keys.num_columns() == right_keys.num_columns());
    assert(_hash_map_cache->build_side() == join_hash_map_cache::build_location::left);
    // Fast path mixed joins: use the cached hash map
    cudf::table_view build_keys;
    cudf::table_view probe_keys;

    build_keys               = left_keys;
    probe_keys               = right_keys;
    join_algorithm join_algo = join_algorithm::MARK_JOIN;

    auto const hash_map =
      _hash_map_cache->hash_map(build_keys, join_algo, join_keys.compare_nulls());

    std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_indices;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_indices;

    predicate_rewriter::cache_type cache;
    auto const predicate_ast =
      parse_predicates(cache, predicates, predicates[0]->clone(), left_view.num_columns());

    std::tie(probe_indices, build_indices) =
      hash_map->probe(probe_keys, left_view, right_view, predicate_ast, _join_type);

    left_indices  = std::move(build_indices);
    right_indices = std::move(probe_indices);
    if (_join_type == join_type_type::left_semi) {
      GQE_LOG_TRACE("Join implementation: cached mixed gqe::mark_join semi");
    } else if (_join_type == join_type_type::left_anti) {
      GQE_LOG_TRACE("Join implementation: cached mixed gqe::mark_join anti");
    }
  } else {
    // Fallback path: reconstruct hash map from scratch
    if (predicates.size() == 0) {
      // Only have equality join conditions
      auto const compare_nulls = join_keys.compare_nulls();
      switch (_join_type) {
        case join_type_type::inner: {
          if (_perfect_hashing) {
            switch (_unique_keys_policy) {
              case gqe::unique_keys_policy::left: {
                GQE_LOG_TRACE("Join implementation: perfect_join.");
                std::tie(left_indices, right_indices) =
                  libperfect::perfect_join(left_keys, right_keys);
                break;
              }
              case gqe::unique_keys_policy::right: {
                GQE_LOG_TRACE("Join implementation: perfect_join.");
                std::tie(right_indices, left_indices) =
                  libperfect::perfect_join(right_keys, left_keys);
                break;
              }
              default: {
                throw std::logic_error(
                  "Perfect hashing requires that at least one side has unique keys");
              }
            }
          } else {
            switch (_unique_keys_policy) {
              case gqe::unique_keys_policy::left: {
                GQE_LOG_TRACE("Join implementation: unique_key_inner_join.");
                std::tie(left_indices, right_indices) =
                  unique_key_inner_join(left_keys, right_keys, compare_nulls);
                break;
              }
              case gqe::unique_keys_policy::right: {
                GQE_LOG_TRACE("Join implementation: unique_key_inner_join.");
                std::tie(right_indices, left_indices) =
                  unique_key_inner_join(right_keys, left_keys, compare_nulls);
                break;
              }
              default: {
                GQE_LOG_TRACE("Join implementation: cudf::inner_join.");
                std::tie(left_indices, right_indices) =
                  cudf::inner_join(left_keys, right_keys, compare_nulls);
              }
            }
          }
        } break;
        case join_type_type::left:
          GQE_LOG_TRACE("Join implementation: cudf::left_join.");
          std::tie(left_indices, right_indices) =
            cudf::left_join(left_keys, right_keys, compare_nulls);
          break;
        case join_type_type::left_semi:
          if (_mark_join) {
            GQE_LOG_TRACE("Join implementation: gqe::left_semi_mark_join");
            left_indices = gqe::left_semi_mark_join(left_keys, right_keys, compare_nulls);
          } else {
            GQE_LOG_TRACE("Join implementation: cudf::left_semi_join");
            left_indices = cudf::left_semi_join(left_keys, right_keys, compare_nulls);
          }
          break;
        case join_type_type::left_anti:
          if (_mark_join) {
            GQE_LOG_TRACE("Join implementation: gqe::left_anti_mark_join.");
            left_indices = gqe::left_anti_mark_join(left_keys, right_keys, compare_nulls);
          } else {
            GQE_LOG_TRACE("Join implementation: cudf::left_anti_join.");
            left_indices = cudf::left_anti_join(left_keys, right_keys, compare_nulls);
          }
          break;
        case join_type_type::full:
          GQE_LOG_TRACE("Join implementation: cudf::full_join.");
          std::tie(left_indices, right_indices) =
            cudf::full_join(left_keys, right_keys, compare_nulls);
          break;
        default: throw std::logic_error("Unknown join type for equality join");
      }
    } else {
      predicate_rewriter::cache_type cache;
      cudf::ast::expression const& predicate_ast =
        *parse_predicates(cache, predicates, predicates[0]->clone(), left_view.num_columns());

      if (!left_keys.is_empty()) {
        // Have both equality and non-equality join conditions
        assert(left_keys.num_columns() == right_keys.num_columns());

        auto const compare_nulls = join_keys.compare_nulls();
        switch (_join_type) {
          case join_type_type::inner:
            GQE_LOG_TRACE("Join implementation: cudf::mixed_inner_join.");
            std::tie(left_indices, right_indices) = cudf::mixed_inner_join(
              left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            break;
          case join_type_type::left:
            GQE_LOG_TRACE("Join implementation: cudf::mixed_left_join.");
            std::tie(left_indices, right_indices) = cudf::mixed_left_join(
              left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            break;
          case join_type_type::left_semi:
            if (_mark_join) {
              GQE_LOG_TRACE("Join implementation: gqe::mixed_left_semi_mark_join.");
              left_indices = gqe::mixed_left_semi_mark_join(
                left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            } else {
              GQE_LOG_TRACE("Join implementation: cudf::mixed_left_semi_join.");
              left_indices = cudf::mixed_left_semi_join(
                left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            }
            break;
          case join_type_type::left_anti:
            if (_mark_join) {
              GQE_LOG_TRACE("Join implementation: gqe::mixed_left_anti_mark_join.");
              left_indices = gqe::mixed_left_anti_mark_join(
                left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            } else {
              GQE_LOG_TRACE("Join implementation: cudf::mixed_left_anti_join.");
              left_indices = cudf::mixed_left_anti_join(
                left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            }
            break;
          case join_type_type::full:
            GQE_LOG_TRACE("Join implementation: cudf::mixed_full_join.");
            std::tie(left_indices, right_indices) = cudf::mixed_full_join(
              left_keys, right_keys, left_view, right_view, predicate_ast, compare_nulls);
            break;
          default: throw std::logic_error("Unknown join type for mixed-condition join");
        }
      } else {
        // Only have non-equality join conditions
        switch (_join_type) {
          case join_type_type::inner:
            GQE_LOG_TRACE("Join implementation: cudf::conditional_inner_join.");
            std::tie(left_indices, right_indices) =
              cudf::conditional_inner_join(left_view, right_view, predicate_ast);
            break;
          case join_type_type::left:
            GQE_LOG_TRACE("Join implementation: cudf::conditional_left_join.");
            std::tie(left_indices, right_indices) =
              cudf::conditional_left_join(left_view, right_view, predicate_ast);
            break;
          case join_type_type::left_semi:
            GQE_LOG_TRACE("Join implementation: cudf::conditional_left_semi_join.");
            left_indices = cudf::conditional_left_semi_join(left_view, right_view, predicate_ast);
            break;
          case join_type_type::left_anti:
            GQE_LOG_TRACE("Join implementation: cudf::conditional_left_anti_join.");
            left_indices = cudf::conditional_left_anti_join(left_view, right_view, predicate_ast);
            break;
          case join_type_type::full:
            GQE_LOG_TRACE("Join implementation: cudf::conditional_full_join.");
            std::tie(left_indices, right_indices) =
              cudf::conditional_full_join(left_view, right_view, predicate_ast);
            break;
          default: throw std::logic_error("Unknown join type for conditional join");
        }
      }
    }
  }
  // Convert the result indices into libcudf columns
  auto left_indices_column =
    std::make_unique<cudf::column>(std::move(*left_indices), rmm::device_buffer{}, 0);

  std::unique_ptr<cudf::column> right_indices_column;
  if (right_indices) {
    right_indices_column =
      std::make_unique<cudf::column>(std::move(*right_indices), rmm::device_buffer{}, 0);
    assert(left_indices_column->size() == right_indices_column->size());
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

  std::unique_ptr<cudf::table> join_result;
  if (_materialize_output) {
    join_result = materialize(left_view,
                              right_view,
                              left_indices_column->view(),
                              left_policy,
                              right_indices_column->view(),
                              right_policy,
                              _projection_indices);
  } else {
    std::vector<std::unique_ptr<cudf::column>> out_columns;
    out_columns.push_back(std::move(left_indices_column));
    if (right_indices) { out_columns.push_back(std::move(right_indices_column)); }
    join_result = std::make_unique<cudf::table>(std::move(out_columns));
  }

  utility::nvtx_mark(std::string("left_size:") + std::to_string(left_view.num_rows()) +
                     ", right_size:" + std::to_string(right_view.num_rows()) +
                     ", result_size:" + std::to_string(join_result->num_rows()));

  GQE_LOG_TRACE(
    "Execute join task: task_id={}, stage_id={}, left_size={}, right_size={}, output_size={}.",
    task_id(),
    stage_id(),
    left_view.num_rows(),
    right_view.num_rows(),
    join_result->num_rows());
  emit_result(std::move(join_result));
  remove_dependencies();
}

materialize_join_from_position_lists_task::materialize_join_from_position_lists_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> left_table,
  std::vector<std::shared_ptr<task>> position_lists,
  join_type_type join_type,
  std::vector<cudf::size_type> projection_indices,
  std::shared_ptr<join_hash_map_cache> hash_map_cache,
  bool mark_join)
  : task(ctx_ref,
         task_id,
         stage_id,
         [&left_table, &position_lists]() -> std::vector<std::shared_ptr<task>> {
           std::vector<std::shared_ptr<task>> dependencies = std::move(position_lists);
           // implementation of this task expects the left table to be the first element in
           // dependencies vector
           dependencies.insert(dependencies.begin(), std::move(left_table));
           return dependencies;
         }(),
         {}),
    _join_type(join_type),
    _projection_indices(projection_indices),
    _hash_map_cache(hash_map_cache),
    _mark_join(mark_join)
{
  assert(join_type == join_type_type::left_semi || join_type == join_type_type::left_anti);
}

void materialize_join_from_position_lists_task::execute()
{
  prepare_dependencies();

  GQE_EXPECTS(
    _join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti,
    "materialize_join_from_position_lists_task only works with left-semi or left-anti join.");

  auto dependent_tasks = dependencies();
  auto left_view       = *dependent_tasks[0]->result();
  std::unique_ptr<cudf::column> bool_mask;

  if (_hash_map_cache && _mark_join) {
    auto hash_map = dynamic_cast<mark_join_interface const*>(_hash_map_cache->hash_map());
    assert(hash_map);
    auto positions = hash_map->compute_positions_list_from_cached_map(_join_type);
    auto position_column =
      std::make_unique<cudf::column>(std::move(*positions), rmm::device_buffer{}, 0);

    bool_mask =
      cudf::make_column_from_scalar(cudf::numeric_scalar<bool>(false), left_view.num_rows());
    detail::set_boolean_mask(bool_mask->mutable_view(), *position_column);
  } else {
    // For left-anti join, since we want to keep rows such that the indices are in *all* position
    // lists, we increment `counts` for each position list, and only keep rows such that the count
    // == the number of position lists.
    std::unique_ptr<cudf::column> counts;

    if (_join_type == join_type_type::left_anti) {
      counts = cudf::make_column_from_scalar(
        cudf::numeric_scalar<int32_t>(0),  // Scalar value to fill with (0 in this case)
        left_view.num_rows()               // Number of rows in the new column
      );
    } else {
      bool_mask =
        cudf::make_column_from_scalar(cudf::numeric_scalar<bool>(false), left_view.num_rows());
    }

    for (std::size_t partition_idx = 1; partition_idx < dependent_tasks.size(); partition_idx++) {
      auto child_table = *dependent_tasks[partition_idx]->result();
      GQE_EXPECTS(child_table.num_columns() == 1,
                  "materialize_join_from_position_lists_task expects position list table to have a "
                  "single column");

      auto position_column = child_table.column(0);

      // Note that we can use `set_boolean_mask` and `increment_counts` because indices from the
      // position lists are unique
      if (_join_type == join_type_type::left_semi) {
        detail::set_boolean_mask(bool_mask->mutable_view(), position_column);
      } else {
        detail::increment_counts(counts->mutable_view(), position_column);
      }
    }

    if (_join_type == join_type_type::left_anti) {
      bool_mask = cudf::binary_operation(counts->view(),
                                         cudf::numeric_scalar<int32_t>(dependent_tasks.size() - 1),
                                         cudf::binary_operator::EQUAL,
                                         cudf::data_type(cudf::type_id::BOOL8));
    }
  }
  auto materialized_result =
    cudf::apply_boolean_mask(left_view.select(_projection_indices), bool_mask->view());

  GQE_LOG_TRACE(
    "Executed materialize_join_from_position_lists task: task_id={}, stage_id={}, input_size={}, "
    "output_size={}.",
    task_id(),
    stage_id(),
    left_view.num_rows(),
    materialized_result->num_rows());
  emit_result(std::move(materialized_result));
  remove_dependencies();
}

}  // namespace gqe
