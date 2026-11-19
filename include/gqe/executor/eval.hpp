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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/expression/unary_op.hpp>

#include <cuda/stream>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

/**
 * @brief A reusable mapping from logical expression indices to physical column positions.
 *
 * The ordered selection defines the physical layout. Construction copies the mapping and checks
 * that indices are nonnegative and unique. No input schema width is required.
 *
 * Operators may share a binding across batches. Keep it unchanged and alive while any
 * `expression_input` references it. Materialization must preserve the selected column order.
 * An unchanged binding supports concurrent reads by multiple host threads.
 */
class expression_binding {
 public:
  /** @brief Construct an empty selection. */
  expression_binding() = default;

  /**
   * @brief Construct a binding from logical indices in physical column order.
   *
   * @param[in] physical_to_logical Logical index represented by each physical column.
   * @throws std::invalid_argument If an index is negative or duplicated, or the mapping cannot
   * fit in a cuDF schema.
   */
  explicit expression_binding(std::span<cudf::size_type const> physical_to_logical);

  /** @brief Return the number of selected physical columns. */
  [[nodiscard]] cudf::size_type physical_column_count() const noexcept
  {
    return _physical_column_count;
  }

  /** @brief Return the minimum logical schema width containing the selection. */
  [[nodiscard]] cudf::size_type required_logical_column_count() const noexcept
  {
    return static_cast<cudf::size_type>(_logical_to_physical.size());
  }

  /**
   * @brief Resolve a selected logical column to its physical position.
   *
   * @param[in] logical_index Logical column index to resolve.
   * @throws std::out_of_range If the logical column is absent from the selection.
   */
  [[nodiscard]] cudf::size_type physical_column_index(cudf::size_type logical_index) const;

 private:
  cudf::size_type _physical_column_count{0};
  std::vector<cudf::size_type> _logical_to_physical;
};

/**
 * @brief A physical cuDF table bound to logical expression indices and batch row geometry.
 *
 * An identity input binds physical column p to logical column p without allocating a mapping.
 * An explicitly bound input borrows an `expression_binding` and checks table compatibility.
 * The binding must remain unchanged and outlive the input. Column owners must remain
 * alive until all device work using the input completes.
 *
 * A zero-column input with an explicit binding may retain nonzero logical row geometry for
 * reference-free projection expressions. Table columns must follow the binding's physical order.
 */
class expression_input {
 public:
  /** @brief Bind a table to identical logical and physical column indices. */
  explicit expression_input(cudf::table_view table) noexcept;

  /**
   * @brief Bind a table and its row geometry to an existing expression binding.
   *
   * @param[in] table Physical table consumed by expression evaluation.
   * @param[in] binding Reusable logical-to-physical mapping. Must outlive this input.
   * @param[in] logical_row_count Logical row count, including zero-column input geometry.
   *
   * @throws std::invalid_argument If the physical column count differs from the binding or the
   * row geometry is invalid.
   */
  expression_input(cudf::table_view table,
                   expression_binding const& binding,
                   cudf::size_type logical_row_count);

  /** @brief Reject temporary bindings because the input borrows the mapping. */
  expression_input(cudf::table_view table,
                   expression_binding const&& binding,
                   cudf::size_type logical_row_count) = delete;

  /** @brief Return the physical cuDF table. */
  [[nodiscard]] cudf::table_view table() const noexcept { return _table; }

  /** @brief Return the logical input row count. */
  [[nodiscard]] cudf::size_type row_count() const noexcept { return _logical_row_count; }

  /**
   * @brief Return the physical table index for one logical column.
   *
   * @param[in] logical_index Logical column index to resolve.
   *
   * @throws std::out_of_range If the logical column is absent from the physical table.
   */
  [[nodiscard]] cudf::size_type physical_column_index(cudf::size_type logical_index) const;

 private:
  cudf::table_view _table;
  cudf::size_type _logical_row_count;
  expression_binding const* _binding;  ///< Borrowed mapping, or null for identity inputs.
};

/**
 * @brief Maps a `cudf::{unary|binary}_operator` to its `cudf::ast::ast_operator` equivalent.
 *
 * @param[in] op Input `cudf::{unary|binary}_operator`.
 *
 * @returns The equivalent `cudf::ast::ast_operator`.
 *
 * @throws std::logic_error if there is no equivalent `cudf::ast::ast_operator`.
 */
[[nodiscard]] cudf::ast::ast_operator cudf_to_ast_operator(
  std::variant<cudf::binary_operator, cudf::unary_operator> op);

class expression_evaluator : public expression_visitor {
 public:
  /**
   * @brief Create a `cudf::ast` expression evaluator over logically bound input columns.
   *
   * @param[in] input Physical columns, logical binding, and row geometry. Must outlive the
   * evaluator.
   * @param[in] root_expression Non-owning root expression using logical column indices.
   * @param[in] use_ast_jit Whether AST-compatible expressions use JIT evaluation.
   * @param[in] stream CUDA stream used for all device work.
   * @param[in] result_mr Memory resource used for returned and temporary expression columns.
   *
   * @throws std::invalid_argument If `root_expression` is null.
   * @throws std::out_of_range If the expression references a logical column absent from `input`.
   */
  expression_evaluator(
    expression_input const& input,
    expression const* root_expression,
    bool use_ast_jit                         = false,
    cuda::stream_ref stream                  = cudf::get_default_stream(),
    rmm::device_async_resource_ref result_mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Evaluate an expression on a table.
   *
   * This function uses cudf's AST module to efficiently evaluate expressions.
   * Since `cudf::ast` only implements a subset of expressions that we need,
   * we evaluate unsupported subexpressions separately using a fallback strategy.
   *
   * @param[in] use_like_shift_and If `true`, use shift_and kernel for computing like filter.
   * @param[in] input_override Optional views with the same column order, types, and row count.
   * This permits selection-mask null injection without rebuilding the expression plan.
   * The caller retains the input owners through stream work. Shape and types are checked.
   *
   * @return A pair of [evaluated_result, column_cache]. For a column-reference root,
   * `column_cache` is null and `evaluated_result` borrows storage from `input_override` when
   * supplied, or from the evaluator's input table otherwise. The caller must retain that storage
   * while the result is used and through stream work. For a computed expression, `column_cache`
   * owns the result and must remain alive while `evaluated_result` is used and through stream work.
   *
   * @throws std::invalid_argument If `input_override` differs from the input in row count,
   * column count, or column types.
   */
  [[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>> evaluate(
    bool use_like_shift_and, std::optional<cudf::table_view> input_override = std::nullopt) const;

  /**
   * @brief Try evaluating a single-task, non-JIT Boolean AST on selected rows.
   *
   * False and null selection entries reject rows. Only selected rows with a valid, true
   * expression result survive.
   * Input and selection owners must remain alive through stream work.
   *
   * @param selection A BOOL8 selection mask, optionally nullable, with the same row count as
   * the input.
   *
   * @return An owning, nonnullable BOOL8 keep mask, or `nullptr` when the general evaluator is
   * needed. Expressions excluded by the eligibility checks return before selection validation.
   *
   * @throws std::logic_error If the eligible path receives a non-BOOL8 selection or a
   * mismatched row count.
   * @throws std::runtime_error If nonempty input requires more shared memory for one thread
   * than the device permits per block.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> try_evaluate_masked_filter(
    cudf::column_view selection) const;

  /**
   * @brief Try evaluating a single-task, non-JIT fixed-width AST on selected rows.
   *
   * False and null selection entries reject rows. Rejected rows are null, and selected rows
   * retain the expression's null semantics.
   * Input and selection owners must remain alive through stream work. Callers must retain
   * the upstream selection for subsequent operators.
   *
   * @param selection A BOOL8 selection mask, optionally nullable, with the same row count as
   * the input.
   *
   * @return An owning computed column, or `nullptr` when the general evaluator is
   * needed. Expressions excluded by the eligibility checks return before selection validation.
   *
   * @throws std::logic_error If the eligible path receives a non-BOOL8 selection or a
   * mismatched row count.
   * @throws std::runtime_error If nonempty input requires more shared memory for one thread
   * than the device permits per block.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> try_evaluate_masked_projection(
    cudf::column_view selection) const;

  /**
   * @copydoc gqe::expression_visitor::visit(column_reference_expression const*)
   */
  void visit(column_reference_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<int8_t> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<int32_t> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<int64_t> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<float> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<double> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<numeric::decimal32> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<numeric::decimal64> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<numeric::decimal128> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<std::string> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<cudf::timestamp_D> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(binary_op_expression const*)
   */
  void visit(binary_op_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(if_then_else_expression const*)
   */
  void visit(if_then_else_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(cast_expression const*)
   */
  void visit(cast_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(unary_op_expression const*)
   */
  void visit(unary_op_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(is_null_expression const*)
   */
  void visit(is_null_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(scalar_function_expression const*)
   */
  void visit(scalar_function_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(subquery_expression const*)
   */
  void visit(subquery_expression const* expression) override;

  /**
   * @brief Context infomation required to evaluate an expression.
   */
  struct evaluation_context {
    /**
     * @brief Construct an evaluation context object.
     *
     * @param[in] expr Expression for which the context is to be created.
     */
    evaluation_context(std::unique_ptr<gqe::expression> expr) : gqe_expression{std::move(expr)} {}

    /**
     * @brief Return the string representation of the evaluation context.
     */
    [[nodiscard]] std::string to_string() const noexcept;

    /**
     * @brief Return the cached output type for this context.
     *
     * @note Use this cached value instead of calling
     * `gqe_expression->data_type(_column_types)` on cloned expressions whose
     * column references may have been rewritten to intermediate task columns.
     */
    [[nodiscard]] cudf::data_type data_type() const noexcept { return result_data_type; }

    std::shared_ptr<gqe::expression> gqe_expression;  ///< Expression this context is for.
    std::vector<evaluation_context*>
      child_contexts;  ///< References the context objects of each child expression.
    std::optional<std::unique_ptr<cudf::ast::expression>>
      cudf_ast_expression;  ///< The equivalent `cudf::ast` expression, if available.
    std::vector<std::unique_ptr<cudf::ast::expression>>
      cudf_ast_dependencies;  ///< Any expression the `cudf::ast`expression depends on.
    std::optional<std::unique_ptr<cudf::scalar>>
      cudf_scalar;  ///< Stores the equivalent `cudf::scalar`, in case the expression is a
                    ///< `literal_expression`.
    std::optional<cudf::size_type>
      column_idx;  ///< Holds the index of the target column if the expression is either a
                   ///< `column_reference_expression` or is evaluated as a sub-task.
    cudf::data_type result_data_type;  ///< Cached output type.
  };

 private:
  /** @brief Select the output contract for shared masked evaluation. */
  enum class masked_evaluation_mode { filter, projection };

  /** @brief Share eligibility checks and dispatch between the two masked operations. */
  [[nodiscard]] std::unique_ptr<cudf::column> try_evaluate_masked(
    cudf::column_view selection, masked_evaluation_mode mode) const;

  /**
   * @brief Flag an expression context to be evaluated as a (sub-)task. If the expression is already
   * a task, return the column index of the result column.
   *
   * @param[in] context Evaluation context of the expression
   *
   * @returns Column offset of the materialized evaluation result.
   */
  cudf::size_type dispatch_task(evaluation_context& context) noexcept;

  /**
   * @brief Flag an expression context to be evaluated as a (sub-)task.
   *
   * @tparam Args Types of forwarded ctor parameters for `evaluation_context`.
   *
   * @param[in] expression Pointer to the input expression (key) to be stored in the context map.
   * @param[in] args Ctor parameters of the evaluation_context (payload) to be emplaced in the
   * context map.
   *
   * @returns A pair consisting of a ref to the stored context object along with a bool indicating
   * if the key has been newly inserted into the map.
   */
  template <typename... Args>
  [[nodiscard]] std::pair<evaluation_context&, bool> emplace_context(expression const* expression,
                                                                     Args&&... args) noexcept;

  /**
   * @brief Fetches the associated `evaluation_context` for a given key expression.
   *
   * @param[in] expression Pointer to the expression (key) to search for.
   *
   * @returns A reference to the stored `evaluation_context` object.
   *
   * @throws If the key expression cannot located inside the map.
   */
  [[nodiscard]] evaluation_context& find_context(expression const* expression);

  /**
   * @brief Helper function to create and store the initial `evaluation_context` for a
   * `literal_expression`.
   *
   * @tparam T Underlying C++ type of the literal.
   *
   * @param[in] expression Input `literal_expression`.
   *
   * @returns A pair consisting of a ref to the stored context object along with a bool indicating
   * if the key has been newly inserted into the map.
   */
  template <typename T>
  void create_literal_context(literal_expression<T> const* expression) noexcept;

  expression_input const& _input;              ///< Bound physical columns and logical geometry.
  cudf::table_view _table;                     ///< Non-owning physical table on which to evaluate.
  std::vector<cudf::data_type> _column_types;  ///< Column data types of the input table.
  expression const* _root_expression;          ///< Root of the AST to be evaluated
  bool _use_ast_jit;                           ///< Whether AST tasks use JIT evaluation.
  cuda::stream_ref _stream;                    ///< Stream used for all device work.
  rmm::device_async_resource_ref _result_mr;   ///< Resource for the root result.
  rmm::device_async_resource_ref _temporary_mr;  ///< Resource for evaluator temporaries.
  evaluation_context const* _root_context;       ///< Context that produces the root result.
  std::vector<evaluation_context const*>
    _tasks;                    ///< Sub-expressions which need to be executed in order.
  cudf::size_type _next_task;  ///< Index of the next available output column.

  std::unordered_map<expression const*, evaluation_context>
    _evaluation_contexts;  ///< Storage for the evaluation context of each sub-expression.
};

/** @brief Results from evaluating a list of expressions. */
struct evaluated_expressions {
  std::vector<cudf::column_view> evaluated_result;  ///< Result views in expression order.
  std::vector<std::unique_ptr<cudf::column>>
    owned_result;  ///< Owned columns in expression order; null for column references.
};

/**
 * @brief Evaluate a batch of expressions on a table.
 *
 * @param[in] parameters Optimization parameters for expression evaluation
 * @param[in] input Physical columns, logical binding, and row geometry.
 * @param[in] exprs Expressions to be evaluated.
 * @param[in] stream CUDA stream used for all device work.
 * @param[in] result_mr Memory resource used for returned and temporary expression columns.
 *
 * @note This function uses the `cudf::ast` module to evaluate expressions efficiently.
 * Whenever an AST node is not supported by `cudf::ast`, we compute the result of
 * this sub-expression using a custom evaluation strategy.
 *
 * @return Results in the same order as `exprs`. Computed expressions own their results through
 * `owned_result`; column-reference expressions return views of `input.table()` and have null
 * owners.
 */
[[nodiscard]] evaluated_expressions evaluate_expressions(
  gqe::optimization_parameters const& parameters,
  expression_input const& input,
  std::span<expression const* const> exprs,
  cuda::stream_ref stream                  = cudf::get_default_stream(),
  rmm::device_async_resource_ref result_mr = cudf::get_current_device_resource_ref());

}  // namespace gqe
