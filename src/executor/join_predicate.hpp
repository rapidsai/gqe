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

#include <cuda/stream>
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace gqe {

class expression;
class expression_input;

namespace detail {

/** @brief Deduplicated predicate indices relative to each input, in first-reference order. */
struct join_predicate_columns {
  std::vector<cudf::size_type> left;  ///< Indices into the logical left input.

  std::vector<cudf::size_type> right;  ///< Indices into the logical right input.
};

/**
 * @brief Collect and split references from a combined logical join schema.
 *
 * Either input may be unreferenced. Callers enforce join-specific requirements and choose
 * how to report invalid bounds. No expression indices are rewritten.
 *
 * @return Input-relative columns, or `std::nullopt` for negative widths, an overflowing
 * combined width, or an out-of-range reference.
 * @throws std::invalid_argument If the expression contains a null child.
 */
[[nodiscard]] std::optional<join_predicate_columns> try_collect_join_predicate_columns(
  expression const& predicate, cudf::size_type left_width, cudf::size_type right_width);

/**
 * @brief Return whether an expression can be represented by the shared cuDF join-predicate AST.
 *
 * This check creates no device allocations or cuDF AST/scalar objects. It recognizes exactly the
 * expression node, operator, literal, and cast subset accepted by `make_join_predicate`.
 */
[[nodiscard]] bool is_join_predicate_supported(expression const& predicate);

/**
 * @brief Owns a cuDF AST join predicate and every node and scalar it references.
 *
 * Column references in the GQE expression address the combined
 * `[logical_left..., logical_right...]` schema. The resulting cuDF AST addresses two separate
 * tables. Every literal allocation uses the supplied stream and memory resource.
 */
class join_predicate {
 public:
  using owned_object =
    std::variant<std::unique_ptr<cudf::ast::expression>, std::unique_ptr<cudf::scalar>>;

  join_predicate(join_predicate&&) noexcept            = default;
  join_predicate& operator=(join_predicate&&) noexcept = default;

  join_predicate(join_predicate const&)            = delete;
  join_predicate& operator=(join_predicate const&) = delete;

  /**
   * @brief Borrow the root AST expression owned by this predicate.
   *
   * The owning predicate must outlive every cuDF call using this reference. Asynchronous scalar
   * uses must be ordered before the owner's stream-ordered deallocation.
   */
  [[nodiscard]] cudf::ast::expression const& root() const;

 private:
  friend join_predicate make_join_predicate(expression const&,
                                            cudf::size_type,
                                            cuda::stream_ref,
                                            rmm::device_async_resource_ref);
  friend join_predicate make_join_predicate(expression const&,
                                            cudf::size_type,
                                            expression_input const&,
                                            expression_input const&,
                                            cuda::stream_ref,
                                            rmm::device_async_resource_ref);

  join_predicate(cudf::ast::expression const* root, std::vector<owned_object> objects);

  cudf::ast::expression const* _root;
  std::vector<owned_object> _objects;
};

/**
 * @brief Convert a GQE join predicate to a lifetime-owning cuDF AST.
 *
 * The supported expression subset is identified by `is_join_predicate_supported`. Callers supply
 * a well-formed expression in that subset and validate column indices against the input schema.
 * Conversion checks node arity, non-negative indices, and supported operators while traversing
 * the expression. It does not run the support check as a separate pass or validate column types.
 *
 * @param[in] predicate Predicate in combined logical-input coordinates.
 * @param[in] left_num_columns Number of columns in the logical left conditional table.
 * @param[in] stream CUDA stream used for literal-scalar allocations.
 * @param[in] mr Device memory resource used for literal-scalar allocations.
 *
 * @return Owned cuDF AST predicate.
 * @throws std::invalid_argument If `left_num_columns` or a referenced column index is negative.
 * @throws std::logic_error If an expression node, operator, or cast is unsupported, or a checked
 *         node arity is invalid. Native allocation and cuDF exceptions propagate unchanged.
 */
[[nodiscard]] join_predicate make_join_predicate(expression const& predicate,
                                                 cudf::size_type left_num_columns,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);

/**
 * @brief Convert a logical join predicate using independently bound physical inputs.
 *
 * References address the combined logical `[left, right]` schema. The explicit logical-left
 * width determines the side, and that side's binding resolves the side-local physical index.
 * The width includes unmaterialized columns and cannot be inferred from a sparse binding.
 * Inputs may have different row counts and sparse or reordered columns. The GQE expression
 * remains unchanged. Bindings are borrowed only during conversion and are not retained by the
 * resulting AST. Conversion reads no column data. Callers keep column owners alive through the
 * cuDF join that consumes the AST.
 *
 * The expression subset and type-validation responsibilities match the identity-mapping overload.
 * Conversion takes O(N) time and O(N) AST storage for N visited expression nodes, excluding
 * literal payload sizes. Recursion uses O(H) stack space for expression depth H, up to N for a
 * degenerate tree. Column lookup is O(1) using the bindings' checked inverse mappings.
 *
 * @param[in] predicate Predicate in combined logical-input coordinates.
 * @param[in] left_num_columns Full logical-left schema width. Must be nonnegative.
 * @param[in] left Physical left columns bound to side-local logical indices.
 * @param[in] right Physical right columns bound to side-local logical indices.
 * @param[in] stream CUDA stream used for literal-scalar allocations.
 * @param[in] mr Device memory resource used for literal-scalar allocations.
 * @return Owned cuDF AST predicate.
 * @throws std::invalid_argument If the left schema width or a referenced column index is negative.
 * @throws std::out_of_range If a reference is outside the logical schema or absent from its
 * binding.
 * @throws std::logic_error If an expression node, operator, cast, or node arity is unsupported.
 */
[[nodiscard]] join_predicate make_join_predicate(expression const& predicate,
                                                 cudf::size_type left_num_columns,
                                                 expression_input const& left,
                                                 expression_input const& right,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace gqe
