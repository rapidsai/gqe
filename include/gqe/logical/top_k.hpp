/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/logical/relation.hpp>

#include <cstdint>
#include <memory>

namespace gqe {
namespace optimizer {
class optimization_rule;
}  // namespace optimizer
namespace logical {

/**
 * @brief Logical single-key top-K relation.
 *
 * Evaluates `key` on every input row, keeps the best `offset + count` rows according to
 * `column_order` and `null_precedence`, and returns the ordered range
 * `[offset, offset + count)`.
 */
class top_k_relation : public relation {
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new top-K relation object
   *
   * @param input_relation Input relation to select rows from
   * @param subquery_relations Subquery relations referenced by `key`
   * @param key Expression evaluated on `input_relation` to rank rows
   * @param column_order Direction in which to order `key`
   * @param null_precedence Whether a null key is smaller or larger than non-null keys
   * @param offset Number of ordered rows to skip
   * @param count Maximum number of rows to return after `offset`
   * @throws std::invalid_argument If `offset` or `count` is negative
   */
  top_k_relation(std::shared_ptr<relation> input_relation,
                 std::vector<std::shared_ptr<relation>> subquery_relations,
                 std::unique_ptr<expression> key,
                 cudf::order column_order,
                 cudf::null_order null_precedence,
                 int64_t offset,
                 int64_t count);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::top_k; }

  /**
   * @copydoc relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the expression used to rank rows.
   *
   * @note The returned expression does not share ownership. This object must be kept alive for the
   * returned expression to be valid.
   *
   * @return Expression used to rank rows
   */
  [[nodiscard]] expression* key_unsafe() const noexcept { return _key.get(); }

  /**
   * @brief Return the direction in which to order the key
   *
   * @return Key ordering direction
   */
  [[nodiscard]] cudf::order column_order() const noexcept { return _column_order; }

  /**
   * @brief Return whether a null key is smaller or larger than non-null keys
   *
   * @return Null comparison precedence
   */
  [[nodiscard]] cudf::null_order null_precedence() const noexcept { return _null_precedence; }

  /**
   * @brief Return the number of ordered rows to skip
   *
   * @return Number of rows to skip
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; }

  /**
   * @brief Return the maximum number of rows to return after the offset
   *
   * @return Maximum number of rows to return
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::unique_ptr<expression> _key;
  cudf::order _column_order;
  cudf::null_order _null_precedence;
  int64_t _offset;
  int64_t _count;
};

}  // namespace logical
}  // namespace gqe
