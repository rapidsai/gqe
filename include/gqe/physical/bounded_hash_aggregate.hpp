/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/physical/relation.hpp>
#include <gqe/types.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Grouped aggregation into a hash map pre-allocated from a plan-time cardinality bound.
 *
 * The relation lowers to the QEP bounded hash aggregate build and scan tasks: input chunks
 * aggregate directly into one pre-allocated map, and result chunks pipeline back out of it.
 * The plan author asserts `group_count_bound`, an upper bound on the number of distinct
 * non-null group keys, typically derived from an FK/PK relationship. A violated bound is
 * reported as an error at execution time.
 *
 * Computed keys and measures are projected into columns before this relation, so keys and
 * values are column indices and the relation carries no expressions or subqueries. The QEP
 * lowering currently supports one grouping key and one aggregate value.
 */
class bounded_hash_aggregate_relation : public relation {
 public:
  /**
   * @brief One aggregate value: the aggregation kind, its input column index, and the column's
   * declared nullability.
   */
  struct value_type {
    cudf::aggregation::Kind kind;  ///< The aggregation kind.

    std::optional<cudf::size_type> value_index;  ///< Input column of the aggregated value,
                                                 ///< `std::nullopt` exactly for `COUNT_ALL`,
                                                 ///< which reads no column.

    /**
     * @brief Declared nullability of the value column (see `gqe::nullability`). `COUNT_ALL` has
     * no column to declare, so it stays `nullable`.
     */
    nullability value_nullability;
  };

  /**
   * @brief Construct a bounded hash aggregate relation.
   *
   * @param[in] input Input table to aggregate on.
   * @param[in] key_indices Input columns of the grouping keys. Rows with the same keys will
   * be grouped together.
   * @param[in] values The aggregate values computed per group.
   * @param[in] group_count_bound Upper bound on the number of distinct non-null group keys.
   * Must be positive.
   * @param[in] desired_load_factor Optional map load factor in `(0, 1]`. `std::nullopt`
   * selects the operator's default.
   *
   * @throw std::logic_error in the following cases:
   *
   *  - A key or value column index is outside the input's schema.
   *  - A value's `value_index` is set for `COUNT_ALL`, or unset for any other kind.
   */
  bounded_hash_aggregate_relation(std::shared_ptr<relation> input,
                                  std::vector<cudf::size_type> key_indices,
                                  std::vector<value_type> values,
                                  int64_t group_count_bound,
                                  std::optional<double> desired_load_factor = std::nullopt);

  /**
   * @copydoc gqe::physical::relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override
  {
    return relation_type::bounded_hash_aggregate;
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the input column indices of the grouping keys.
   */
  [[nodiscard]] std::vector<cudf::size_type> const& key_indices() const noexcept
  {
    return _key_indices;
  }

  /**
   * @brief Return the aggregate values computed per group.
   */
  [[nodiscard]] std::vector<value_type> const& values() const noexcept { return _values; }

  /**
   * @brief Return the upper bound on the number of distinct non-null group keys.
   */
  [[nodiscard]] int64_t group_count_bound() const noexcept { return _group_count_bound; }

  /**
   * @brief Return the map load factor, or `std::nullopt` for the operator's default.
   */
  [[nodiscard]] std::optional<double> desired_load_factor() const noexcept
  {
    return _desired_load_factor;
  }

 private:
  std::vector<cudf::size_type> _key_indices;
  std::vector<value_type> _values;
  int64_t _group_count_bound;
  std::optional<double> _desired_load_factor;
};

}  // namespace physical
}  // namespace gqe
