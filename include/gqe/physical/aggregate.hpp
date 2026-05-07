/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/aggregation.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Abstract base class for all physical aggregate relations.
 */
class aggregate_relation_base : public relation {
 public:
  /**
   * @brief Construct a physical aggregate relation.
   *
   * @param[in] input Input table to aggregate on.
   * @param[in] subquery_relations Subquery relations that are referenced within the `keys`
   * and/or `values` expressions.
   * @param[in] keys Expressions evaluated on `input` to represent groups. Rows with the same keys
   * will be grouped together. Note that this argument can be an empty vector. In that case, all
   * rows belong to the same group (reductions).
   * @param[in] values List of `(op, expr)` pairs such that each `expr` will be evaluated on `input`
   * and then rows of the evaluated results in the same group will be combined together using `op`.
   * @param[in] condition An optional boolean expression evaluated on `input` to represent the
   * filter condition. Note: That this is currently not supported for pure reductions
   * @param[in] perfect_hashing Whether to use perfect hashing.
   */
  aggregate_relation_base(
    std::shared_ptr<relation> input,
    std::vector<std::shared_ptr<relation>> subquery_relations,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values,
    std::unique_ptr<expression> condition = nullptr,
    bool perfect_hashing                  = false)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _keys(std::move(keys)),
      _values(std::move(values)),
      _condition(std::move(condition)),
      _perfect_hashing(perfect_hashing)
  {
  }

  /**
   * @brief Return the grouping keys.
   */
  std::vector<expression*> keys_unsafe() const { return utility::to_raw_ptrs(_keys); }

  /**
   * @brief Returns the filter condition.
   */
  expression* condition_unsafe() const { return _condition.get(); }

  /**
   * @brief Return the values to be aggregated on.
   */
  std::vector<std::pair<cudf::aggregation::Kind, expression*>> values_unsafe() const
  {
    std::vector<std::pair<cudf::aggregation::Kind, expression*>> values;
    for (auto const& [kind, agg] : _values)
      values.emplace_back(kind, agg.get());
    return values;
  }

  /**
   * @brief Return a boolean indicating whether to use perfect hashing.
   */
  [[nodiscard]] bool perfect_hashing() const noexcept { return _perfect_hashing; }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @brief Print all its member variables as well as its output data types
   */
  [[nodiscard]] std::string print() const;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _values;
  std::unique_ptr<expression> _condition;
  bool _perfect_hashing;
};

/**
 * @brief Apply-concat-apply aggregation.
 *
 * Apply an aggregation on each partition, concatenate the aggregation results into a single
 * partition, and finally apply another aggregation. The result would have one partition only.
 */
class concatenate_aggregate_relation : public aggregate_relation_base {
  using aggregate_relation_base::aggregate_relation_base;

 public:
  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;
};

}  // namespace physical
}  // namespace gqe
